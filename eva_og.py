from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import PIL.Image as pil

# 引入你的项目依赖
import networks
from layers import disp_to_depth
from utils import readlines

# === 配置区域 ===
# 你的权重文件夹路径 (请修改这里指向具体的 weights_xx 文件夹)
model_path = "/home/yxy/lgw/monodepth2-master/allrun/run2_100/models/weights_96/" 

# 数据路径
rgb_dir = "/home/yxy/lgw/monodepth2-master/eval_data/rgb/"
gt_dir = "/home/yxy/lgw/monodepth2-master/eval_data/depth_gt/"

# 模型输入分辨率 (必须与训练时一致！)
input_height = 480  
input_width = 640

# 深度限制 (米)
MIN_DEPTH = 1e-3
MAX_DEPTH = 200
# =================

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    Strictly following the formula provided by user.
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-> Using device: {}".format(device))

    # 1. 加载模型结构
    print("-> Loading model from {}".format(model_path))
    
    # 实例化 Encoder (mpvit_small)
    # 注意：这里必须与你训练时的定义一致
    encoder = networks.mpvit_small()
    encoder.num_ch_enc = [64, 128, 216, 288, 288] 
    
    # 实例化 Decoder
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    # 2. 加载权重
    encoder_dict = torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
    decoder_dict = torch.load(os.path.join(model_path, "depth.pth"), map_location=device)

    # 处理可能的 DataParallel 'module.' 前缀
    # 如果保存时没用 .module，这两行不会有影响；如果用了，会去掉前缀以便加载
    encoder_model_dict = encoder.state_dict()
    decoder_model_dict = depth_decoder.state_dict()

    encoder_dict = {k.replace('module.', ''): v for k, v in encoder_dict.items() if k.replace('module.', '') in encoder_model_dict}
    decoder_dict = {k.replace('module.', ''): v for k, v in decoder_dict.items() if k.replace('module.', '') in decoder_model_dict}

    encoder.load_state_dict(encoder_dict)
    depth_decoder.load_state_dict(decoder_dict)

    encoder.to(device)
    encoder.eval()
    depth_decoder.to(device)
    depth_decoder.eval()

    # 3. 准备数据
    # 获取所有 jpg 文件
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
    
    print("-> Found {} images in {}".format(len(rgb_files), rgb_dir))

    errors = []
    ratios = []

    # 图像预处理 (ToTensor)
    to_tensor = transforms.ToTensor()

    print("-> Computing predictions and evaluating...")
    
    with torch.no_grad():
        for i, file_name in enumerate(rgb_files):
            # === 读取 RGB ===
            rgb_path = os.path.join(rgb_dir, file_name)
            input_image = pil.open(rgb_path).convert('RGB')
            original_width, original_height = input_image.size
            
            # Resize 到模型输入大小
            input_image_resized = input_image.resize((input_width, input_height), pil.LANCZOS)
            input_tensor = to_tensor(input_image_resized).unsqueeze(0).to(device)

            # === 模型推理 ===
            features = encoder(input_tensor)
            outputs = depth_decoder(features)
            
            # 获取最高分辨率的视差图 (disp scale 0)
            pred_disp = outputs[("disp", 0)]
            pred_disp, _ = disp_to_depth(pred_disp, MIN_DEPTH, MAX_DEPTH)
            pred_disp = pred_disp.cpu().numpy()[0, 0] # shape: (H_net, W_net)

            # === 读取 GT ===
            # 假设 GT 文件名与 RGB 一致，后缀为 png
            base_name = os.path.splitext(file_name)[0]
            gt_file_name = base_name + ".png"
            gt_path = os.path.join(gt_dir, gt_file_name)

            if not os.path.exists(gt_path):
                print(f"Warning: GT file not found for {file_name}, skipping.")
                continue

            # 读取 16-bit PNG (单位: mm)
            gt_depth = cv2.imread(gt_path, -1) 
            
            if gt_depth is None:
                print(f"Warning: Could not load GT {gt_path}, skipping.")
                continue
            
            gt_height, gt_width = gt_depth.shape[:2]

            # === 转换单位 ===
            # GT (mm) -> GT (meters)
            # 因为 Monodepth2 默认是在米制下评估的，RMSE 等指标如果不转单位会非常大
            gt_depth = gt_depth.astype(np.float32) / 1000.0

            # === 后处理与对齐 ===
            # 将预测的视差图 resize 回原始 GT 的分辨率
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            # 生成 Mask
            # 1. GT 必须有值
            mask = gt_depth > 0
            # 2. (可选) 限制在有效深度范围内 [0.001, 80] 米
            mask = np.logical_and(mask, gt_depth > MIN_DEPTH)
            mask = np.logical_and(mask, gt_depth < MAX_DEPTH)

            # 过滤无效点
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            if len(gt_depth) == 0:
                print(f"Warning: No valid depth points for {file_name}, skipping.")
                continue

            # === Median Scaling (中位数缩放) ===
            # 单目深度估计没有绝对尺度，必须用 GT 进行对齐
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

            # 限制预测值的范围
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            # === 计算指标 ===
            # 使用你提供的公式函数
            error = compute_errors(gt_depth, pred_depth)
            errors.append(error)

            if (i + 1) % 50 == 0:
                print(f"   Processed {i+1}/{len(rgb_files)} images")

    # === 汇总输出 ===
    if len(errors) > 0:
        mean_errors = np.array(errors).mean(0)
        
        # 打印 Scaling 统计
        ratios = np.array(ratios)
        med = np.median(ratios)
        print("\nScaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        print("\nResults:")
        print(("{:>10} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 10.3f} " * 7).format(*mean_errors.tolist()) + "\\\\")
        
        print("\nNote: RMSE and Sq Rel are computed in METERS (assuming input GT was mm).")
    else:
        print("No valid data processed.")

if __name__ == "__main__":
    evaluate()

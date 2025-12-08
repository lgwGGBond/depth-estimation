from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# 引入你的项目依赖
try:
    import networks
    from layers import disp_to_depth
    from utils import readlines
except ImportError:
    print("错误：无法导入项目依赖 (networks, layers, utils)。请确保你在项目根目录下运行。")
    exit()

# === 配置区域 ===
# 你的权重文件夹路径
model_path = "/home/yxy/lgw/monodepth2-master/allrun/run2_100/models/weights_96/" 

# 数据路径
rgb_dir = "/home/yxy/lgw/monodepth2-master/eval_data/rgb/"
gt_dir = "/home/yxy/lgw/monodepth2-master/eval_data/depth_gt/"

# 结果保存目录 (新增)
OUTPUT_DIR = "/home/yxy/lgw/effect_img/monovit/"

# 模型输入分辨率 (必须与训练时一致！)
input_height = 480  
input_width = 640

# 深度限制 (米)
MIN_DEPTH = 1.0
MAX_DEPTH = 1.2 # 通常 KITTI/Outdoor 设置为 80，如果你确实是 200 请改回 200
# =================

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
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
    encoder = networks.mpvit_small()
    encoder.num_ch_enc = [64, 128, 216, 288, 288] 
    
    # 实例化 Decoder
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    # 2. 加载权重
    enc_path = os.path.join(model_path, "encoder.pth")
    dec_path = os.path.join(model_path, "depth.pth")
    
    if not os.path.exists(enc_path) or not os.path.exists(dec_path):
        print(f"错误: 找不到模型文件 {enc_path} 或 {dec_path}")
        return

    encoder_dict = torch.load(enc_path, map_location=device)
    decoder_dict = torch.load(dec_path, map_location=device)

    # 处理 DataParallel 'module.' 前缀
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
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print("-> Found {} images in {}".format(len(rgb_files), rgb_dir))
    print(f"-> Saving visualization to {OUTPUT_DIR}")

    errors = []
    ratios = []

    to_tensor = transforms.ToTensor()

    print("-> Computing predictions and evaluating...")
    
    with torch.no_grad():
        for i, file_name in enumerate(rgb_files):
            # === 读取 RGB ===
            rgb_path = os.path.join(rgb_dir, file_name)
            input_image = pil.open(rgb_path).convert('RGB')
            # original_width, original_height = input_image.size
            
            # Resize 到模型输入大小
            input_image_resized = input_image.resize((input_width, input_height), pil.LANCZOS)
            input_tensor = to_tensor(input_image_resized).unsqueeze(0).to(device)

            # === 模型推理 ===
            features = encoder(input_tensor)
            outputs = depth_decoder(features)
            
            # 获取最高分辨率的视差图
            pred_disp = outputs[("disp", 0)]
            pred_disp, _ = disp_to_depth(pred_disp, MIN_DEPTH, MAX_DEPTH)
            pred_disp = pred_disp.cpu().numpy()[0, 0] # shape: (H_net, W_net)

            # === 读取 GT ===
            base_name = os.path.splitext(file_name)[0]
            
            # 尝试寻找不同后缀的 GT
            gt_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                potential_path = os.path.join(gt_dir, base_name + ext)
                if os.path.exists(potential_path):
                    gt_path = potential_path
                    break

            if gt_path is None:
                # print(f"Warning: GT file not found for {file_name}, skipping.")
                continue

            # 读取 16-bit PNG (单位: mm) 或其他格式
            gt_depth = cv2.imread(gt_path, -1) 
            
            if gt_depth is None:
                print(f"Warning: Could not load GT {gt_path}, skipping.")
                continue
            
            gt_height, gt_width = gt_depth.shape[:2]

            # === 转换单位 ===
            # 假设 GT 是毫米，转为米
            gt_depth = gt_depth.astype(np.float32) / 1000.0

            # === 后处理与对齐 ===
            # 将预测的视差图 resize 回原始 GT 的分辨率
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            
            # ==============================================================================
            # [新增功能] 保存可视化的深度图 (使用 magma colormap)
            # ==============================================================================
            try:
                # 1. 获取用于可视化的 numpy 数组
                disp_vis = pred_disp.copy()

                # 2. 按照截图逻辑进行归一化 (使用 Percentile)
                # 90%分位数作为最大值，1%分位数作为最小值
                vmax = np.percentile(disp_vis, 90)
                vmin = np.percentile(disp_vis, 1)
                normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

                # 3. 应用 magma 颜色映射
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                
                # 4. 转换并保存
                # mapper.to_rgba 输出 0-1 的 float，需要乘 255 转 uint8
                colormapped_im = (mapper.to_rgba(disp_vis)[:, :, :3] * 255).astype(np.uint8)
                
                # 保存为 png
                save_name = f"{base_name}.png"
                save_path = os.path.join(OUTPUT_DIR, save_name)
                pil.fromarray(colormapped_im).save(save_path)

            except Exception as e_vis:
                print(f"[警告] 保存可视化图片失败 {file_name}: {e_vis}")
            # ==============================================================================

            # 视差转深度
            pred_depth = 1 / pred_disp

            # 生成 Mask
            mask = gt_depth > 0
            mask = np.logical_and(mask, gt_depth > MIN_DEPTH)
            mask = np.logical_and(mask, gt_depth < MAX_DEPTH)

            # 过滤无效点
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            if len(gt_depth) == 0:
                continue

            # === Median Scaling ===
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            # === 计算指标 ===
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
        print("\n" + "="*50)
        print(f"Evaluation Done! Processed {len(errors)} images.")
        print(f"Images saved to: {OUTPUT_DIR}")
        print("Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
        print("="*50)

        print(("{:>10} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 10.3f} " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("="*50)
    else:
        print("No valid data processed.")

if __name__ == "__main__":
    evaluate()

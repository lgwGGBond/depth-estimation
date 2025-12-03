这份 Markdown 文档为你总结了使用 Monodepth2 (配合 MPViT Encoder) 进行训练和自定义评测的全流程。你可以直接将以下内容保存为 `README.md` 或发布到你的技术博客中。

---

# Monodepth2 (MPViT版) 训练与评测实战指南

本文档总结了基于 Monodepth2 框架（替换 Encoder 为 MPViT-Small）进行单目深度估计模型训练及编写自定义评测脚本的实战经验。

## 1. 训练流程 (Training)

在开始训练前，针对显存碎片化问题和多卡训练环境，需要进行特定的环境配置。

### 启动命令

```bash
# 解决 PyTorch 显存碎片化导致的 OOM 问题
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 指定使用 GPU 4, 5 进行训练
CUDA_VISIBLE_DEVICES=4,5 python train.py \
  --log_dir /home/yxy/lgw/monodepth2-master/ \
  --model_name mono1 \
  --dataset custom \
  --data_path /home/yxy/lgw/split_data \
  --split custom \
  --height 480 \
  --width 640 \
  --batch_size 8 \
  --num_workers 12 \
  --num_epochs 20
```

### 关键参数说明
*   **显存优化**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 是在高版本 PyTorch 中解决显存不足（尽管显存看起来还够）的关键环境变量。
*   **多卡训练**: 使用 `CUDA_VISIBLE_DEVICES` 指定多张卡，代码内部通常会使用 `DataParallel`，这会导致保存权重的 Key 带有 `module.` 前缀（评测时需注意）。
*   **分辨率**: `--height 480 --width 640`，请确保显存足够，且评测时必须保持一致。
*   **数据**: 使用 `--dataset custom` 和自定义的数据分割 `--split custom`。

---

## 2. 评测脚本详解 (Evaluation)

由于 Monodepth2 原版评测脚本通常针对 KITTI 数据集，针对自定义数据集（如 PNG 格式的 Depth GT），我们需要重写评测逻辑。

### 2.1 依赖与配置

```python
from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import PIL.Image as pil

# 引入项目依赖 (确保 networks, layers, utils 在 python path 下)
import networks
from layers import disp_to_depth

# === 配置区域 ===
# 权重路径 (指向包含 encoder.pth 和 depth.pth 的文件夹)
model_path = "/home/yxy/lgw/monodepth2-master/mono2/models/weights_29/"

# 数据路径
rgb_dir = "/home/yxy/lgw/monodepth2-master/eval_data/rgb/"
gt_dir = "/home/yxy/lgw/monodepth2-master/eval_data/depth_gt/"

# 模型输入分辨率 (必须与训练时完全一致！)
input_height = 480
input_width = 640

# 深度限制 (米) - 用于过滤无效值和天空
MIN_DEPTH = 1e-3
MAX_DEPTH = 200
```

### 2.2 定义评价指标

严格遵循标准深度估计评价公式（AbsRel, SqRel, RMSE, $\delta < 1.25^n$）。

```python
def compute_errors(gt, pred):
    """
    计算预测值与真实值之间的误差指标
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
```

### 2.3 模型加载与权重处理 (核心坑点)

训练时使用了多卡 (`DataParallel`)，导致保存的权重字典 Key 中包含 `module.` 前缀。单卡推理时必须去除该前缀，否则会报错 `Missing key(s)`。

```python
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Using device: {device}")
    print(f"-> Loading model from {model_path}")

    # 1. 实例化模型结构
    # Encoder: MPViT Small (通道数需对应)
    encoder = networks.mpvit_small()
    encoder.num_ch_enc = [64, 128, 216, 288, 288] 
    
    # Decoder
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    # 2. 加载权重文件
    encoder_dict = torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
    decoder_dict = torch.load(os.path.join(model_path, "depth.pth"), map_location=device)

    # 3. 清洗权重 Key (去除 'module.' 前缀)
    encoder_model_dict = encoder.state_dict()
    decoder_model_dict = depth_decoder.state_dict()

    encoder_dict = {k.replace('module.', ''): v for k, v in encoder_dict.items() 
                    if k.replace('module.', '') in encoder_model_dict}
    decoder_dict = {k.replace('module.', ''): v for k, v in decoder_dict.items() 
                    if k.replace('module.', '') in decoder_model_dict}

    encoder.load_state_dict(encoder_dict)
    depth_decoder.load_state_dict(decoder_dict)

    encoder.to(device)
    encoder.eval()
    depth_decoder.to(device)
    depth_decoder.eval()
    
    # ... (接下文推理循环)
```

### 2.4 推理循环与数据对齐

单目深度估计存在**尺度不确定性 (Scale Ambiguity)**，因此在评测时通常使用 **Median Scaling**（中位数对齐）将预测值对齐到真实值的尺度。

**主要步骤：**
1.  **预处理**: RGB 图片 Resize 到 `(480, 640)` 并转 Tensor。
2.  **推理**: 获得视差图 (Disp)，转换为深度图，并 Resize 回原始 GT 的分辨率。
3.  **单位转换**: 这里的 GT 是 16-bit PNG (单位 mm)，需要除以 1000 转换为米。
4.  **Mask**: 过滤掉 GT 为 0 或超出 `[MIN_DEPTH, MAX_DEPTH]` 的无效像素。
5.  **Scaling**: `pred_depth *= np.median(gt) / np.median(pred)`。

```python
    # ... (接上文模型加载)
    
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
    print(f"-> Found {len(rgb_files)} images")

    errors = []
    ratios = []
    to_tensor = transforms.ToTensor()

    print("-> Computing predictions and evaluating...")

    with torch.no_grad():
        for i, file_name in enumerate(rgb_files):
            # === 1. 读取与预处理 ===
            rgb_path = os.path.join(rgb_dir, file_name)
            input_image = pil.open(rgb_path).convert('RGB')
            # 必须使用 LANCZOS 或 BILINEAR 缩放
            input_image_resized = input_image.resize((input_width, input_height), pil.LANCZOS)
            input_tensor = to_tensor(input_image_resized).unsqueeze(0).to(device)

            # === 2. 模型推理 ===
            features = encoder(input_tensor)
            outputs = depth_decoder(features)
            
            # 取出 scale 0 的视差
            pred_disp = outputs[("disp", 0)]
            pred_disp, _ = disp_to_depth(pred_disp, MIN_DEPTH, MAX_DEPTH)
            pred_disp = pred_disp.cpu().numpy()[0, 0]

            # === 3. 读取 GT 并对齐分辨率 ===
            base_name = os.path.splitext(file_name)[0]
            gt_path = os.path.join(gt_dir, base_name + ".png")

            if not os.path.exists(gt_path):
                continue

            # 读取 16-bit PNG (单位通常为 mm)
            gt_depth = cv2.imread(gt_path, -1)
            if gt_depth is None: continue

            gt_height, gt_width = gt_depth.shape[:2]
            
            # 将 GT 从毫米转换为米
            gt_depth = gt_depth.astype(np.float32) / 1000.0

            # 将预测视差 Resize 到 GT 的原始分辨率
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            # === 4. 生成有效掩码 (Mask) ===
            mask = gt_depth > 0
            mask = np.logical_and(mask, gt_depth > MIN_DEPTH)
            mask = np.logical_and(mask, gt_depth < MAX_DEPTH)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            if len(gt_depth) == 0: continue

            # === 5. 中位数缩放 (Median Scaling) ===
            # 单目模型没有绝对尺度，必须对齐
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

            # 截断预测值范围
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            # === 6. 计算单张图片指标 ===
            error = compute_errors(gt_depth, pred_depth)
            errors.append(error)

            if (i + 1) % 50 == 0:
                print(f"   Processed {i+1}/{len(rgb_files)}")

    # === 结果汇总 ===
    if len(errors) > 0:
        mean_errors = np.array(errors).mean(0)
        ratios = np.array(ratios)
        med = np.median(ratios)
        print("\nScaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
        print("\nResults:")
        print(("{:>10} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 10.3f} " * 7).format(*mean_errors.tolist()) + "\\\\")
    else:
        print("No valid data processed.")

if __name__ == "__main__":
    evaluate()
```

## 3. 经验总结 (Key Takeaways)

1.  **输入尺寸一致性**: 评测时 `input_image.resize` 的目标尺寸 `(input_width, input_height)` 必须严格等于训练时的参数，否则 Encoder 输出的特征图尺寸与 Decoder 不匹配会报错，或者推理结果极差。
2.  **单位换算**: 很多深度数据集（如 TUM, KITTI-Raw png, Custom）的 16-bit PNG 默认单位是**毫米**。Monodepth2 的内部计算和 RMSE 指标通常基于**米**。务必执行 `/ 1000.0` 操作。
3.  **DataParallel 权重加载**: 直接 `torch.load` 多卡训练的权重会带有 `module.` 前缀，导致模型加载失败。使用字典推导式 `{k.replace('module.', ''): v ...}` 是最通用的解决方案。
4.  **Median Scaling**: 自监督单目深度估计无法恢复绝对尺度。如果你没有使用双目或 LiDAR 进行尺度监督，必须在评测代码中引入 `ratio = median(gt) / median(pred)` 进行对齐，否则 RMSE 和 AbsRel 会非常大且无意义。

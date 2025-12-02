
# 自制单目深度估计数据集 (KITTI-Style) 准备与部署教程

本文档提供了一个系统性的流程，用于将无标签的连续视频数据转化为适用于自监督单目深度估计（如 Monodepth2、MonoViT）模型的标准数据集格式。此流程涵盖了数据切分、索引生成以及自定义数据加载器的实现，以确保学术研究的可复现性和数据集的兼容性。

## 目录

1.  数据预处理：视频切分与帧提取
2.  数据集索引生成：划分训练集与验证集
3.  核心修改：实现自定义数据加载器 (`CustomDataset`)
4.  训练部署与运行指南

---

## 1. 数据预处理：视频切分与帧提取

自监督单目深度估计模型依赖于连续的视频序列作为输入，以利用时间一致性进行自监督信号构建。此步骤的目标是将原始视频文件进行时间采样，并将连续帧按结构化目录存储。

### 1.1 脚本配置与路径定义 (`split_video.py`)

| 配置项 | 描述 |
| :--- | :--- |
| `source_folder_name` | 存放原始视频的路径。 |
| `output_root` | 切分后的图像数据集根目录 (`project/data/`)。 |
| `STEP` | 时间采样间隔。设置为 $N$，则表示每隔 $N$ 帧提取一帧，用于降低时间冗余。|

### 1.2 Python 实现：视频帧提取

以下脚本使用 `cv2` 库执行视频读取和图像帧提取，并确保输出的文件名格式符合深度学习框架的要求（六位零填充）。

```python
import cv2
import os
import glob
import sys

# ================= 配置区 =================
# 存放视频的源文件夹 (请根据实际情况配置)
source_folder_name = "/Users/lgw/Desktop/mk_data/trainset" 
# 存放切分后图片的根目录 (此路径将作为 --data_path 的输入)
output_root = "/Users/lgw/Desktop/mk_data/split_data"
# 采样步长
STEP = 3 
# =======================================

print(f"=== 开始运行视频切分脚本 ===")
# ... [路径检查与视频查找代码略] ...

# 核心处理逻辑
# 确保源路径已调整为您的实际运行环境
# source_path = os.path.join(current_dir, source_folder_name) 

# ... [假设 video_files 已成功获取] ...

if not os.path.exists(output_root):
    os.makedirs(output_root)

for i, video_path in enumerate(video_files):
    # ... [OpenCV 视频读取及诊断代码略] ...
    
    # 准备保存目录：使用零填充的文件夹名，例如 video_01
    folder_name = f"video_{i+1:02d}"
    save_dir = os.path.join(output_root, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    read_count = 0
    save_count = 0
    
    # 帧提取与采样
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if read_count % STEP == 0:
            # 帧文件名格式: 000000.jpg, 000001.jpg, ...
            filename = os.path.join(save_dir, "{:06d}.jpg".format(save_count))
            cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            save_count += 1
            
        read_count += 1
        
    cap.release()
    print(f"\n   -> 保存完成: 共 {save_count} 张图片。")

print("\n=== 数据切分完成 ===")
```

### 1.3 预期输出结构

切分后的数据集应遵循如下的层级结构：

```
{output_root}/
├── video_01/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── video_02/
│   ├── 000000.jpg
│   └── ...
└── ...
```

---

## 2. 数据集索引生成：划分训练集与验证集

此步骤旨在生成用于训练和验证的数据索引文件 (`train_files.txt`, `val_files.txt`)，这些文件指导数据加载器获取正确的连续帧。

### 2.1 索引文件格式

每个索引文件行格式：`<序列文件夹名> <帧ID> <左右标识>`。

*   **帧ID：** 对应于图像文件名（不含前导零）。
*   **左右标识 (`l`)：** 在单目自监督场景中，我们默认使用 `'l'` 标识符以兼容 Monodepth2/KITTI 数据集的结构。

### 2.2 Python 实现：生成 Splits (`generate_splits.py`)

我们采用时间序列划分：每个视频序列的前 90% 用于训练，后 10% 用于验证。为确保上下文完整性（$t-1, t, t+1$），我们排除序列的首帧（索引 0）和尾帧。

```python
import os
import random

# ================= 配置区 =================
DATA_PATH = "/Users/lgw/Desktop/mk_data/split_data" 
SPLIT_OUTPUT_DIR = "/Users/lgw/Desktop/mk_data/splits/custom"
# =======================================

if not os.path.exists(SPLIT_OUTPUT_DIR):
    os.makedirs(SPLIT_OUTPUT_DIR)

print(f"正在扫描数据: {DATA_PATH} ...")

train_lines = []
val_lines = []
folders = sorted([f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))])

for folder in folders:
    # ... [文件获取及数量检查代码略] ...
    
    split_point = int(len(files) * 0.9)
    
    # 训练集: 索引 [1, split_point - 1]
    for i in range(1, split_point):
        train_lines.append(f"{folder} {i} l\n")
        
    # 验证集: 索引 [split_point, len(files) - 2]
    for i in range(split_point, len(files) - 1):
        val_lines.append(f"{folder} {i} l\n")

random.shuffle(train_lines)

# 写入文件
with open(os.path.join(SPLIT_OUTPUT_DIR, "train_files.txt"), "w") as f:
    f.writelines(train_lines)
    
with open(os.path.join(SPLIT_OUTPUT_DIR, "val_files.txt"), "w") as f:
    f.writelines(val_lines)

print("\n索引文件生成完毕。")
print(f"文件保存在: {os.path.abspath(SPLIT_OUTPUT_DIR)}")
```

---

## 3. 核心修改：实现自定义数据加载器 (`CustomDataset`)

为了使训练框架能够正确解析数据路径并应用正确的相机内参，必须在框架的 `datasets` 模块中实现一个自定义类。

**文件路径：** `{project}/datasets/custom_dataset.py`

### 3.1 相机内参的归一化与恢复

数据加载器的核心在于正确处理相机内参矩阵 $K$。如果您的内参是归一化格式（$[0, 1]$ 范围内），需要在加载时根据当前图像的实际分辨率进行缩放恢复。

**请根据您的相机标定结果修改 `self.K` 和 `self.full_res_shape`。**

### 3.2 自定义数据集代码

```python
# 文件路径: {project}/datasets/custom_dataset.py
from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset

class CustomDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(CustomDataset, self).__init__(*args, **kwargs)

        # ==========================================
        # 1. 归一化内参矩阵 K (必须精确)
        # 示例: 0.93 * W, 1.25 * H
        # ==========================================
        self.K = np.array([[0.93, 0, 0.596, 0],
                           [0, 1.25, 0.687, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # 2. 原始视频/图像分辨率 (W, H)
        self.full_res_shape = (640, 480) 

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        # 实现图像加载逻辑
        path = self.get_image_path(folder, frame_index, side)
        with open(path, 'rb') as f:
            color = pil.open(f).convert('RGB')
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_image_path(self, folder, frame_index, side):
        # 构造六位零填充的图像文件名
        f_str = "{:06d}.jpg".format(frame_index)
        image_path = os.path.join(self.data_path, folder, f_str)
        return image_path

    def get_intrinsic_matrix(self, width, height):
        """将归一化 K 矩阵根据实际加载分辨率 (width, height) 恢复为像素 K 矩阵。"""
        K = np.eye(4, dtype=np.float32)
        
        # fx = K[0, 0] * width, cx = K[0, 2] * width
        K[0, 0] = self.K[0, 0] * width
        K[0, 2] = self.K[0, 2] * width
        
        # fy = K[1, 1] * height, cy = K[1, 2] * height
        K[1, 1] = self.K[1, 1] * height
        K[1, 2] = self.K[1, 2] * height

        return K[:3, :3]
```

---

## 4. 训练部署与运行指南

完成上述数据和代码准备后，即可启动自监督训练过程。请确保您的训练环境（如 PyTorch、CUDA）已正确配置。

### 4.1 训练参数解析

以下部署命令针对多 GPU 环境和高性能训练进行了优化：

*   `--dataset custom`: 指定使用第 3 节实现的 `CustomDataset` 类。
*   `--data_path /home/yxy/lgw/split_data`: 指定第 1 节生成的图像数据根目录。
*   `--split custom`: 指定使用第 2 节生成的 `splits/custom/` 索引文件。
*   `--height 480 --width 640`: 指定模型输入的分辨率，应与 `CustomDataset` 中的 `full_res_shape` 保持一致或适配。
*   `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: PyTorch 内存分配配置，有助于优化显存碎片化问题。

### 4.2 训练部署命令

请在项目根目录下执行以下命令：

```bash
# 内存优化配置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 多 GPU 训练启动命令 (使用 GPU 4 和 5)
CUDA_VISIBLE_DEVICES=4,5 python train.py \
  --model_name monovit_custom_train \
  --dataset custom \
  --data_path /home/yxy/lgw/split_data \
  --split custom \
  --height 480 \
  --width 640 \
  --batch_size 8 \
  --num_workers 12 \
  --num_epochs 20
```

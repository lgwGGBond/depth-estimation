# Litemono
---
## Train
>CUDA_VISIBLE_DEVICES=4,5 python train.py   --model_name "run1_8m_multi_gpu"   --model "lite-mono-8m"   --log_dir "/home/yxy/lgw/Lite-Mono/allrun"   --data_path "/home/yxy/lgw/split_data/"   --dataset "custom"   --split "custom"   --height 480   --width 640   --batch_size 24   --num_epochs 30   --lr 0.0001 5e-6 31 0.0001 1e-5 31   --mypretrain "/home/yxy/lgw/Lite-Mono/weights/lite_mono_8m_imagenet.pth"

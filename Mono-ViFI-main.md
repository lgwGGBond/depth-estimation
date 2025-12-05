# Mono-ViFI-main

## Train
>CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 train.py -c configs/resnet18/ResNet18_Custom.txt

##Tensorboard
tensorboard --logdir /home/yxy/lgw/Mono-ViFI-main/logs_custom1/ResNet18_Custom/ --port 6006


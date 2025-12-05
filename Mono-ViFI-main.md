# Mono-ViFI-main

## Train
'''Termial
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 train.py -c configs/resnet18/ResNet18_Custom.txt
'''

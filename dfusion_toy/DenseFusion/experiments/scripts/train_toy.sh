#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

python3 ./tools/train_duplicate.py --dataset toy\
  --dataset_root '/run/user/1000/gvfs/smb-share:server=149.201.37.169,share=anwar/densfusion_download_dataset/datasets/dataset_toy_airplane/dataset'\
  --resume_posenet 'pose_model_11_0.022067401026530813.pth'\
  --start_epoch 10
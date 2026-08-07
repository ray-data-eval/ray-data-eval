#!/bin/bash

# Install NVIDIA driver
sudo apt install nvidia-headless-535-server nvidia-utils-535-server -y
sudo reboot

# Aws
aws configure

# Copy Data
cd ~/
aws s3 cp s3://ray-data-eval-us-west-2/imagenet/imagenet-object-localization-challenge.zip .
unzip imagenet-object-localization-challenge.zip
cd ~/ILSVRC/Data/CLS-LOC/val

# get script from soumith and run; this script creates all class directories and moves images into corresponding directories
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash


# Git clone
git clone https://github.com/pytorch/examples.git
git clone https://github.com/franklsf95/ray-data-eval.git

# Conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

conda create -n mlperf
conda init bash
bash

conda activate mlperf

# Install Pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Run
cd ~/examples/imagenet
python pytorch_e2e_training.py -a resnet50 -b 512 ~/ILSVRC/Data/CLS-LOC > training_v100_4.out 2>&1 &

# Worker num_workers * num_gpus.
python pytorch_e2e_training.py -a resnet50 -b 512 --workers 16 --dist-backend 'nccl' --dist-url 'tcp://127.0.0.1:8080' --multiprocessing-distributed --world-size 1 --rank 0 ~/ILSVRC/Data/CLS-LOC > training_v100_4.out 2>&1 &

# Ray Data
python ray_data_e2e_training.py -a resnet50 -b 128 ~/ILSVRC/Data/CLS-LOC > ray_data_training_v100_1.out 2>&1 &
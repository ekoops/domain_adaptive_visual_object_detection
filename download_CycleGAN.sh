#!/usr/bin/env bash

# The repository is a customize CycleGAN PyTorch implementation
# based on the CycleGAN PyTorch implementation provided in
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
git clone https://github.com/ekoops/pytorch-CycleGAN-and-pix2pix.git
mv pytorch-CycleGAN-and-pix2pix CycleGAN

# installing the CycleGAN implementation requirements
pip install -r CycleGAN/requirements.txt

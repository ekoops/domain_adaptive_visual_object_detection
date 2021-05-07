#!/usr/bin/env bash

# The repository is a customize AdaIN PyTorch implementation
# based on the AdaIN PyTorch implementation provided in
# https://github.com/naoto0804/pytorch-AdaIN
git clone https://github.com/ekoops/pytorch-AdaIN.git
mv pytorch-AdaIN AdaIN

# installing the AdaIN implementation requirements
pip install -r AdaIN/requirements.txt

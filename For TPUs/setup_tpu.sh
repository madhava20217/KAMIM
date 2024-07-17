#!/bin/sh

# Miniconda setup
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

# restart shell for conda
source ~/.bashrc

# creating environment
conda create -n tpu_3.10 python==3.10 -y
conda activate tpu_3.10
pip install torch~=2.2.0 torch_xla[tpu]~=2.2.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html
pip install -r requirements.txt

# logging into wandb
wandb login


# Only support for linux based systems 

1. Install Anaconda
curl -O https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh

Restart terminal
source ~/.bashrc

clone repository
git clone https://github.com/nileshlin/testing_123.git

Download Vila3.5B

git lfs install
git clone https://huggingface.co/Efficient-Large-Model/VILA1.5-3b


Create Conda environment:
conda create -n vila python=3.10 -y

Activate virtualenv

conda activate vila

conda install -c nvidia cuda-toolkit -y





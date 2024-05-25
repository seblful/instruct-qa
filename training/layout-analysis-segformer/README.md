RUNPOD

pip install gdown
gdown https://drive.google.com/uc?id=1pSGZGcYfl1skckHOqfrzP89Vt8RMYIfW
apt-get update
apt-get install unzip

mkdir data
mkdir modules
unzip dataset.zip -d data

pip install --upgrade pip
pip install transformers
pip install evaluate
pip install pytorch-lightning
pip install tensorboard

modules: module.py, trainer.py
train.py

python train.py --num_epochs 50 --image_side 1024 --batch_size 4 --num_workers 2

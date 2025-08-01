source ~/.bashrc
conda activate swimAD

python train.py --cfg cfgs/model_yolo11l_v20250728_0724_0630_0623+0623val+0630val.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0728_0724_0630_0623+0623+0630_bs64 --device 0,3 --batch 64
python train.py --cfg cfgs/model_yolo11l_v20250728_0724_0630_0623_0604.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0728_0724_0630_0623_0604+0604_bs64 --device 0,3 --batch 64
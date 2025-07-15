source ~/.bashrc
conda activate swimAD

python train.py --cfg cfgs/model_yolo11l_v20250630_0623_0604.yaml --ckpt ../ckpts/yolo11s.pt --name 11S_v0630_0623_0604+0604_bs64_crop --device 1,2 --batch 64
python train.py --cfg cfgs/model_yolo11l_v20250630_0623_0604.yaml --ckpt ../ckpts/yolo11m.pt --name 11M_v0630_0623_0604+0604_bs64_crop --device 1,2 --batch 64
python train.py --cfg cfgs/model_yolo11l_v20250630_0623_0604.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0630_0623_0604+0604_bs64_crop --device 1,2 --batch 64

python train.py --cfg cfgs/model_yolo11l_v20250630_0623.yaml --ckpt ../ckpts/yolo11s.pt --name 11S_v0630_0623+0604_bs64_crop --device 1,2 --batch 64
python train.py --cfg cfgs/model_yolo11l_v20250630_0623.yaml --ckpt ../ckpts/yolo11m.pt --name 11M_v0630_0623+0604_bs64_crop --device 1,2 --batch 64
python train.py --cfg cfgs/model_yolo11l_v20250630_0623.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0630_0623+0604_bs64_crop --device 1,2 --batch 64

python train.py --cfg cfgs/model_yolo11l_v20250630.yaml --ckpt ../ckpts/yolo11s.pt --name 11S_v0630+0604_bs64_crop --device 1,2 --batch 64
python train.py --cfg cfgs/model_yolo11l_v20250630.yaml --ckpt ../ckpts/yolo11m.pt --name 11M_v0630+0604_bs64_crop --device 1,2 --batch 64
python train.py --cfg cfgs/model_yolo11l_v20250630.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0630+0604_bs64_crop --device 1,2 --batch 64

python train.py --cfg cfgs/model_yolo11l_v20250630_shuffle.yaml --ckpt ../ckpts/yolo11s.pt --name 11S_v0630+0630shuffle_bs64_crop --device 1,2 --batch 64
python train.py --cfg cfgs/model_yolo11l_v20250630_shuffle.yaml --ckpt ../ckpts/yolo11m.pt --name 11M_v0630+0630shuffle_bs64_crop --device 1,2 --batch 64
python train.py --cfg cfgs/model_yolo11l_v20250630_shuffle.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0630+0630shuffle_bs64_crop --device 1,2 --batch 64

python train.py --cfg cfgs/model_yolo11l_v20250604.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0604+0604_bs64_crop --device 1,2 --batch 64
python train.py --cfg cfgs/model_yolo11l_v20250604_shuffle.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0604+0604shuffle_bs64_crop --device 1,2 --batch 64
python train.py --cfg cfgs/model_yolo11l_v20250604_no_domain_gap.yaml --ckpt ../ckpts/yolo11l.pt --name 11L_v0604+0604noDomainGap_bs64_crop --device 1,2 --batch 64
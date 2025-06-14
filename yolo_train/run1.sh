source ~/.bashrc
conda activate swimAD

# python train.py --cfg cfgs/model_yolo11l_v20250506_shuffle.yaml --name yolov11l_swimAD_v0506_shuffle_bs64_e250_op_auto --ckpt ../ckpts/yolo11l.pt --device 2,3 --batch 64

# python train.py --cfg cfgs/model_yolo11l_v20250604_shuffle.yaml --name yolov11l_swimAD_v0604_shuffle_bs64_e250_op_auto --ckpt ../ckpts/yolo11l.pt --device 2,3 --batch 64

# python train.py --cfg cfgs/model_yolo11l_v20250506_shuffle_align.yaml --name yolov11l_swimAD_v0506_shuffle_bs64_e250_op_auto_align --ckpt ../ckpts/yolo11l.pt --device 2,3 --batch 64

# python train.py --cfg cfgs/model_yolo11l_v20250506_align.yaml --name yolov11l_swimAD_v0506_bs64_e250_op_auto_align --ckpt ../ckpts/yolo11l.pt --device 2,3 --batch 64

# python train.py --cfg cfgs/model_yolo11l_v20250604_custom_aug_v1.yaml --name yolov11l_swimAD_v0604_bs64_e250_op_auto_aug_v1 --ckpt ../ckpts/yolo11l.pt --device 2,3 --batch 64

# python train.py --cfg cfgs/model_yolo11m_v20250604.yaml --name yolov11m_swimAD_v0604_bs64_e250_op --ckpt ../ckpts/yolo11m.pt --device 2,3 --batch 64

# python train.py --cfg cfgs/model_yolo11s_v20250604.yaml --name yolov11s_swimAD_v0604_bs64_e250_op --ckpt ../ckpts/yolo11s.pt --device 2,3 --batch 64

# python train.py --cfg cfgs/model_yolo11l_v20250604_AdamW.yaml --name yolov11l_swimAD_v0604_bs64_e250_AdamW_lr --ckpt ../ckpts/yolo11l.pt --device 2,3 --batch 64


python train.py --cfg cfgs/model_yolo11l_v20250604_fold_3.yaml --name yolov11l_swimAD_v0604_bs64_e250_op_auto_fold_3 --ckpt ../ckpts/yolo11l.pt --device 0,1 --batch 64

python train.py --cfg cfgs/model_yolo11l_v20250604_fold_4.yaml --name yolov11l_swimAD_v0604_bs64_e250_op_auto_fold_4 --ckpt ../ckpts/yolo11l.pt --device 0,1 --batch 64


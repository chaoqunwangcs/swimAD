source ~/.bashrc
conda activate swimAD

python train.py -c cfgs/model_yolo11l_v20250630_0623_0604_moaisc0.yaml -n 11L_v0604_0634_0630+0604_bs64_moaisc0_0 -d 1,3 -b 64
python train.py -c cfgs/model_yolo11l_v20250630_0623_0604_moaisc01.yaml -n 11L_v0604_0634_0630+0604_bs64_moaisc0_1 -d 1,3 -b 64
python train.py -c cfgs/model_yolo11l_v20250630_0623_0604_rect.yaml -n 11L_v0604_0634_0630+0604_bs64_rect -d 1,3 -b 64
python train.py -c cfgs/model_yolo11l_v20250630_0623_0604_cls1_0.yaml -n 11L_v0604_0634_0630+0604_bs16_cls1_0 -d 1,3 -b 64
python train.py -c cfgs/model_yolo11l_v20250630_0623_0604_cls1_5.yaml -n 11L_v0604_0634_0630+0604_bs16_cls1_5 -d 1,3 -b 64
python train.py -c cfgs/model_yolo11l_v20250630_0623_0604_imgsz1280.yaml -n 11L_v0604_0634_0630+0604_bs16_imgsz1280 -d 1,3 -b 16

from ultralytics import YOLO
from custom_dataset import CustomTrainer, CustomValidator
import os
import argparse
import torch
import pdb

from pathlib import Path

project_path = Path(__file__).parent.resolve()
os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{project_path}"

def parse_arguments():
    """
    使用 argparse 解析命令行参数
    """
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("-c", "--cfg", type=str, default="cfgs/model_yolo11l_v20250506.yaml", help="the training config")
    parser.add_argument("-n", "--name", type=str, default="val", help="the exp name")
    parser.add_argument("-p", "--ckpt", type=str, default="../ckpts/yolo11l.pt", help="the pretrained model")
    parser.add_argument("-d", "--device", type=str, default="2,3", help="the training device")
    parser.add_argument("-b", "--batch", type=int, default=64, help="the training batch size")
    parser.add_argument("-w", "--workers", type=int, default=16, help="the training dataloader workers")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    model = YOLO(args.ckpt)  
    # debug. if the device id > 0 and only one device, it will used CUDA:0.
    device = args.device
    if len(args.device) == 1 and args.device != 0:
        device = torch.device(int(args.device))
    # pdb.set_trace()
    results = model.train(
        trainer = CustomTrainer,
        cfg=args.cfg,                  
        name=args.name,
        device=device, 
        batch=args.batch,
        workers=args.workers       
    )
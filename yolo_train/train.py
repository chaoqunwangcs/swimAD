from ultralytics import YOLO
from custom_dataset import CustomTrainer, CustomValidator
import os
import argparse
import pdb

def parse_arguments():
    """
    使用 argparse 解析命令行参数
    """
    
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("-c", "--cfg", type=str, default="cfgs/model_yolo11l_v20250506.yaml", help="the training config")
    parser.add_argument("-n", "--name", type=str, default="yolov11l_swimAD", help="the exp name")
    parser.add_argument("-p", "--ckpt", type=str, default="../ckpts/yolo11l.pt", help="the pretrained model")
    args = parser.parse_args()



    return args

if __name__ == '__main__':

    args = parse_arguments()
    model = YOLO(args.ckpt)  
    results = model.train(
        trainer = CustomTrainer,
        cfg=args.cfg,                  
        name=args.name            
    )
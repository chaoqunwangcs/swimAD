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
    parser.add_argument("-d", "--data-cfg", type=str, default="cfgs/data_swimAD_v20250506.yaml", help="the data config, only test the val set.")
    parser.add_argument("-p", "--ckpt", type=str, default="./runs/detect/yolov11l_swimAD/weights/best.pt", help="the pretrained model")
    args = parser.parse_args()



    return args

if __name__ == '__main__':

    args = parse_arguments()
    model = YOLO(args.ckpt)  
    model.overrides.update({'data':args.data_cfg})  # update new val dataset

    metrics = model.val(CustomValidator)  
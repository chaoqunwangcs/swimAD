from ultralytics import YOLO
from custom_dataset import CustomTrainer, CustomValidator
import os
import argparse
from functools import partial
import pdb

def vis(model):
    aa = model('../dataset/dataset_v20250604/afternoon_v2/1/afternoon_1_1001.jpg')
    bb = model('../dataset/dataset_v20250604/noon_v2/1/noon_1_1433.jpg')
    cc = model('../dataset/dataset_v20250604/noon_v2/2/noon_2_1211.jpg')
    aa[0].save(filename='aa.jpg', line_width=3, font_size=4)
    bb[0].save(filename='bb.jpg', line_width=3, font_size=4)
    cc[0].save(filename='cc.jpg', line_width=3, font_size=4)

def parse_arguments():
    """
    使用 argparse 解析命令行参数
    """
    
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("-d", "--data-cfg", type=str, default="cfgs/data_swimAD_v20250604.yaml", help="the data config, only test the val set.")
    parser.add_argument("-p", "--ckpt", type=str, default="./runs/detect/yolov11l_swimAD/weights/best.pt", help="the pretrained model")
    parser.add_argument("-r", "--region", type=str, default="region", choices=["all", "region", "remain"], help="the evaluation region")
    args = parser.parse_args()



    return args

if __name__ == '__main__':

    args = parse_arguments()
    model = YOLO(args.ckpt)  
    model.overrides.update({'data':args.data_cfg})  # update new val dataset
    # vis(model)
    # pdb.set_trace()
    metrics = model.val(validator=partial(CustomValidator, region=args.region))  
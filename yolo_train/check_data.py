from ultralytics import YOLO
from custom_dataset import CustomTrainer
import os
import pdb

model = YOLO('../ckpts/yolo11m.pt')  

results = model.train(
    trainer = CustomTrainer,
    cfg='yolo11m_v20250506.yaml',                  
    name='yolov11m_swimAD'            
)

metrics = model.val()  
print(f"mAP50-95: {metrics.box.map}")  

pdb.set_trace()
train_dataset = model.trainer.train_dataset
val_dataset = model.trainer.val_dataset

# 绘制训练集图像及其标注
print("Visualizing training dataset...")
plot_images(train_dataset.images, train_dataset.targets, train_dataset.paths, names=train_dataset.names, max_size=1920, max_subplots=16)

# 绘制验证集图像及其标注
print("Visualizing validation dataset...")
plot_images(val_dataset.images, val_dataset.targets, val_dataset.paths, names=val_dataset.names, max_size=1920, max_subplots=16)
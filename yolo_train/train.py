from ultralytics import YOLO
from custom_dataset import CustomTrainer
import os
import pdb

# pdb.set_trace()
model = YOLO('../ckpts/yolo11m.pt')  # 可选: 'yolov8n.pt'(小), 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'(大)

data_path = 'swimAD_v20250506.yaml'
# model.overrides.update({'data':data_path})
# model.trainer = CustomTrainer(overrides=model.overrides)

results = model.train(
    trainer = CustomTrainer,
    cfg='yolo11m_v20250506.yaml',                  
    name='yolov11m_swimAD'            
)

metrics = model.val()  
print(f"mAP50-95: {metrics.box.map}")  

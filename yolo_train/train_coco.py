from ultralytics import YOLO

model = YOLO('yolo11l.yaml')  # 加载预训练模型
results = model.train(
    data='coco.yaml',  # 数据集配置文件路径
    epochs=300,        # 训练轮次
    imgsz=640,         # 图像尺寸
    batch=16,          # 批量大小
    lr0=0.01,          # 初始学习率
    lrf=0.1,           # 学习率衰减因子
    save_json=True,     # 启用 COCO 指标计算
    workers=0
)
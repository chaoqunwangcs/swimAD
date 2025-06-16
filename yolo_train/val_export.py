from ultralytics import YOLO
from custom_dataset import CustomTrainer, CustomValidator
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO模型验证和ONNX导出")
    parser.add_argument("-d", "--data-cfg", type=str, 
                       default="cfgs/data_swimAD_v20250604.yaml", 
                       help="数据配置文件路径")
    parser.add_argument("-p", "--ckpt", type=str, 
                       default="./runs/detect/yolov11l_swimAD/weights/best.pt", 
                       help="预训练模型权重路径")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="使用设备 (cuda:0 或 cpu)")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    # 1. 验证原始模型
    model = YOLO(args.ckpt).to(args.device)
    model.overrides.update({'data': args.data_cfg})
    print("验证原始PyTorch模型...")
    pt_metrics = model.val(validator=CustomValidator)
    
    # 2. 导出ONNX
    print("\n导出ONNX模型...")
    model.export(format="onnx", dynamic=True, opset=12, simplify=True)
    onnx_path = args.ckpt.replace('.pt', '.onnx')
    
    # 3. 验证ONNX模型
    print("\n验证ONNX模型...")
    try:
        # 优先尝试使用CUDA
        import pdb; pdb.set_trace()
        onnx_model = YOLO(onnx_path)
        onnx_metrics = onnx_model.val(validator=CustomValidator)
    except Exception as e:
        print(f"CUDA错误: {e}\n回退到CPU...")
        onnx_model = YOLO(onnx_path)
        onnx_metrics = onnx_model.val(validator=CustomValidator, providers=['CPUExecutionProvider'])
    
    # 结果比较
    print("\n验证结果比较:")
    print(f"原始模型 mAP50: {pt_metrics.box.map50:.4f}")
    print(f"ONNX模型 mAP50: {onnx_metrics.box.map50:.4f}")
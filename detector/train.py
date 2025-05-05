from ultralytics import YOLO
import os 

# --- 配置参数 ---
# 指定数据集配置
CONFIG_PATH = "train_config.yaml.yaml"
# 使用预训练模型
MODEL_NAME = "/root/yolov12/yolov12m.pt"

EPOCHS = 250

BATCH_SIZE = 64 #default 16

IMG_SIZE = 640

PATIENCE = 50 

save_period_epoch = 30 # 每 30 轮保存一次权重

# --- 指定结果保存路径 ---
# 指定结果保存的父目录
PROJECT_SAVE_DIR = "/root/detetion_v1/runs"
# 本次实验的名称 (会作为 PROJECT_SAVE_DIR 下的子目录)
EXP_NAME = "yolov12m_run1_epoch250_batchsize128_imgsize640"

#配置结束

# --- 检查配置文件是否存在 ---
if not os.path.exists(CONFIG_PATH):
    # 如果 config.yaml 不在当前目录，尝试从脚本所在目录的上级查找
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    alt_config_path = os.path.join(script_dir, CONFIG_PATH)
    if os.path.exists(alt_config_path):
        CONFIG_PATH = alt_config_path
    else:
        print(f"错误：找不到配置文件 {CONFIG_PATH} 或 {alt_config_path}")
        exit() # 或者抛出异常

# --- 加载模型 ---
model = YOLO(MODEL_NAME)

# --- 开始训练 ---
train_results = None # 初始化变量
try:
    print(f"开始训练，结果将保存在 {PROJECT_SAVE_DIR}/{EXP_NAME} 目录下...")
    train_results = model.train(
        data=CONFIG_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        project=PROJECT_SAVE_DIR, # 指定保存结果的父目录
        name=EXP_NAME,           # 指定本次实验的子目录名
        exist_ok=True,           # 如果目录已存在，允许覆盖或继续（可选）
        save_period=save_period_epoch,        
        # device=0 # 如果需要指定 GPU，取消注释
    )
    # train_results.save_dir 在这种情况下可能不会直接反映 project/name 结构，
    # 但文件确实保存在了 project/name 指定的路径下
    print(f"训练完成！结果应保存在: {os.path.join(PROJECT_SAVE_DIR, EXP_NAME)}")

except Exception as e:
    print(f"训练过程中发生错误: {e}")


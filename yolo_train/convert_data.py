import os, pdb
import argparse, logging
from datetime import datetime
from pathlib import Path

EXTS = ['.jpg','.png']

data_root = r'../dataset/'              # default setting
anno_root = r'../data_transfer/yolo'    # default setting

def setup_logging(log_level, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{current_date}.log")

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file), 
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """
    使用 argparse 解析命令行参数
    """
    
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("-v", "--data-version", type=str, default="dataset_v20250506", help="the version of the dataset")
    parser.add_argument("-l", "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="set log level, default is INFO")
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level)
    setup_logging(log_level, "logs")

    return args





# training ratio = 0.5, one clip to train and one clip to val

training_ratio = 0.9

def scan_clips(clips, save_path, data_version):
    all_items = []
    for clip in clips:
        data_path = os.path.join(data_root, data_version, clip)
        anno_path = os.path.join(anno_root, data_version, clip)
        views = sorted(os.listdir(data_path))
        for view in views:
            data_view_path = os.path.join(data_path, view)
            anno_view_path = os.path.join(anno_path, view)
            items = sorted(os.listdir(data_view_path))
            items = sorted(list(filter(lambda x: Path(x).suffix in EXTS, items)))
            for item in items:
                # pdb.set_trace()
                item_path = os.path.join(data_view_path, item)
                yolo_path = os.path.join(anno_view_path, item.replace(Path(item).suffix, '.txt'))
                if not (Path(item_path).is_file() and Path(yolo_path).is_file()): continue
                item_path = os.path.abspath(item_path)
                yolo_path = os.path.abspath(yolo_path)  # absolute path
                line = f"{item_path},{yolo_path}"
                all_items.append(line)
    # pdb.set_trace()
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_items) + "\n")
    print(f"Generate {save_path}, total {len(all_items)} items!")



if __name__ == '__main__':

    args = parse_arguments()

    save_train_path = os.path.join(anno_root, f"{args.data_version}_train.txt")
    save_val_path = os.path.join(anno_root, f"{args.data_version}_val.txt")
    data_path = os.path.join(data_root, args.data_version)
    clips = sorted(os.listdir(data_path))

    train_clips = clips[:int(len(clips)*training_ratio)]
    val_clips = clips[int(len(clips)*training_ratio):]


    scan_clips(train_clips, save_train_path, args.data_version)
    scan_clips(val_clips, save_val_path, args.data_version)
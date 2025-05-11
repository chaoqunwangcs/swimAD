import os, pdb
from pathlib import Path

EXTS = ['.jpg','.png']

data_root = r'../dataset/'
anno_root = r'../data_transfer/yolo'
data_version = 'dataset_v20250506'

save_train_path = os.path.join(anno_root, f"{data_version}_train.txt")
save_val_path = os.path.join(anno_root, f"{data_version}_val.txt")

# training ratio = 0.5, one clip to train and one clip to val

training_ratio = 0.5

def scan_clips(clips, save_path):
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

    data_path = os.path.join(data_root, data_version)
    clips = sorted(os.listdir(data_path))

    train_clips = clips[:int(len(clips)*training_ratio)]
    val_clips = clips[int(len(clips)*training_ratio):]


    scan_clips(train_clips, save_train_path)
    scan_clips(val_clips, save_val_path)
import os 
import pdb

data_path = r'/home/chaoqunwang/swimAD/dataset/dataset_v20250604'
anno_path = r'/home/chaoqunwang/swimAD/data_transfer/yolo/dataset_v20250604'

train_file = r'dataset_v20250604_train_no_domain_gap.txt'
val_file = r'dataset_v20250604_val_no_domain_gap.txt'


train_lines = []
val_lines = []

folders = os.listdir(data_path)
for folder in folders:
    data_folder = os.path.join(data_path, folder)
    anno_folder = os.path.join(anno_path, folder)
    views = os.listdir(data_folder)
    for view in views:
        data_view = os.path.join(data_folder, view)
        anno_view = os.path.join(anno_folder, view)


        # pdb.set_trace()
        imgs = list(sorted(next(os.walk(data_view))[2]))
        imgs = [x for x in imgs if x.endswith('.jpg')]
        train_len = int(0.8 * len(imgs))
        if folder == 'morning_v2':
            train_len = int(1.0 * len(imgs))
        # imgs = [os.path.join(data_view, x) for x in imgs]
        lines = []
        for img in imgs:
            line = f"{os.path.join(data_view, img)},{os.path.join(anno_view, img.replace('.jpg','.txt'))}\n"
            lines.append(line)
        
        train_lines += lines[:train_len]
        val_lines += lines[train_len:]

with open(train_file, 'w', encoding='utf-8') as f20:
    f20.writelines(train_lines)

with open(val_file, 'w', encoding='utf-8') as f80:
    f80.writelines(val_lines)
        
        
import random
import pdb
import os

# 打开原始文件
with open('dataset_v20250604_train.txt', 'r', encoding='utf-8') as file:
    lines_train = file.readlines()

pdb.set_trace()
with open('dataset_v20250604_val.txt', 'r', encoding='utf-8') as file:
    lines_val = file.readlines()
    lines = lines_train + lines_val

# k-fold
dirs = os.listdir(r'../../dataset/dataset_v20250604')
k = 5
val_len = int(len(dirs)/5.0)

for i in range(k):
    val_folders = dirs[:val_len]
    train_folders = dirs[val_len:]
    dirs = train_folders + val_folders
    file_val = []
    file_train = []
    for line in lines:
        is_val = False
        for val_folder in val_folders:
            if f"/{val_folder}/" in line:
                file_val.append(line)
                is_val = True
        if not is_val:
            file_train.append(line)
    
    with open(f"dataset_v20250604_folder_{i}_val.txt", 'w', encoding='utf-8') as f20:
        f20.writelines(file_val)

    with open(f"dataset_v20250604_folder_{i}_train.txt", 'w', encoding='utf-8') as f80:
        f80.writelines(file_train)



print("文件拆分完成！")
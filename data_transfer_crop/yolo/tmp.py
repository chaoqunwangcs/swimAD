import random
import pdb

# 打开原始文件
with open('dataset_v20250623_train.txt', 'r', encoding='utf-8') as file:
    lines_train = file.readlines()

pdb.set_trace()
with open('dataset_v20250623_val.txt', 'r', encoding='utf-8') as file:
    lines_val = file.readlines()
    lines = lines_train + lines_val
# 打乱行顺序
random.shuffle(lines)

# 计算拆分点
split_point = int(len(lines) * 0.1)

# 拆分数据
file_10_percent = lines[:split_point]
file_90_percent = lines[split_point:]

# 写入到新的文件
with open('dataset_v20250623_shuffle_val.txt', 'w', encoding='utf-8') as f20:
    f20.writelines(file_10_percent)

with open('dataset_v20250623_shuffle_train.txt', 'w', encoding='utf-8') as f80:
    f80.writelines(file_90_percent)

print("文件拆分完成！")
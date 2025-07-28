import os
import glob
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
import pdb



def calculate_mean_and_std(image_folder, batch_size=100):
    # 使用 glob 递归搜索所有 .jpg 文件
    image_paths = glob.glob(os.path.join(image_folder, '**', '*.jpg'), recursive=True)
    random.shuffle(image_paths)
    # 初始化变量
    total_pixels = 0
    sum_pixels = np.zeros(3, dtype=np.float64)
    sum_squared_pixels = np.zeros(3, dtype=np.float64)
    all_mean = []
    all_std = []
    # 分批处理图像
    for i in tqdm(range(0, len(image_paths[:1000]), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        batch_data = []

        for image_path in batch_paths:
            image = np.array(Image.open(image_path).convert('RGB'))
            batch_data.append(image)

        batch_data = np.concatenate(batch_data, axis=0)
        batch_mean = np.mean(batch_data, axis=(0, 1)) / 255.0 
        batch_std = np.std(batch_data, axis=(0, 1)) / 255.0 
        all_mean.append(batch_mean)
        all_std.append(batch_std)
        # print(all_mean, all_std)
        # pdb.set_trace()
    # 计算均值和标准差
    all_mean = np.array(all_mean)
    mean = np.mean(all_mean, axis=0)
    all_std = np.array(all_std)
    std = np.mean(all_std, axis=0)

    return mean, std

# 指定图像文件夹路径
image_folder = '../../dataset/dataset_v20250604/'

# 计算均值和标准差
mean, std = calculate_mean_and_std(image_folder, batch_size=100)

print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")




# image_folder = '../../dataset/dataset_v20250604/'
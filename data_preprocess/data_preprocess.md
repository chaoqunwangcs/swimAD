### 文件夹格式
dataset/
|----dataset_v20250506/
|--------clip1/
            view1
                img1.jpg
                img1.json
                img2.jpg
                img2.json
                ...
            view2
            view3
            view4
        clip2
        ...
    dataset_v<yyyymmdd>

### 数据可视化

```
python data_process/data_visualization.py --dataset_v20250506 --labelme 
```

### 数据预处理

#### 转换为TXT格式(用于训练YOLO)

python data_process/data_transfer.py --dataset_v20250506 --yolo

#### 转换为COCO格式

```
python data_process/data_transfer.py --dataset_v20250506 --coco
```


#### 转换为MOT17格式（用于训练MOT）

```
python data_process/data_transfer.py --dataset_v20250506 --mot
```


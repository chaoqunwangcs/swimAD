### 文件操作指令

#### coco_eval.py

**example**

```
python data_preprocess\coco_eval.py    --gt_file "..\instances_val_gt.json"     --pred_file "..\pred.json"  --results_dir ./evaluation_output_folder
```



### 文件夹格式

```plaintext
dataset/
|---dataset_v20250506/ 不同的标注版本 v后面是标注日期
|   |---clip1/  video片段
|   |   |---view1   相机视角
|   |   |   |---img1.jpg    图像名
|   |   |   |---img1.json   对应labelme生成json标注
|   |   |   |---img2.jpg
|   |   |   |---img2.json
|   |   |   |---...
|   |   |---view2
|   |   |---view3
|   |   |---view4
|   |---clip2
|   |---...
|---dataset_v<yyyymmdd>
```

### 数据可视化

```
python data_preprocess/data_visualization.py --input_version dataset_v20250506 --label_format labelme[labelme/coco/txt/mot]
```

### 数据预处理

！默认存放在data_transfer文件夹，如果修改，后续模型训练和推理需要对应修改

1. - [x] 转换为TXT格式(用于训练和评测YOLO)

```
python data_process/data_transfer.py --input_version dataset_v20250506 --label_format yolo
```


2. - [x] 转换为COCO格式

```
python data_process/data_transfer.py --input_version dataset_v20250506 --label_format coco
```


3. - [x] 转换为MOT17格式（用于训练和评测MOT）

```
python data_process/data_transfer.py --input_version dataset_v20250506 --label_format mot
```


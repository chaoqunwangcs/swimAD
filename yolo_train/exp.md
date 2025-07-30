# 检测模型训练：
## Data: 2025-06-08
1. 训练v20250604_shuffle(mAP50=0.9) vs v20250506_shuffle(0.9) vs  v20250506_shuffle_align(mAP50=0.6)实验结果，大批量数据远好于小批量数据-->结论：v20250604的验证集更难

2. 训练v20250604(mAP50=0.52) vs v20250506_align(mAP50=0.48), 大批量数据效果更好，但是均发生了过拟合，需要策略缓解过拟合

3. 训练v20250604(mAP50=0.52) vs v20250604_aug(mAP50=0.52), 效果完全一致，应该是数据增强参数没有正确打开

TODO

0. 0604训练，0506测试, 0.95，很高
1. 训练数据shuffle, 结果一致，说明默认shuffle了
3. 5折实验,实验结果依旧过拟合，不是数据划分的问题
4. data aug: 默认开启的
5. 降低模型参数量，从Large 降低到small or medium，依旧过拟合
6. 更改优化器，先更改weight decay系数， 没用
7. 可视化看着近处效果还行，需要测量不同scale object的mAP（李健）
8. 为了防止domain gap, 前400帧训练，后100帧测试，效果比完全shuffle差，但是比之前video shuffle好，问题发生在不同video之间的domai gap，目前看有两个，一个是人不同，一个是光照不同

当前结论(0614):
1. 现在模型，在video level进行shuffle性能下降不是overfitting的问题，而是domain gap的问题
2. 要缓解这个问题，需要从两个地方出发，一个是增加训练集的diversity，一个是修改训练的时候数据在光线方面的增强方式

原因分析:
1. k-fold折实验,实验结果依旧下降，不是数据划分的问题
2. 降低模型参数量，从Large 降低到small or medium，依旧不行
3. 增大weight decay系数，没有效果
4. 新增实验。不在video level shuffle，而是每一个视频都前400帧训练，后100帧验证，结果也会下滑

TODO (0614):
1. 数据增强(尤其光照条件)
2. 增加数据diversity

## 2025。06.15
数据增强光照部分增强失败，是因为anaconda3/envs/swimAD/lib/python3.9/site-packages/ultralytics/data/augment.py 中，albumentations库没有正确安装，同时albumentations.ImageCompression函数形参有误导致增强函数调用失败，在custom_dataset中，将albumentations.ImageCompression中的quality_range(75,100)改为quality_lower=75, quality_upper=100即可

其他的数据增强： 需要在在albumentations库中设置
1. RandomBrightnessContrast设置p=0.1
2. RandomGamma设置p=0.1
3. ImageCompression设置p=0.01

结果依旧不行

预处理中加Normalization,结果不行，不知道是不是有问题

设置更强的AUG
1. RandomBrightnessContrast设置p=0.5
2. RandomGamma设置p=0.5
3. ImageCompression设置p=0.01
4. ToGray(p=0.5), 原来是0.01

上述设置还是不行

TODO: 
1. 增加object损失权重
2. 可以试试调整匹配阈值
3. 可以把aug降低点儿，这样可以降低一些数据分布的误差
4. 可以试试在推理的时候降低置信度阈值

## 实验结果

| 参数   | 值    | mAP50-90 | 备注 |
|--------|-------|-----------|---|
|mosaic  | 1.0   |  0.256    | 默认  |
|mosaic  | 0.1   |           |  减少数据分布gap |
|mosaic  | 0.0   |           |  减少数据分布gap |
|cls     | 0.5   |  0.256    | 默认  |
|cls     | 1.0   |           |  增大分类的权重 |
|cls     | 1.5   |           |  增大分类的权重 |
|imgsz   | 640   |  0.256    | 默认  |
|imgsz   | 1280  |           | 增大分辨率，后续可用于蒸馏 |
|rect    | False |           | 默认  |
|rect    | True  |           | 仅尝试|

|conf*   | 0.25  |   0.256   |推理默认值|
|conf*   | 0.1   |   0.29     |recall没有明显提升|
|conf*   | 0.01   |   0.26     |recall没有明显提升,分类错误率提升|

|iou*    | 0.7   |   0.256     |默认|
|iou*    | 0.4   |   0.241     |recall几乎没有提升，precision降低|
|max_det* | 300   |   0.256     | 默认 |
|max_det* | 50    |   0.274     | 整体作用不大  |


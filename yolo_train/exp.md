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

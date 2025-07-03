## 20250624进度
目前逻辑：
1. 都GT数据生成测试数据
2. 每一个view的所有点的数据生成center point, 并计算其网格
3. 初始化view之间point和grid的映射关系。point映射依靠对应关键点（泳池四个角），grid映射根据坐标系关系进行映射
4. 将view2中的点和grid映射到view1, 并根据point/grid之间的距离，进行匈牙利匹配
5. 根据匹配结果，对点的位置进行更新（后续可以改为对grid进行更新），并重新计算其grid
6. 对于view3, view4重复view2的操作
7. 输出匹配结果，可视化

现象/结论
1. 基于point projection/grid projection + hungarian_match的方法，效果都还不错，原因是反畸变做好之后，映射相对和grid划分相对准确
2. 基于point projection + hungarian_match的方法效果更好，（可能是因为grid划分和判定还有问题，尤其是远处）
3. grid划分，在远处还是有问题，这会导致匹配有问题，还需要更精确一些@浩然，
4. 目前point更新是基于point projection的，目前看还是有一定误差，这些误差会导致轨迹不稳定。是否要改成grid更新。
5. 目前没有增加权重，这个还是很有必要的
6. 目前测试版本在GT上进行测试，没有漏检、误检例子，还需要继续debug

[TODO]
1. 远处的grid划分, point projection仍然不够准确，可能会导致匹配错误
    a. 更精确的grid标注
    b. 是否应该考虑偏移问题，因为标注的grid和水面，人的位置不在一个平面，人的box实际上因为透视关系导致grid判断错误

2. 如何给权重，用于做point update， 近距离物体给较大权重，远距离物体给较小权重，目前全部相等
3. model inference的时候，各种边界情况，如漏检、假阳性等如何处理，目前还没写这部分逻辑
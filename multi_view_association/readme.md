## 20250624进度
1. 基于point projection + hungarian_match的方法，效果不好，原因目前是point projection现在还有问题
2. 基于grid projection + hungarian_match的方法，目前看效果还不错，错误的case应该是远处物体，grid划分误差问题

[TODO]
1. debug point projection函数，现在是错误的
2. 远处的grid划分问题
3. 如何给权重，用于做point update， 近距离物体给较大权重，远距离物体给较小权重，目前全部相等
4. 是否应该考虑偏移问题，因为标注的grid和水面，人的位置不在一个平面，人的box实际上因为透视关系导致grid判断错误
5. model inference的时候，各种边界情况，如漏检、假阳性等如何处理
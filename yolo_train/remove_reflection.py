import cv2
import albumentations as A
import numpy as np
import pdb

def remove_reflection(image, **kwargs):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 用高亮阈值检测反光区域（可调整阈值）
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 可选：膨胀 mask 让边缘更平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 用 inpainting 填补反光区域
    inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

    return inpainted

# 包装成 Albumentations 的 transform
transform = A.Compose([
    A.Lambda(image=remove_reflection, p=1.0)
])


# 使用示例
your_image = cv2.imread('/home/chaoqunwang/swimAD/dataset/dataset_v20250630/one_clip/1/0531_0929_1.jpg')
aug = transform(image=your_image)
result = aug['image']
import pdb
pdb.set_trace()
cv2.imwrite('aa.jpg', result)
import cv2
import numpy as np

# -------------------- 1. 相机参数 --------------------
p1 = 0
p2 = 0
k1 = -0.3957
k2 = 0.1834
k3 = -0.0504
fx = 2383.2
fy = 2377.5
cx = 1627.9
cy = 869.4700
fx_ratio = 1
fy_ratio = 1

camera_matrix = np.array([[fx * fx_ratio, 0, cx],
                          [0, fy * fy_ratio, cy],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# -------------------- 2. 六个畸变图像上的点 --------------------
# 按需改成自己的 6 个点
distorted_points = [
    (102, 581),
    (1046, 2016),
    (3153, 329),
    (2116, 45)
]

# -------------------- 3. 读图并计算反畸变映射 --------------------
image_path = "/home/chaoqunwang/swimAD/dataset/dataset_v20250804/video_1/2/1_1.jpg"
output_original_points = "original_with_points_v2.jpg"
output_undistorted_points = "undistorted_with_points_v2.jpg"

image = cv2.imread(image_path)
import pdb; pdb.set_trace()
if image is None:
    raise FileNotFoundError(f"无法加载图像: {image_path}")

h, w = image.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), 1, (w, h))
map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)
undistorted_image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

# -------------------- 4. 在原始图上绘制 6 个点 --------------------
original_vis = image.copy()
for (x, y) in distorted_points:
    cv2.circle(original_vis, (int(x), int(y)), 10, (0, 255, 0), 20)  # 红色
cv2.imwrite(output_original_points, original_vis)

# -------------------- 5. 计算 6 个点在去畸变图上的位置并绘制 --------------------
undistorted_points = []
undistorted_vis = undistorted_image.copy()

for (x, y) in distorted_points:
    # 注意 cv2.undistortPoints 输入是 (1,N,2) 形状
    pts_in = np.array([[[x, y]]], dtype=np.float32)
    pts_out = cv2.undistortPoints(
        pts_in, camera_matrix, dist_coeffs, P=new_camera_matrix)
    ux, uy = pts_out[0, 0]
    undistorted_points.append((ux, uy))
    cv2.circle(undistorted_vis, (int(ux), int(uy)), 10, (0, 255, 0), 20)  # 绿色

cv2.imwrite(output_undistorted_points, undistorted_vis)

# -------------------- 6. 打印结果 --------------------
print("原始畸变图像坐标 -> 去畸变图像坐标")
for (dx, dy), (ux, uy) in zip(distorted_points, undistorted_points):
    print(f"({dx:7.2f}, {dy:7.2f}) -> ({ux:7.2f}, {uy:7.2f})")
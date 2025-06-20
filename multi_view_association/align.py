import cv2
import numpy as np

fx, fy = 1621.9, 1856.1
cx, cy = 1116.3, 742.9178

fx_ratio = 1.33
fy_ratio = 1.33

fx = fx * fx_ratio
fy = fy * fy_ratio

k1, k2, k3 =  -0.5353, 0.2875, -0.0906   # 径向畸变系数
# p1, p2 = 0.00016542, 0.00076033     #切向畸变系数
p1, p2 = 0, 0

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

dist_coeffs = np.array([k1, k2, p1, p2, k3])  # 径向和切向畸变系数


image = cv2.imread(r"G:\project\calibration_data\morning_1_01001.jpg")
h, w = image.shape[:2]

new_K, roi = cv2.getOptimalNewCameraMatrix(
    K, dist_coeffs, (w, h), alpha=1, newImgSize=(w, h))
# alpha=0：裁剪黑边，alpha=1：保留所有有效像素

mapx, mapy = cv2.initUndistortRectifyMap(
    K, dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1)
undistorted = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

cv2.imwrite(r"G:\project\calibration_data\undistorted.jpg", undistorted)
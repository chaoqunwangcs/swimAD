import cv2
import numpy as np

# 相机参数
p1 = 0
p2 = 0
k1 = -0.5153
k2 = 0.2845
k3 = -0.0906
fx = 1621.9
fy = 1856.1
cx = 1116.3
cy = 742.9178
fx_ratio = 1.34
fy_ratio = 1.34

distorted_point = (2186, 210)

# 构建相机矩阵和畸变系数
camera_matrix = np.array([[fx*fx_ratio, 0, cx], 
                         [0, fy*fy_ratio, cy], 
                         [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# 图像路径
image_path = "/home/chaoqunwang/swimAD/dataset/dataset_v20250506/afternoon/1/afternoon_1_0001.jpg"
output_path = "undistorted_afternoon_1_0001.jpg"

# 读取图像
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"无法加载图像: {image_path}")

# 计算反畸变映射
h, w = image.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)

# 应用反畸变
undistorted_image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

# 保存结果
cv2.imwrite(output_path, undistorted_image)

# 定义一个函数来计算点在反畸变图像中的位置
def get_undistorted_point(x, y, camera_matrix, dist_coeffs, new_camera_matrix):
    # 将点转换为归一化坐标
    import pdb; pdb.set_trace()
    point = np.array([[x, y]], dtype=np.float32)
    undistorted_point = cv2.undistortPoints(point, camera_matrix, dist_coeffs, P=new_camera_matrix)
    return undistorted_point[0][0]

# 示例：计算原图中某点(比如中心点)在反畸变图像中的位置
original_point = distorted_point  # 使用图像中心点作为示例
undistorted_point = get_undistorted_point(original_point[0], original_point[1], 
                                         camera_matrix, dist_coeffs, new_camera_matrix)

print(f"原图中的点: {original_point}")
print(f"反畸变图像中的对应点: {undistorted_point}")
import cv2
import numpy as np
import pdb



################## view1 ##########################
# 相机参数
p1 = 0
p2 = 0
k1 = -0.5353
k2 = 0.2875
k3 = -0.0906
fx = 1621.9
fy = 1856.1
cx = 1116.3
cy = 742.9178
fx_ratio = 1.33
fy_ratio = 1.33

# 构建相机矩阵和畸变系数
camera_matrix = np.array([[fx*fx_ratio, 0, cx], 
                         [0, fy*fy_ratio, cy], 
                         [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# 图像路径
image_path = "/home/chaoqunwang/swimAD/dataset/dataset_v20250506/afternoon/1/afternoon_1_0001.jpg"

# 读取图像
image = cv2.imread(image_path)

# 计算反畸变映射
h, w = image.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)

# 应用反畸变
image_view1 = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)


################## view2 ##########################
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

# 构建相机矩阵和畸变系数
camera_matrix = np.array([[fx*fx_ratio, 0, cx], 
                         [0, fy*fy_ratio, cy], 
                         [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# 图像路径
image_path = "/home/chaoqunwang/swimAD/dataset/dataset_v20250506/afternoon/2/afternoon_2_0001.jpg"

# 读取图像
image = cv2.imread(image_path)

# 计算反畸变映射
h, w = image.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)

# 应用反畸变
image_view2 = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)


view1_points = [[116.50074718706048, 488.61005859374995], [786.2053885372715, 1231.890005493164], [562.9705080872012, 439.41005859374997], [1814.766101148617, 657.0900054931641]]

view2_points = [[1633.661509610877, 305.28749999999997], [305.05391467416786, 587.2875], [2326.1697140178153, 342.4875], [1288.0075011720583, 1277.2875]]

def draw_circle(image, center, color, string, radius=5, thickness=3): # center = (x,y)
    cv2.circle(image, (int(center[0]), int(center[1])), radius, color, thickness=thickness)
    cv2.putText(image, string, (int(center[0]), int(center[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 3, color, thickness=thickness)

for idx, point in enumerate(view1_points):
    draw_circle(image_view1, point, (0, 255, 0), str(idx))

for idx, point in enumerate(view2_points):
    draw_circle(image_view2, point, (0, 255, 0), str(idx))


cv2.imwrite('img1.jpg', image_view1)
cv2.imwrite('img2.jpg', image_view2)

view2_point = (1291.4756, 641.4832)
draw_circle(image_view2, view2_point, (255, 0, 0), 'point')
cv2.imwrite('img2.jpg', image_view2)

homography_matrix, _ = cv2.findHomography(np.array(view2_points), np.array(view1_points))

def point_projection(homography_matrix, point):
    point_homogeneous = np.array([point[0], point[1], 1])
    projected_point_homogeneous = homography_matrix @ point_homogeneous
    projected_point = projected_point_homogeneous[:2] / projected_point_homogeneous[2]
    return [projected_point[0], projected_point[1]]

pdb.set_trace()
view1_point = point_projection(homography_matrix, view2_point)
draw_circle(image_view1, view1_point, (255, 0, 0), 'point')
cv2.imwrite('img1.jpg', image_view1)


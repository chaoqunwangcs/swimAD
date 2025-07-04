from test_stream_split import MultiViewAssociationStream
import numpy as np



associator = MultiViewAssociationStream(r'region_calibration_data_v1.json')

view1_points = []   # cx, cy
view2_points = []
view3_points = []
view4_points = []



view1_data = []
for point in view1_points:
    view1_data.append([point[0]-50, point[1]-50, point[0]+50, point[1]+50, 1.0, 1.0])
np.array(view1_data)

view2_data = []
for point in view2_points:
    view2_data.append([point[0]-50, point[1]-50, point[0]+50, point[1]+50, 1.0, 1.0])
np.array(view2_data)

view3_data = []
for point in view3_points:
    view3_data.append([point[0]-50, point[1]-50, point[0]+50, point[1]+50, 1.0, 1.0])
np.array(view3_data)

view4_data = []
for point in view4_points:
    view4_data.append([point[0]-50, point[1]-50, point[0]+50, point[1]+50, 1.0, 1.0])
np.array(view4_data)

view1 = {'image_path': r'/home/chaoqunwang/swimAD/dataset/dataset_v20250506/afternoon/1/afternoon_1_0001.jpg', 'det':view1_data}

view2 = {'image_path': r'/home/chaoqunwang/swimAD/dataset/dataset_v20250506/afternoon/2/afternoon_2_0001.jpg', 'det':view2_data}

view3 = {'image_path': r'/home/chaoqunwang/swimAD/dataset/dataset_v20250506/afternoon/3/afternoon_3_0001.jpg', 'det':view3_data}

view4 = {'image_path': r'/home/chaoqunwang/swimAD/dataset/dataset_v20250506/afternoon/4/afternoon_4_0001.jpg', 'det':view4_data}

dets = [view1, view2, view3, view4]
associator.forward(dets)
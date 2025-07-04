from test_stream_split import MultiViewAssociationStream
import numpy as np
import pdb
import cv2
from tqdm import tqdm


associator = MultiViewAssociationStream(r'region_calibration_data_v2.json')

view1_point = (338, 573) # cx, cy
view2_point = (5457-3533, 500) # cx, cy
view3_point = (1755, 1979-1440) # cx, cy
view4_point = (4013-3533, 1975-1440) # cx, cy


def process(point):
    # N * 2 * 6 # N: num frame, 2: num_box
    det = np.array([[point[0] - 50, point[1] - 50, point[0] + 50, point[1] + 50, 1.0, 1.0]])
    return det

view1_data = process(view1_point)  # 1 * 6
view2_data = process(view2_point)
view3_data = process(view3_point)
view4_data = process(view4_point)

pdb.set_trace()
view1 = {'image_path': r'/home/chaoqunwang/swimAD/dataset/dataset_v20250506/afternoon/1/afternoon_1_0001.jpg', 'det':view1_data}

view2 = {'image_path': r'/home/chaoqunwang/swimAD/dataset/dataset_v20250506/afternoon/2/afternoon_2_0001.jpg', 'det':view2_data}

view3 = {'image_path': r'/home/chaoqunwang/swimAD/dataset/dataset_v20250506/afternoon/3/afternoon_3_0001.jpg', 'det':view3_data}

view4 = {'image_path': r'/home/chaoqunwang/swimAD/dataset/dataset_v20250506/afternoon/4/afternoon_4_0001.jpg', 'det':view4_data}

dets = [view1, view2, view3, view4]
img = associator.forward(dets)
cv2.imwrite('aa.jpg', img)
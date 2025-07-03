import os, sys
import numpy as np
import json
import math
import cv2
import copy
import random 
random.seed(42)


import pdb

MAIN_VIEW = 'MAIN'
HEIGHT = 1440
WIDTH = 2560

POOL_WIDTH = 1500
POOL_HEIGHT = 2500
MARGIN_WIDTH = 300
MARGIN_HEIGHT = 300

class Box(object):
    def __init__(self, x1, y1, x2, y2, conf, cls_id, cur_view, source_view, image_path, is_distorted=True):
        self.cx, self.cy = (x1 + x2)/2.0, (y1 + y2)/2.0
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.original_cx, self.original_cy = (x1 + x2)/2.0, (y1 + y2)/2.0
        self.original_x1, self.original_y1, self.original_x2, self.original_y2 = x1, y1, x2, y2
        self.conf = conf
        self.cls_id = cls_id
        self.view = cur_view
        self.source_view = source_view
        self.image_path = image_path
        self.is_distorted = is_distorted
        self.anti_distortion()
    
    def data(self):
        return [self.x1, self.y1, self.x2, self.y2, self.conf, self.cls_id]

    def anti_distortion(self):
        if self.is_distorted:
            point1 = np.array([[self.x1, self.y1]], dtype=np.float32)
            undistorted_point1 = cv2.undistortPoints(point1, self.view.camera_matrix, self.view.dist_coeffs, P=self.view.new_camera_matrix)
            self.x1, self.y1 = undistorted_point1[0][0]

            point2 = np.array([[self.x2, self.y2]], dtype=np.float32)
            undistorted_point2 = cv2.undistortPoints(point2, self.view.camera_matrix, self.view.dist_coeffs, P=self.view.new_camera_matrix)
            self.x2, self.y2 = undistorted_point2[0][0]

            pointc = np.array([[self.cx, self.cy]], dtype=np.float32)
            undistorted_pointc = cv2.undistortPoints(pointc, self.view.camera_matrix, self.view.dist_coeffs, P=self.view.new_camera_matrix)
            self.cx, self.cy = undistorted_pointc[0][0]

            new_w, new_h = (self.x2-self.x1)/2.0, (self.y2-self.y1)/2.0
            self.x1, self.y1, self.x2, self.y2 = self.cx-new_w, self.cy-new_h, self.cx+new_w, self.cy+new_h
            self.is_distorted = False
        else:
            pass
    
    def projection(self, views_projections, view=MAIN_VIEW):
        assert self.view.name != view
        views_projection = views_projections[f'({view},{self.view.name})']
        new_box = views_projection.box_projection(self)
        return new_box
    
    def is_keep(self, views_projections, view=MAIN_VIEW):
        assert self.source_view.name != view and self.view.name == view
        views_projection = views_projections[f'({view},{self.source_view.name})']
        return views_projection.keep_box(self)

class Point(object):
    def __init__(self, x, y, cls_id, view, is_distorted=True):
        '''
        input:
            x,y: float point location
            view: View Object
            is_distorted: True is the distorted point, else wise
        output: 
            None
        '''
        self.x, self.y = x, y 
        self.original_x, self.original_y = x, y 
        self.cls_id = cls_id
        self.view = view 
        self.is_distorted = is_distorted
        self.anti_distortion()

    def anti_distortion(self):
        '''
        func:
            project the points from distortion to anti_distortion images according to the view distortion matrix
        input:
            None
        output:
            the Point object. convert point from distorted <-> anti_distorted state
        '''

        if self.is_distorted:
            point = np.array([[self.x, self.y]], dtype=np.float32)
            undistorted_point = cv2.undistortPoints(point, self.view.camera_matrix, self.view.dist_coeffs, P=self.view.new_camera_matrix)
            self.original_x, self.original_y = copy.deepcopy(self.x), copy.deepcopy(self.y)
            self.x, self.y = undistorted_point[0][0]
            self.is_distorted = False
        else:
            pass

    def projection(self, views_projections, view=MAIN_VIEW):
        '''
        func:  
            the same as Grid.projection, find the corresponding points on the main view with given points on the view2
        '''
        assert view != self.view.name
        views_projection = views_projections[f'({view},{self.view.name})']
        new_point = views_projection.point_projection(self)
        return new_point

class View(object):
    def __init__(self, name, p1=0, p2=0, k1=0, k2=0, k3=0, fx=0, fy=0, cx=0, cy=0, fx_ratio=1, fy_ratio=1, grid_info=None):
        '''
        input: 
            init matrix of the view camera, used for projection
        output:
            None
        '''
        self.name = name
        self.p1, self.p2 = p1, p2
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.alpha = 1.0
        self.fx_ratio, self.fy_ratio = fx_ratio, fy_ratio

        self.set_camera_matrix()

        self.grid_info = grid_info
        if name == MAIN_VIEW:
            self.set_main_lines()
        else:
            self.set_lines()

        self.set_camera_matrix()
    
    def set_main_lines(self):
        '''
        func:
            set the lines of the  main view
            assume the swim pool is 25m X 15m
            assume the grid is 22 X 8
        '''
        margin_width = MARGIN_WIDTH
        margin_height = MARGIN_HEIGHT
        width = POOL_WIDTH # cm
        height = POOL_HEIGHT # cm
        num_grid_width = 8
        num_grid_height = 22
        left_up, left_bottom, right_up, right_bottom = (margin_width, margin_height), (margin_width, height+margin_height), (width+margin_width, margin_height), (width+margin_width, height+margin_height)
        width_grid = np.linspace(0, width, num_grid_width+1)
        height_grid = np.linspace(0, height, num_grid_height+1)

        top_border = [(x+margin_width, margin_height) for x in width_grid]
        bottom_border = [(x+margin_width, height+margin_height) for x in width_grid]
        left_border = [(margin_width, x+margin_height) for x in height_grid]
        right_border = [(width+margin_width, x+margin_height) for x in height_grid]
        

        self.vertical_lines = []
        self.horizontal_lines = []
        
        num_vertical_lines = len(top_border)
        num_horizontal_lines = len(left_border)
        for i in range(num_vertical_lines):
            top_point = top_border[i]
            bottom_point = bottom_border[i]
            self.vertical_lines.append([top_point, bottom_point])
            self.vertical_num = num_vertical_lines - 2

        for i in range(num_horizontal_lines):
            left_point = left_border[i]
            right_point = right_border[i]
            self.horizontal_lines.append([left_point, right_point])
            self.horizontal_num = num_horizontal_lines - 2


    def anti_distortion(self, image):
        image = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)
        return image

    def set_camera_matrix(self):
        '''
        func:
            set camera matrix for this view, used for point unti-distortion
        input:
            the self p1, ... k3
        output:
            the self camera_matrix roi and mapx mapy
        '''
        w, h = 2560, 1440
        fx = self.fx * (self.fx_ratio if hasattr(self, 'fx_ratio') else 1.0)
        fy = self.fy * (self.fy_ratio if hasattr(self, 'fy_ratio') else 1.0)
        cx, cy = self.cx, self.cy

        self.camera_matrix = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]], dtype=np.float32)

        self.dist_coeffs = np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32)
        
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha=self.alpha, newImgSize=(w, h)
        )

        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, 
            (w, h), cv2.CV_32FC1)

    def set_lines(self):
        '''
        func:
            calculate the annotated lines of the grid, used for calcaluting the grid index
        input:
            self.grid_info
            [
                top_border[pts1[x,y], pts2[x,y], ...],
                bottom_border,
                ...
            ]
        output:
            self.vertical_lines
            [
                line1[[pts1[x,y], pts2[x,y]]],
                line2,
                ...
            ]
            self.horizontal_lines
        '''
        assert self.grid_info is not None
        
        self.vertical_lines = []
        self.horizontal_lines = []

        top_border, bottom_border, left_border, right_border = self.grid_info
        assert len(top_border) == len(bottom_border)
        assert len(left_border) == len(right_border)

        num_vertical_lines = len(top_border)
        num_horizontal_lines = len(left_border)
        for i in range(num_vertical_lines):
            top_point = top_border[i]
            bottom_point = bottom_border[i]
            self.vertical_lines.append([top_point, bottom_point])
            self.vertical_num = num_vertical_lines - 2

        for i in range(num_horizontal_lines):
            left_point = left_border[i]
            right_point = right_border[i]
            self.horizontal_lines.append([left_point, right_point])
            self.horizontal_num = num_horizontal_lines - 2


class LabelData(object):
    def __init__(self, objects=[]):
        '''
        input:
            [(id, x1, y1, x2, y2, conf, cls_id), ...], a list store the labels data of a image sample
        output:
            None
        '''
        self.objects = self.process_objects(objects)
        
    def process_objects(self, objects):
        if isinstance(objects, list):
            return objects
        elif isinstance(objects, np.ndarray):
            return objects.tolist()
        else:
            raise NotImplementedError('Not support this label type!')

    def update(self, object):
        '''
        input:
            (x1, y1, x2, y2, conf, cls_id)
        output: 
            None
        '''
        self.objects.append(object)


class ImageData(object):
    def __init__(self, image_path, labels=None, view=None):
        '''
        input:
            image_path: str, the image path
            labels: LabelData object, used for inference in the system
        output:
            None
        '''
        self.image_path = image_path
        self.labels = labels
        self.view = view

    def update(self, label):
        '''
        func:
            update the labels
        input:
            (id, x1, y1, x2, y2)
        output: 
            None
        '''
        self.labels.update(label)

class ViewAssociation(object):
    def __init__(self, target_view, source_view):
        '''
        input: 
            view1, view2: View objects
            other necessary parameters.
        '''
        self.target_view = target_view
        self.source_view = source_view

        '''
        for point projection:
        [MODIFY] find the corresponding points to calculate the Homography Matrix, need modified corresponding to the views and grid coordinate system
        the source points are points on the associate view, the target points are points on the main view
        '''
        # pdb.set_trace()
        target_points = self.target_view.vertical_lines[0] + self.target_view.vertical_lines[-1]
        # if self.source_view.name in ['1', '2']:
        #     source_points = self.source_view.vertical_lines[0] + self.source_view.vertical_lines[-1]   
        # elif self.source_view.name in ['3', '4']:
        #     source_points = [self.source_view.vertical_lines[-1][1], self.source_view.vertical_lines[-1][0], self.source_view.vertical_lines[0][1], self.source_view.vertical_lines[0][0]]
        #     pdb.set_trace()
        if self.source_view.name == '1':
            source_points = [[132,490],[800,1200],[546,442],[1793,657]] 
        elif self.source_view.name == '2':
            source_points = [[1623,317],[327,603],[2315,341],[1285,1255]]
        elif self.source_view.name == '3':
            source_points = [[2127, 675], [705,411], [953,1251], [272,459]]
        elif self.source_view.name == '4':
            source_points = [[1152, 1272], [2072, 400], [217, 667], [1456, 381]]
        else:
            raise NotImplementedError('Not support this view {self.source_view.name}!')


        self.homography_matrix, _ = cv2.findHomography(np.array(source_points), np.array(target_points))

        # coor for keep box
        self.x_shift, self.y_shift = 0, 0
        if self.source_view.name == '1':
            self.y_shift = 1
        elif self.source_view.name == '2':
            self.x_shift, self.y_shift = 1, 1
        elif self.source_view.name == '3':
            self.x_shift = 1
        elif self.source_view.name == '4':
            pass
        else:
            raise NotImplementedError('Not support this view name!')

    def keep_box(self, box):
        width_half, height_half = POOL_WIDTH / 2.0, POOL_HEIGHT / 2.0
        x_min = MARGIN_WIDTH + width_half * self.x_shift
        x_max = x_min + width_half
        y_min = MARGIN_HEIGHT + height_half * self.y_shift
        y_max = y_min + height_half

        cx, cy = box.cx, box.cy

        if self.x_shift == 0 and self.y_shift == 0:
            return ((cx <= x_max) and (cy <= y_max))
        elif self.x_shift == 1 and self.y_shift == 0:
            return ((cx >= x_min) and (cy <= y_max))
        elif self.x_shift == 0 and self.y_shift == 1:
            return ((cx <= x_max) and (cy >= y_min))
        elif self.x_shift == 1 and self.y_shift == 1:
            return ((cx >= x_min) and (cy >= y_min))
        else:
            raise NotImplementedError('Wrong case!')


    def point_projection(self, point):
        '''
        func:
            given point object of view2, find the corresponding point object in view1
        input:
            Point: point object in view2
        output:
            None, init necessary parameters
        '''
        assert point.view.name == self.source_view.name
        point_homogeneous = np.array([point.x, point.y, 1])
        projected_point_homogeneous = self.homography_matrix @ point_homogeneous
        projected_point = projected_point_homogeneous[:2] / projected_point_homogeneous[2]
        return Point(projected_point[0], projected_point[1], point.cls_id, self.target_view, is_distorted=False)

    def box_projection(self, box):
        assert box.view.name == self.source_view.name
        point_homogeneous1 = np.array([box.x1, box.y1, 1])
        projected_point_homogeneous1 = self.homography_matrix @ point_homogeneous1
        projected_point1 = projected_point_homogeneous1[:2] / projected_point_homogeneous1[2]
        point_homogeneous2 = np.array([box.x2, box.y2, 1])
        projected_point_homogeneous2 = self.homography_matrix @ point_homogeneous2
        projected_point2 = projected_point_homogeneous2[:2] / projected_point_homogeneous2[2]
        new_w1, new_h1 = abs(projected_point2[0]-projected_point1[0])/2.0, abs(projected_point2[1]-projected_point1[1])/2.0

        point_homogeneous1 = np.array([box.x2, box.y1, 1])
        projected_point_homogeneous1 = self.homography_matrix @ point_homogeneous1
        projected_point1 = projected_point_homogeneous1[:2] / projected_point_homogeneous1[2]
        point_homogeneous2 = np.array([box.x1, box.y2, 1])
        projected_point_homogeneous2 = self.homography_matrix @ point_homogeneous2
        projected_point2 = projected_point_homogeneous2[:2] / projected_point_homogeneous2[2]
        new_w2, new_h2 = abs(projected_point2[0]-projected_point1[0])/2.0, abs(projected_point2[1]-projected_point1[1])/2.0

        if new_w1 * new_h1 >= new_w2 * new_h2:
            new_w, new_h = new_w1, new_h1
        else:
            new_w, new_h = new_w2, new_h2

        point_homogeneousc = np.array([box.cx, box.cy, 1])
        projected_point_homogeneousc = self.homography_matrix @ point_homogeneousc
        projected_pointc = projected_point_homogeneousc[:2] / projected_point_homogeneousc[2]
        new_cx, new_cy = projected_pointc[0], projected_pointc[1]

        
        
        new_x1, new_y1, new_x2, new_y2 = new_cx-new_w, new_cy-new_h, new_cx+new_w, new_cy+new_h
        
        return Box(new_x1, new_y1, new_x2, new_y2, box.conf, box.cls_id, self.target_view, box.source_view, box.image_path, is_distorted=box.is_distorted)

class MultiViewAssociationStream(object):
    def __init__(self, grid_root):
        self.grid_root = grid_root
        self.init_views()
        self.views_projections = self.init_view_association()
    
    def init_views(self):
        '''
        func:
            init the view parameter used for association
            init the view name p1~p3, k1~k2
            init the grid-wise weight used for multi view association
        '''
        with open(self.grid_root, 'r') as f:
            grid_info = json.load(f)

        self.main_view = View(MAIN_VIEW, p1=0, p2=0, k1=0, k2=0, k3=0, fx=1280, fy=720, cx=1280, cy=720, fx_ratio=1.0, fy_ratio=1.0, grid_info=None)
        view1 = View('1', p1=0, p2=0, k1=-0.5353, k2=0.2875, k3=-0.0906, fx=1621.9, fy=1856.1, cx=1116.3, cy=742.9178, fx_ratio=1.33, fy_ratio=1.33, grid_info=grid_info[0])
        view2 = View('2', p1=0, p2=0, k1=-0.5153, k2=0.2845, k3=-0.0906, fx=1621.9, fy=1856.1, cx=1116.3, cy=742.9178, fx_ratio=1.34, fy_ratio=1.34, grid_info=grid_info[1])
        view3 = View('3', p1=0, p2=0, k1=-0.5253, k2=0.2845, k3=-0.0906, fx=1621.9, fy=1856.1, cx=1116.3, cy=742.9178, fx_ratio=1.34, fy_ratio=1.34, grid_info=grid_info[2])
        view4 = View('4', p1=0, p2=0, k1=-0.5253, k2=0.2875, k3=-0.0906, fx=1621.9, fy=1856.1, cx=1116.3, cy=742.9178, fx_ratio=1.34, fy_ratio=1.34, grid_info=grid_info[3])
        self.views = [view1, view2, view3, view4]

    def init_view_association(self):
        '''
        func:
            init the view projection of view1 <-> view2, view1 <-> view3, view1 <-> view4
            used for the view projection
        input:
            None 
        output: dict()
            {
            ('1', '2'): ViewAssociation object,
            ('1', '3'): ViewAssociation object,
            ('1', '4'): ViewAssociation object,
            }
        '''
        views_projections = dict()
        for view in self.views:
            views_projections[f"({MAIN_VIEW},{view.name})"] = ViewAssociation(target_view=self.main_view, source_view=view)
        
        return views_projections

    def forward(self, dets):
        '''
        func:
            the main function of multi view association
        input:
            dets: [{image_path: str, det: N*6}, N*6, N*6, N*6]
        output:
        '''
        association_data = AssociationData()
        for idx, view in enumerate(self.views):
            ViewImageData = ImageData(image_path=dets[idx]['image_path'], labels=LabelData(dets[idx]['det']), view=view)
            association_data.LabelData2AssociationData(ViewImageData)
        
        association_data.associate(self.views_projections)
        
        return [x.data() for x in association_data.association_data[self.main_view.name]]

class Drawer(object):
    def __init__(self, w, h, path):
        self.w, self.h = w, h 
        self.path = path
        self.image = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
        
    def set_full_image(self):
        self.full_image = copy.deepcopy(self.image)

    def clear_image(self):
        self.image = copy.deepcopy(self.full_image)

    def draw_image(self, image, position=(0,0)):    # position = (x,y)
        self.image[position[1]*self.h:(position[1]+1)*self.h, position[0]*self.w:(position[0]+1)*self.w] = image
    
    def draw_circle(self, center, position, color, string, radius=5, thickness=3): # center = (x,y)
        cv2.circle(self.image, (int(center[0])+position[0]*self.w, int(center[1])+position[1]*self.h), radius, color, thickness=thickness)
        cv2.putText(self.image, string, (int(center[0])+position[0]*self.w, int(center[1])+position[1]*self.h-20), cv2.FONT_HERSHEY_SIMPLEX, 3, color, thickness=thickness)

    def draw_str(self, string, position, color=(255,255,255), thickness=3):
        cv2.putText(self.image, string, (position[0]*self.w+5, position[1]*self.h+70), cv2.FONT_HERSHEY_SIMPLEX, 3, color, thickness=thickness)


    def save_image(self):
        cv2.imwrite(self.path, self.image)
        print(f"saving image to {self.path}")


class AssociationData(object):
    def __init__(self):
        '''
        input:
            label_data: LabelData object
            distance_threshold: int, mm, when the distance of 2 box larger than distance_threshold, they are 2 objects
        self.association_data:
            {
            view1: [([idx], Point, Grid), ([idx], Point, Grid), ...]
            viewx: [([idx], Point, Grid), ([idx], Point, Grid), ...]
            }   
        '''
        self.association_data = {}
        self.views = []
    
    def LabelData2AssociationData(self, image_data):
        '''
        func:
            project ImageData to the association space and update the data in association space
        input:
            image_data: ImageData object of a view
            view: View object
        output:
            update the self.association_data
        '''
        view = image_data.view
        self.views.append(view)
        self.association_data[view.name] = []

        for label in image_data.labels.objects:
            x1, y1, x2, y2, conf, cls_id = label
            box = Box(x1, y1, x2, y2, conf, cls_id, view, view, image_data.image_path)
            self.association_data[view.name].append(box)

    def associate(self, views_projections):
        '''
        func:
            associate the results of view1 and viewx, with manually set weight, 
            the weight depend on the grid and view, could be initialized on the ViewAssociation object
            could set 1 for all views and all grid in the first version for east implement
            the short distance grid for each view should be larger, vice versa
            associate with the grid index with Hungarian algorithm, update the point location and grid
        input:
            self.association_data:
                {
                view1: [box1, box2, ...]
                viewx: [box1, box2, ...]
                }
        output:
            self.association_data:
                {
                MAIN_VIEW: [box1, box2, ...]
                }
        '''
        main_view_data = []
        for view in self.views:
            view_data = self.association_data[view.name]
            # tmp = []
            for box in view_data:
                main_view_data.append(box.projection(views_projections))
                # tmp.append(box.projection(views_projections))
            # plot box on source view both original and anti_distorted image
            # source_img = cv2.imread(box.image_path)
            # anti_distorted_source_img = view.anti_distortion(source_img)
            # for idx, box in enumerate(view_data):
            #     cv2.rectangle(source_img, (int(box.original_x1), int(box.original_y1)), (int(box.original_x2), int(box.original_y2)), (0,255,0), 3)
            #     cv2.putText(source_img, str(idx), (int(box.original_x1), int(box.original_y1)-20), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), thickness=3)

            # for idx, box in enumerate(view_data):
            #     cv2.rectangle(anti_distorted_source_img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0,255,0), 3)
            #     cv2.putText(anti_distorted_source_img, str(idx), (int(box.x1), int(box.y1)-20), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), thickness=3)

            # is_keeps = [x.is_keep(views_projections) for x in tmp]
            # main_view = np.zeros((3100, 2100, 3), dtype=np.uint8)
            # for idx, box in enumerate(tmp):
            #     cv2.rectangle(main_view, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (0,255,0), 3)
            #     cv2.putText(main_view, str(idx), (int(box.x1), int(box.y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), thickness=3)
            #     cv2.putText(main_view, str(is_keeps[idx]), (int(box.x1)+20, int(box.y1)-20), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), thickness=3)
            
            # cv2.imwrite(f'{view.name}_a1.jpg', source_img)
            # cv2.imwrite(f'{view.name}_a2.jpg', anti_distorted_source_img)
            # cv2.imwrite(f'{view.name}_a3.jpg', main_view)
            # pdb.set_trace() 
        # merge by the region
        keep_objects = [x for x in main_view_data if x.is_keep(views_projections)]
        self.association_data[MAIN_VIEW] = keep_objects

# grid_root = r'calibration_v1.json'
# associator = MultiViewAssociation(grid_root)
# associator.forward()

import os, sys
import numpy as np
import json
import math
import cv2
import copy

from scipy.optimize import linear_sum_assignment
from boxmot.multiview_tool.grid_determine import find_position_between_lines
import random 


import pdb

MAIN_VIEW = '1'

def point_to_line_distance(point, line):
    """
    计算点到直线的距离
    :param point: 点的坐标 (x, y)
    :param line: 直线的两个点 [(x1, y1), (x2, y2)]
    :return: 点到直线的距离
    """
    x, y = point
    (x1, y1), (x2, y2) = line

    # 计算直线的系数 a, b, c
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2

    # 计算点到直线的距离
    distance = abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

    return distance


class Point(object):
    def __init__(self, x, y, view, is_distorted=True):
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
        self.view = view 
        self.is_distorted = is_distorted
        self.grid = self.get_grid()
    
    def get_grid(self):
        '''
        func:
            find the grid index from points of a view
            you should first anti_distortion according to the view distortion matrix
            and find the grid index according to the grid coordinates
        input:
            None
        output:
            the Grid object of points in a view
        '''
        if self.is_distorted:
            self.anti_distortion()
        
        col = find_position_between_lines(self.x, self.y, self.view.vertical_lines, is_vertical=True)
        row = find_position_between_lines(self.x, self.y, self.view.horizontal_lines, is_vertical=False)
        if col is None:
            # pdb.set_trace()
            col = 0 if point_to_line_distance((self.x, self.y), self.view.vertical_lines[0]) < point_to_line_distance((self.x, self.y), self.view.vertical_lines[-1]) else self.view.vertical_num - 2
        if row is None:
            # pdb.set_trace()
            row = 0 if point_to_line_distance((self.x, self.y), self.view.horizontal_lines[0]) < point_to_line_distance((self.x, self.y), self.view.horizontal_lines[-1]) else self.view.horizontal_num - 2
        return Grid(col, row, self.view)   
    
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
            pdb.set_trace()
            self.is_distorted = True
            return Point

    def projection(self, view, views_projections):
        '''
        func:  
            the same as Grid.projection, find the corresponding points on the main view with given points on the view2
        '''
        assert view.name != self.view.name
        assert view.name == MAIN_VIEW
        views_projection = views_projections[f'({view.name},{self.view.name})']
        new_point = views_projection.point_projection(self)
        return new_point
    
    def update(self, point, weight=(1.0,1.0)):
        '''
        func:
            update the object location with given Point and weight provided by the Point.view
        input:
            point: Point object
        output:
            None, update the Point information
        '''
        assert self.is_distorted == point.is_distorted
        new_x = (self.x + point.x * weight[0]) / (1 + weight[0])
        new_y = (self.y + point.y * weight[1]) / (1 + weight[1])
        return Point(new_x, new_y, self.view, self.is_distorted)


class Grid(object):
    def __init__(self, x, y, view):
        '''
        input:
            x, y: int number of object grid index, x is the width index (0~7), y is the height index (0~21)
            view: View object
        output:
            None
        '''
        self.x, self.y = x, y
        self.view = view
    
    def projection(self, view, views_projections):
        '''
        func:
            according to grid view and input view, find the projection grid on the input view
            used for association function
        input: 
            View object, the main view
        output:
            grid: Grid object of the projection grid on input view
        '''
        assert self.view.name != view.name
        assert view.name == MAIN_VIEW # input the main view
        views_projection = views_projections[f'({view.name},{self.view.name})']
        new_grid = views_projection.grid_projection(self)
        return new_grid

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

        self.grid_info = grid_info
        self.set_lines()

        self.set_camera_matrix()
    
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
            [(id, x, y, w, h), ...], a list store the labels data of a image sample
        output:
            None
        '''
        self.objects = []
        self.objects += objects
    
    def update(self, object):
        '''
        input:
            (id, x, y, w, h)
        output: 
            None
        '''
        self.objects.append(object)


class ImageData(object):
    def __init__(self, image_path, labels=None):
        '''
        input:
            image_path: str, the image path
            labels: LabelData object, used for inference in the system
        output:
            None
        '''
        self.image_path = image_path
        self.labels = labels
    
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
    def __init__(self, view1, view2):
        '''
        input: 
            view1, view2: View objects
            other necessary parameters.
        '''
        self.view1 = view1
        self.view2 = view2
        self.x_grid_num = self.view1.vertical_num     
        self.y_grid_num = self.view1.horizontal_num
        assert self.view1.vertical_num == self.view2.vertical_num
        assert self.view1.horizontal_num == self.view2.horizontal_num   # two views have same grid number

        self.x_shift = 0
        self.y_shift = 0

        '''
        for grid projection:
            when x_shift == 1:
                new_x =  x_grid_num - x
            when x_shift == 0:
                new_x = x
        [MODIFY] need modified  corresponding to the views and grid coordinate system
        '''
        if self.view2.name == '2':              
            pass                                # the same grid coordinate system with view 1
        elif self.view2.name == '3':
            self.x_shift, self.y_shift = 1, 1   # inverse grid coordinate system with view 1
        elif self.view2.name == '4':
            self.x_shift, self.y_shift = 1, 1   # inverse grid coordinate system with view 1

        '''
        for point projection:
        [MODIFY] find the corresponding points to calculate the Homography Matrix, need modified corresponding to the views and grid coordinate system
        the source points are points on the associate view, the target points are points on the main view
        '''
        # pdb.set_trace()
        target_points = self.view1.vertical_lines[0] + self.view1.vertical_lines[-1]
        if self.view2.name == '2':
            source_points = self.view2.vertical_lines[0] + self.view2.vertical_lines[-1]      
        elif self.view2.name == '3':
            source_points = [self.view2.vertical_lines[-1][1], self.view2.vertical_lines[-1][0], self.view2.vertical_lines[0][1], self.view2.vertical_lines[0][0]]
        elif self.view2.name == '4':
            source_points = [self.view2.vertical_lines[-1][1], self.view2.vertical_lines[-1][0], self.view2.vertical_lines[0][1], self.view2.vertical_lines[0][0]]

        self.homography_matrix, _ = cv2.findHomography(np.array(source_points), np.array(target_points))

    def grid_projection(self, grid):
        '''
        func:
            given grid object of view2, find the corresponding grid object in view1
        input:
            Grid: grid object in view2
        output:
            None, init necessary parameters
        '''
        assert self.view1.name == MAIN_VIEW
        assert grid.view == self.view2
        new_x = (1 - 2 * self.x_shift) * grid.x + self.x_shift * self.x_grid_num
        new_y = (1 - 2 * self.y_shift) * grid.y + self.y_shift * self.y_grid_num
        return Grid(new_x, new_y, self.view1)

    def point_projection(self, point):
        '''
        func:
            given point object of view2, find the corresponding point object in view1
        input:
            Point: point object in view2
        output:
            None, init necessary parameters
        '''
        # pdb.set_trace()
        assert point.view.name == self.view2.name
        point_homogeneous = np.array([point.x, point.y, 1])
        projected_point_homogeneous = self.homography_matrix @ point_homogeneous
        projected_point = projected_point_homogeneous[:2] / projected_point_homogeneous[2]
        return Point(projected_point[0], projected_point[1], self.view1, is_distorted=False)

class MultiViewAssociation(object):
    def __init__(self, img_root, label_root, grid_root):
        self.img_root = img_root
        self.label_root = label_root
        self.grid_root = grid_root
        self.views = self.init_views()
        self.main_view = self.views[0]
        self.asso_views = self.views[1:]
        self.views_imgs = self.scan_files()
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

        view1 = View('1', p1=0, p2=0, k1=-0.5353, k2=0.2875, k3=-0.0906, fx=1621.9, fy=1856.1, cx=1116.3, cy=742.9178, fx_ratio=1.33, fy_ratio=1.33, grid_info=grid_info[0])
        view2 = View('2', p1=0, p2=0, k1=-0.5153, k2=0.2845, k3=-0.0906, fx=1621.9, fy=1856.1, cx=1116.3, cy=742.9178, fx_ratio=1.34, fy_ratio=1.34, grid_info=grid_info[1])
        view3 = View('3', p1=0, p2=0, k1=-0.5253, k2=0.2845, k3=-0.0906, fx=1621.9, fy=1856.1, cx=1116.3, cy=742.9178, fx_ratio=1.34, fy_ratio=1.34, grid_info=grid_info[2])
        view4 = View('4', p1=0, p2=0, k1=-0.5253, k2=0.2875, k3=-0.0906, fx=1621.9, fy=1856.1, cx=1116.3, cy=742.9178, fx_ratio=1.34, fy_ratio=1.34, grid_info=grid_info[3])
        views = [view1, view2, view3, view4]

        return views

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
        MainView = self.main_view
        for View in self.asso_views:
            views_projections[f"({MainView.name},{View.name})"] = ViewAssociation(MainView, View)
        
        return views_projections

    def read_label(self, view):
        '''
        input: 
            view: str, the name of the view folder
        output:
            view data: [ImageData, ImageData, ...]
        '''
        lines = []
        with open(os.path.join(self.label_root, view, 'normal_seq.txt'), 'r') as f:
            lines += f.readlines()
        
        try:
            with open(os.path.join(self.label_root, view, 'abnormal_seq.txt'), 'r') as f:
                lines += f.readlines()
        except: 
            pass
        lines = sorted(lines)
        images = [os.path.join(self.img_root, view, x) for x in sorted(list(filter(lambda x: x.endswith('.jpg'), list(next(os.walk(os.path.join(self.img_root, view)))[2]))))]
        # pdb.set_trace()
        res = []
        for image in images:
            res.append(ImageData(image, LabelData()))
        
        for line in lines:
            line_data = list(map(int, map(float, line.strip().split(','))))
            frame_id = line_data[0] # start from 1

            if line_data[7] != 0:   # filter the ashore person
                idx = line_data[1]
                x,y,w,h = line_data[2:6]
                res[frame_id-1].update((idx, x, y, w, h))
        return res

    def scan_files(self):
        '''
        input: 
            None
        output:
            {
            view1: [ImageData, ImageData, ...]
            view2: [ImageData, ImageData, ...]
            ...
            }

        '''
        res = dict()
        for view in self.views:
            view_data = self.read_label(view.name)
            res[view.name] = view_data
        self.seq_len = len(res[view.name])
        for view in self.views:
            assert self.seq_len == len(res[view.name]) 
        return res 

    def forward(self, idx):
        '''
        func:
            the main function of multi view association
        input:
            idx: int number, frame id used for test
        output:
        '''
        association_data = AssociationData()
        main_image_data = self.views_imgs[self.main_view.name][idx]
        association_data.LabelData2AssociationData(main_image_data, self.main_view)
        association_data.RandomDropObject(self.main_view)
        print(self.main_view.name, [x[0][0] for x in association_data.association_data[self.main_view.name]])
        for view in self.asso_views:
            ViewImageData = self.views_imgs[view.name][idx]
            association_data.LabelData2AssociationData(ViewImageData, view)
            association_data.RandomDropObject(view)
            print(view.name, [x[0][0] for x in association_data.association_data[view.name]])
            association_data.associate(self.views_projections)

        print('matching result ...')
        for obj in association_data.association_data[self.main_view.name]:
            print(obj[0])       #print the associate results
        
        # self.visualization(association_data)    # if association error happened
    
    def visualization(self, association_data):
        '''
        func:
            [TODO] visualization the four view, including the points/grid, projection points/grid, and assocaition results
        input:
            association_data.vis_association_data:
            {view1: [([idx_view1, idx_viewx],  [Point_view1, Point_viewx], [Grid_view1, Grid_viewx], [img_path_view1, img_path_viewx], [view1, viewx]), ...]}
        '''
        view_position = {'1': (0, 0), '2': (1,0), '3': (0,1), '4': (1,1)}
        # pdb.set_trace()
        data = association_data.vis_association_data[MAIN_VIEW]

        # read all images
        original_image = Drawer(2560, 1440, 'original_image.jpg')
        anti_distorted_image = Drawer(2560, 1440, 'anti_distorted_image.jpg')
        image_paths, views = data[0][3], data[0][4]
        for image_path, view in zip(image_paths, views):
            image = cv2.imread(image_path)
            original_image.draw_image(image, view_position[view.name])
            original_image.draw_str(f"view:{view.name}", view_position[view.name])

            anti_distorted_image.draw_image(view.anti_distortion(image), view_position[view.name])
            anti_distorted_image.draw_str(f"view:{view.name}", view_position[view.name])

            anti_distorted_image.set_full_image()

        for (obj_idxs, points, grids, _, views) in data:
            anti_distorted_image.clear_image()
            # draw object center point on the original image
            for view_idx, view in enumerate(views):
                original_image.draw_circle((points[view_idx].original_x, points[view_idx].original_y), view_position[view.name], (255,0,0), str(obj_idxs[view_idx]))
                anti_distorted_image.draw_circle((points[view_idx].x, points[view_idx].y), view_position[view.name], (255,0,0), str(obj_idxs[view_idx]))
                if view != self.main_view:
                    # pdb.set_trace()
                    project_point = points[view_idx].projection(self.main_view, self.views_projections)
                    anti_distorted_image.draw_circle((project_point.x, project_point.y), view_position[MAIN_VIEW], (0,255,0), str(obj_idxs[view_idx]))
                
            original_image.save_image()
            anti_distorted_image.save_image()
            pdb.set_trace()


        # original points
        
        # num_objects = len(data)

        # for item in 
        # idxs, points, grids, paths, views = zip()
        # for view_id in range(len(idxs)):
        #     idx, point, grid, path, view = idxs[view_id], points[view_id], grids[view_id], paths[view_id], views[view_id]
        #     # draw original points on original view image
        #     image = cv
            

            
        pass

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
        # pdb.set_trace()
        cv2.putText(self.image, string, (position[0]*self.w+5, position[1]*self.h+70), cv2.FONT_HERSHEY_SIMPLEX, 3, color, thickness=thickness)


    def save_image(self):
        cv2.imwrite(self.path, self.image)
        print(f"saving image to {self.path}")


class AssociationData(object):
    def __init__(self):
        '''
        input:
            label_data: LabelData object
        self.association_data:
            {
            view1: [([idx], Point, Grid), ([idx], Point, Grid), ...]
            viewx: [([idx], Point, Grid), ([idx], Point, Grid), ...]
            }   
        self.vis_association_data:  # record the association process used for visualization
            {
            view1: [([idx], [Point], [Grid], [img_path], [view1]), ([idx], [Point], [Grid], [img_path], [view1]), ...]
            viewx: [([idx], [Point], [Grid], [img_path], [viewx]), ([idx], [Point], [Grid], [img_path], [viewx]), ...]
            }
        '''
        self.association_data = {}
        self.vis_association_data = {}
        self.views = []
    
    def LabelData2AssociationData(self, image_data, view):
        '''
        func:
            project ImageData to the association space and update the data in association space
        input:
            image_data: ImageData object of a view
            view: View object
        output:
            update the self.association_data
        '''
        self.association_data[view.name] = []
        self.vis_association_data[view.name] = []
        self.views.append(view)

        for label in image_data.labels.objects:
            object_id, cx, cy = label[0], label[1] + label[3]/2.0, label[2] + label[4]/2.0
            point = Point(cx, cy, view)
            grid = point.grid
            self.association_data[view.name].append(([object_id], point, grid))
            self.vis_association_data[view.name].append(([object_id], [point], [grid], [image_data.image_path], [view]))
    
    def RandomDropObject(self, view):
        '''
        func:
            random drop one object
        input:
            label_data: LabelData object
        self.association_data:
            {
            view1: [([idx], Point, Grid), ([idx], Point, Grid), ...]
            viewx: [([idx], Point, Grid), ([idx], Point, Grid), ...]
            }   
        self.vis_association_data:  # record the association process used for visualization
            {
            view1: [([idx], [Point], [Grid], [img_path], [view1]), ([idx], [Point], [Grid], [img_path], [view1]), ...]
            viewx: [([idx], [Point], [Grid], [img_path], [viewx]), ([idx], [Point], [Grid], [img_path], [viewx]), ...]
            }    
        '''
        self.association_data[view.name].pop(random.choice(range(len(self.association_data))))

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
                view1: [([idx], Point, Grid), ([idx], Point, Grid), ...]
                viewx: [([idx], Point, Grid), ([idx], Point, Grid), ...]
                }
        output:
            self.association_data:
                {
                view1: [([idx_view1, idx_viewx], Point', Grid'), ([idx_view1, idx_viewx], Point', Grid'), ...]
                }
            self.vis_association_data:  # record the association process used for visualization
                {
                view1: [([idx_view1, idx_viewx],  [Point_view1, Point_viewx], [Grid_view1, Grid_viewx], [img_path_view1, img_path_viewx], [view1, viewx]), ...]
                }
        [TODO] save the visualization data
        '''
        # pdb.set_trace()
        assert len(self.views) == 2 and self.views[0].name == MAIN_VIEW
        main_view = self.views[0] 
        associate_view = self.views.pop()

        main_view_data = self.association_data[main_view.name]
        associate_view_data = self.association_data[associate_view.name]

        # 
        main_view_point = [x[1] for x in main_view_data]
        main_view_grid = [x[2] for x in main_view_data]
        main_view_path = [x[3] for x in self.vis_association_data[main_view.name]]

        associate_view_point = [x[1] for x in associate_view_data]
        associate_view_grid = [x[2] for x in associate_view_data]
        associate_view_path = [x[3] for x in self.vis_association_data[associate_view.name]]       

        # grid projection
        associate_view_point_projection = [x[1].projection(main_view, views_projections) for x in associate_view_data]
        associate_view_grid_projection = [x[2].projection(main_view, views_projections) for x in associate_view_data]

        # hungarian_match
        cost_matrix = np.zeros((len(main_view_grid), len(associate_view_grid)))
        # for i, grid_idx1 in enumerate(main_view_point):  #main_view_point or main_view_grid
        #     for j, grid_idx2 in enumerate(associate_view_point_projection):  #associate_view_point_projection or associate_view_grid_projection

        for i, grid_idx1 in enumerate(main_view_grid):
            for j, grid_idx2 in enumerate(associate_view_grid_projection):  
                cost_matrix[i, j] = math.sqrt((grid_idx1.x - grid_idx2.x)**2 + (grid_idx1.y - grid_idx2.y)**2)
        
        # pdb.set_trace()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        match_result = []
        for i, j in zip(row_ind, col_ind):
            matched_item = (
                [main_view_data[i][0][0], associate_view_data[j][0][0]],    # [view1_idx, viewx_idx]
                [main_view_point[i], associate_view_point_projection[j], associate_view_point[j]],  # [view1_point, viewx_point_projection, viewx_point]
                [main_view_grid[i], associate_view_grid_projection[j], associate_view_grid[j]],  # [view1_grid, viewx_grid_projection, viewx_grid]
                [main_view_path[i], associate_view_path[j]]     # [view1_path, viewx_path]
            )
            match_result.append(matched_item)

        # record the visualization result
        for idx, item in enumerate(match_result):
            main_view_id = self.association_data[main_view.name][idx][0][0]
            matched_main_view_id = match_result[idx][0][0]
            assert main_view_id == matched_main_view_id
            matched_asso_view_id = match_result[idx][0][1]

            # pdb.set_trace()
            self.vis_association_data[main_view.name][idx] = (
                self.vis_association_data[main_view.name][idx][0] + [matched_asso_view_id], # [view1_idx, viewx_idx]
                self.vis_association_data[main_view.name][idx][1] + [copy.deepcopy(match_result[idx][1][2])], # [view1_pts, viewx_pts]
                self.vis_association_data[main_view.name][idx][2] + [copy.deepcopy(match_result[idx][2][2])], # [view1_grid, viewx_grid]
                self.vis_association_data[main_view.name][idx][3] + copy.deepcopy(match_result[idx][3][1]), # [view1_path, viewx_path]
                self.vis_association_data[main_view.name][idx][4] + [associate_view]                  # [view1, viewx]
            )


        # update the match_results for association
        for item in match_result:
            item[1][0].update(item[1][1])   # update the point location
            item[2][0] = item[1][0].grid    # updated point grid
        # pdb.set_trace()
        for idx, item in enumerate(match_result):
            main_view_id = self.association_data[main_view.name][idx][0][0]
            matched_main_view_id = match_result[idx][0][0]
            assert main_view_id == matched_main_view_id
            matched_asso_view_id = match_result[idx][0][1]
            self.association_data[main_view.name][idx] = (
                self.association_data[main_view.name][idx][0] + [matched_asso_view_id],  # [view1_idx, viewx_idx]
                item[1][0],                  # Point'
                item[2][0],                  # Grid'
            )             
            # pdb.set_trace()
        


img_root = r'/home/chaoqunwang/swimAD/dataset/dataset_v20250506/afternoon'
label_root = r'/home/chaoqunwang/swimAD/data_transfer/mot/dataset_v20250506/afternoon'
grid_root = r'calibration_v1.json'
associator = MultiViewAssociation(img_root, label_root, grid_root)
for i in range(10):
    associator.forward(i)
    pdb.set_trace()

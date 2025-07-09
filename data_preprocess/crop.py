#!/usr/bin/env python3
"""
图像和标注裁剪脚本
根据指定的四边形区域裁剪图像并过滤标注
"""

import os
import json
import argparse
from pathlib import Path
import cv2
import numpy as np


class RegionCropper:
    """区域裁剪，用于处理图像和标注的裁剪"""
    
    def __init__(self):
        # 定义四个机位的五边形裁剪参数 (x1, y1, x2, y2, x3, y3, x4, y4, x5, y5)
        self.camera_regions = {
            '1': np.array([[67,307], [682,121], [2362,590], [1162,1420], [672,1416]], dtype=np.int32),
            '2': np.array([[207,475], [1906,67], [2511,273], [1776,1430], [1412,1420]], dtype=np.int32),
            '3': np.array([[58,547], [1781,153], [2386,292], [1671,1420], [1061,1425]], dtype=np.int32),
            '4': np.array([[163,288], [898,86], [2544,600], [1719,1440], [1013,1440]], dtype=np.int32)
        }
    
    def point_in_polygon(self, point, polygon):
        """判断点是否在多边形内部"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def calculate_intersection_ratio(self, box_points, polygon):
        """计算box与多边形的交集面积比例"""
        if len(box_points) < 2:
            return 0
        
        x1, y1 = box_points[0]
        x2, y2 = box_points[1]
        
        box_left = min(x1, x2)
        box_right = max(x1, x2)
        box_top = min(y1, y2)
        box_bottom = max(y1, y2)
        
        # 使用采样计算交集面积
        intersection_area = 0
        total_area = 0
        step = 10  # 采样步长
        
        for x in range(int(box_left), int(box_right), step):
            for y in range(int(box_top), int(box_bottom), step):
                total_area += step * step
                if self.point_in_polygon((x + step/2, y + step/2), polygon):
                    intersection_area += step * step
        
        return intersection_area / total_area if total_area > 0 else 0
    
    def crop_image(self, image_path, output_path, polygon):
        """裁剪图像，多边形外区域变为黑色"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像 {image_path}")
                return False
            
            # 创建掩码
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            
            # 应用掩码
            result = image.copy()
            result[mask == 0] = [0, 0, 0]  # 黑色
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存裁剪后的图像
            cv2.imwrite(output_path, result)
            return True
            
        except Exception as e:
            print(f"错误: 处理图像 {image_path} 时出错: {e}")
            return False
    
    def crop_annotation(self, annotation_path, output_path, polygon):
        """裁剪标注，调整边界框或删除小面积框"""
        try:
            # 读取标注文件
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
            
            # 过滤标注
            filtered_shapes = []
            for shape in annotation_data.get('shapes', []):
                if shape.get('shape_type') == 'rectangle':
                    points = shape.get('points', [])
                    if len(points) >= 2:
                        # 计算交集面积比例
                        intersection_ratio = self.calculate_intersection_ratio(points, polygon)
                        
                        # 如果超过40%面积在多边形外（即交集<60%），直接删除
                        if intersection_ratio < 0.6:
                            continue
                        
                        # 通过面积筛选，保留原始box
                        filtered_shapes.append(shape)
                else:
                    # 非矩形标注保留
                    filtered_shapes.append(shape)
            
            # 更新标注数据
            annotation_data['shapes'] = filtered_shapes
            
            # 如果有imageData字段，清空它（因为图像已经被修改）
            if 'imageData' in annotation_data:
                annotation_data['imageData'] = None
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存过滤后的标注
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"错误: 处理标注 {annotation_path} 时出错: {e}")
            return False
    
    def process_camera_folder(self, input_folder, output_folder, camera_id):
        """处理单个机位文件夹"""
        if camera_id not in self.camera_regions:
            print(f"警告: 未找到机位 {camera_id} 的裁剪参数")
            return
        
        polygon = self.camera_regions[camera_id]
        print(f"处理机位 {camera_id}...")
        
        # 遍历输入文件夹中的所有文件
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                input_file_path = os.path.join(root, file)
                
                # 计算相对路径
                rel_path = os.path.relpath(input_file_path, input_folder)
                output_file_path = os.path.join(output_folder, rel_path)
                
                file_ext = os.path.splitext(file)[1].lower()
                
                # 处理图像文件
                if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    if self.crop_image(input_file_path, output_file_path, polygon):
                        print(f"  已处理图像: {rel_path}")
                    else:
                        print(f"  图像处理失败: {rel_path}")
                
                # 处理标注文件
                elif file_ext == '.json':
                    if self.crop_annotation(input_file_path, output_file_path, polygon):
                        print(f"  已处理标注: {rel_path}")
                    else:
                        print(f"  标注处理失败: {rel_path}")
                
                # 其他文件跳过，不处理
                else:
                    print(f"  跳过文件: {rel_path}")
    
    def process_dataset(self, input_dir, output_dir):
        """处理整个数据集"""
        input_path = Path(input_dir)
        
        # 获取输入目录名称并添加_crop后缀
        input_dir_name = input_path.name
        crop_dir_name = f"{input_dir_name}_crop"
        output_path = Path(output_dir) / crop_dir_name
        
        if not input_path.exists():
            print(f"错误: 输入目录不存在: {input_dir}")
            return
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"输出目录: {output_path}")
        
        # 遍历输入目录下的所有子文件夹
        processed_folders = 0
        for sub_folder in input_path.iterdir():
            if not sub_folder.is_dir():
                continue
            
            print(f"\n处理文件夹: {sub_folder.name}")
            
            # 在每个子文件夹中查找机位文件夹
            camera_folders = {}
            for item in sub_folder.iterdir():
                if item.is_dir() and item.name in ['1', '2', '3', '4']:
                    camera_folders[item.name] = item
            
            if not camera_folders:
                print(f"  警告: 在文件夹 {sub_folder.name} 中未找到机位文件夹 (1, 2, 3, 4)")
                continue
            
            # 创建对应的输出子文件夹
            output_sub_folder = output_path / sub_folder.name
            
            # 处理每个机位
            for camera_id, folder_path in camera_folders.items():
                output_camera_folder = output_sub_folder / camera_id
                self.process_camera_folder(
                    str(folder_path),
                    str(output_camera_folder),
                    camera_id
                )
            
            processed_folders += 1
        
        if processed_folders == 0:
            print("警告: 未找到任何包含机位文件夹的子目录")
        else:
            print(f"\n处理完成! 共处理了 {processed_folders} 个文件夹，结果保存在: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='根据指定区域裁剪图像和标注',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    python crop.py -i ./input_data -o ./output_data
    python crop.py --input ./dataset --output ./cropped_dataset
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入数据目录路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=False,
        default=None,
        help='输出数据目录路径 (默认与输入路径同地址)'
    )
    
    args = parser.parse_args()
    
    # 如果没有指定输出路径，使用输入路径的父目录
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent)
    
    # 创建裁剪器并处理数据
    cropper = RegionCropper()
    cropper.process_dataset(args.input, args.output)


if __name__ == '__main__':
    main()

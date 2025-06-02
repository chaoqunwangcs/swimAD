#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
泳池图像反畸变处理脚本
用于校正相机畸变，为后续的泳道标定提供准确的图像基础
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import io
from typing import Union, Tuple, Optional


class AntiDistortion:
    """反畸变处理类"""
    
    def __init__(self):
        """初始化反畸变处理器，加载相机标定参数"""
        # 相机矩阵 (Camera Matrix)
        self.camera_matrix = np.array([
            [1621.9, 0, 1116.3],
            [0, 1856.1, 742.9],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 畸变系数 (Distortion Coefficients: k1, k2, p1, p2, k3)
        self.dist_coeffs = np.array([
            -0.4893, 0.2845, 0.00016542, 0.00076033, -0.0906
        ], dtype=np.float32)
        
        # 优化的新相机矩阵（用于getOptimalNewCameraMatrix）
        self.new_camera_matrix = None
        self.roi = None
        
    def setup_optimal_camera_matrix(self, image_size: Tuple[int, int], alpha: float = 1.0):
        """
        设置优化的相机矩阵
        
        Args:
            image_size: 图像尺寸 (width, height)
            alpha: 自由缩放参数，0表示裁剪所有无效像素，1表示保留所有像素
        """
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, image_size, alpha
        )
    
    def undistort_image(self, image: np.ndarray, crop_valid_area: bool = False) -> np.ndarray:
        """
        对输入图像进行反畸变处理
        
        Args:
            image: 输入的畸变图像 (numpy array)
            crop_valid_area: 是否裁剪有效区域
            
        Returns:
            处理后的反畸变图像
        """
        h, w = image.shape[:2]
        
        # 如果还没有设置优化的相机矩阵，则设置它
        if self.new_camera_matrix is None:
            self.setup_optimal_camera_matrix((w, h))
        
        # 执行反畸变
        undistorted = cv2.undistort(
            image, 
            self.camera_matrix, 
            self.dist_coeffs, 
            None, 
            self.new_camera_matrix
        )
          # 如果需要，裁剪到有效区域（扩大roi以保留更多有效区域）
        if crop_valid_area and self.roi is not None:
            x, y, w_roi, h_roi = self.roi
            
            # 扩大roi区域的边距（像素）
            roi_margin = 220
            
            # 调整roi坐标，确保不超出图像边界
            img_height, img_width = undistorted.shape[:2]
            x_expanded = max(0, x - roi_margin)
            y_expanded = max(0, y - roi_margin)
            w_roi_expanded = min(img_width - x_expanded, w_roi + 2 * roi_margin)
            h_roi_expanded = min(img_height - y_expanded, h_roi + 2 * roi_margin)
            
            if w_roi_expanded > 0 and h_roi_expanded > 0:
                undistorted = undistorted[y_expanded:y_expanded+h_roi_expanded, 
                                        x_expanded:x_expanded+w_roi_expanded]
        
        return undistorted
    
    def undistort_image_from_path(self, input_path: str, output_path: Optional[str] = None, 
                                 crop_valid_area: bool = False) -> np.ndarray:
        """
        从文件路径读取图像并进行反畸变处理
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径，如果为None则不保存
            crop_valid_area: 是否裁剪有效区域
            
        Returns:
            处理后的反畸变图像
        """
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入图像文件不存在: {input_path}")
        
        # 读取图像
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"无法读取图像文件: {input_path}")
        
        # 执行反畸变
        undistorted = self.undistort_image(image, crop_valid_area)
        
        # 如果指定了输出路径，保存结果
        if output_path:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            success = cv2.imwrite(output_path, undistorted)
            if not success:
                raise RuntimeError(f"保存图像失败: {output_path}")
            print(f"反畸变图像已保存到: {output_path}")
        
        return undistorted
    
    def undistort_image_from_bytes(self, image_bytes: bytes, 
                                  output_format: str = 'jpg') -> bytes:
        """
        从字节流读取图像并进行反畸变处理，返回处理后的字节流
        用于Web服务调用
        
        Args:
            image_bytes: 输入图像的字节流
            output_format: 输出格式 ('jpg', 'png')
            
        Returns:
            处理后图像的字节流
        """
        # 从字节流解码图像
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("无法解码输入的图像字节流")
          # 执行反畸变
        undistorted = self.undistort_image(image, crop_valid_area=True)
        
        # 编码为字节流
        if output_format.lower() in ['jpg', 'jpeg']:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, buffer = cv2.imencode('.jpg', undistorted, encode_param)
        elif output_format.lower() == 'png':
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 1]
            _, buffer = cv2.imencode('.png', undistorted, encode_param)
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")
        
        return buffer.tobytes()
    
    def get_camera_info(self) -> dict:
        """
        获取相机标定信息
        
        Returns:
            包含相机参数的字典
        """
        return {
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coefficients': self.dist_coeffs.tolist(),
            'new_camera_matrix': self.new_camera_matrix.tolist() if self.new_camera_matrix is not None else None,
            'roi': self.roi
        }


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description='泳池图像反畸变处理工具')
    parser.add_argument('input', help='输入图像路径')
    parser.add_argument('-o', '--output', help='输出图像路径（可选）')
    parser.add_argument('--no-crop', action='store_true', help='不裁剪有效区域')
    parser.add_argument('--show', action='store_true', help='显示处理前后对比')
    parser.add_argument('--info', action='store_true', help='显示相机标定信息')
    
    args = parser.parse_args()
    
    # 创建反畸变处理器
    anti_distortion = AntiDistortion()
    
    # 如果只是查看相机信息
    if args.info:
        info = anti_distortion.get_camera_info()
        print("相机标定信息:")
        print(f"相机矩阵:\n{info['camera_matrix']}")
        print(f"畸变系数: {info['distortion_coefficients']}")
        return
    
    try:
        # 处理图像
        crop_valid_area = not args.no_crop
        undistorted = anti_distortion.undistort_image_from_path(
            args.input, 
            args.output, 
            crop_valid_area
        )
        
        print(f"成功处理图像: {args.input}")
        print(f"输出图像尺寸: {undistorted.shape[1]}x{undistorted.shape[0]}")
        
        # 如果需要显示对比
        if args.show:
            original = cv2.imread(args.input)
            if original is not None:
                # 调整窗口大小以便显示
                display_height = 600
                h_orig, w_orig = original.shape[:2]
                display_width = int(w_orig * display_height / h_orig)
                
                original_resized = cv2.resize(original, (display_width, display_height))
                undistorted_resized = cv2.resize(undistorted, (display_width, display_height))
                
                # 水平拼接显示
                comparison = np.hstack([original_resized, undistorted_resized])
                
                cv2.imshow('原图 vs 反畸变图 (按任意键关闭)', comparison)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


# Web服务调用接口
def process_image_bytes(image_bytes: bytes, output_format: str = 'jpg') -> bytes:
    """
    Web服务调用的简化接口
    
    Args:
        image_bytes: 输入图像字节流
        output_format: 输出格式
        
    Returns:
        处理后的图像字节流
    """
    anti_distortion = AntiDistortion()
    return anti_distortion.undistort_image_from_bytes(image_bytes, output_format)


if __name__ == "__main__":
    main()
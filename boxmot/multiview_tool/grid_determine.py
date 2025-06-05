# -*- coding: utf-8 -*-
"""
泳池网格判定脚本 - 修正版本
实现基于直线判定的网格判定算法
"""
import json
import os
import argparse
import sys
from typing import List, Optional


class GridDeterminer:
    """网格判定处理类 - 修正版本"""

    def __init__(self, calibration_file_path: str = None):
        """
        初始化网格判定器

        Args:
            calibration_file_path: 标定文件路径，如果为None则使用默认路径
        """
        # 硬编码的默认标定文件路径（在实际部署时需要修改为正确路径）
        if calibration_file_path is None:
            calibration_file_path = "calibration_data.txt"

        self.calibration_file_path = calibration_file_path
        self.all_calibration_data = None

    def load_calibration_data(self):
        """从文件加载标定数据"""
        try:
            if not os.path.exists(self.calibration_file_path):
                raise FileNotFoundError(f"标定文件不存在: {self.calibration_file_path}")

            with open(self.calibration_file_path, 'r', encoding='utf-8') as f:
                self.all_calibration_data = json.load(f)

            # 验证数据结构
            self._validate_calibration_data()
            print(f"成功加载标定数据: {self.calibration_file_path}")

        except Exception as e:
            print(f"加载标定数据失败: {e}")
            self.all_calibration_data = None
            raise
    
    def _validate_calibration_data(self):
        """验证标定数据格式"""
        if not isinstance(self.all_calibration_data, list):
            raise ValueError("标定数据必须是列表格式")

        # 检查是否有4个相机视角的数据
        if len(self.all_calibration_data) != 4:
            raise ValueError(f"标定数据必须包含4个相机视角，当前有{len(self.all_calibration_data)}个")

        for camera_idx, camera_data in enumerate(self.all_calibration_data):
            if not isinstance(camera_data, list) or len(camera_data) != 4:
                raise ValueError(f"相机{camera_idx}的数据格式错误，应包含4个边框数据")
                
            # 检查每个边框的数据格式
            border_names = ["上边框", "下边框", "左边框", "右边框"]
            for border_idx, border in enumerate(camera_data):
                if not isinstance(border, list):
                    raise ValueError(f"相机{camera_idx}的{border_names[border_idx]}必须是列表格式")
                
                # 可以为空，但如果不为空则必须是点坐标列表
                for point_idx, point in enumerate(border):
                    if not isinstance(point, list) or len(point) != 2:
                        raise ValueError(f"相机{camera_idx}的{border_names[border_idx]}第{point_idx}个点格式错误")
                    
                    if not all(isinstance(coord, (int, float)) for coord in point):
                        raise ValueError(f"相机{camera_idx}的{border_names[border_idx]}第{point_idx}个点坐标必须是数字")
    
    def set_calibration_data(self, calibration_data: List):
        """
        直接设置标定数据（用于Web服务中的会话数据）

        Args:
            calibration_data: 标定数据
        """
        self.all_calibration_data = calibration_data
        self._validate_calibration_data()  # 验证数据格式

    def _calculate_dividing_lines(self, camera_idx: int) -> tuple:
        """
        第一步：计算所有分割线
        
        竖线（垂直分割线）= 上下边框对应点连接
        横线（水平分割线）= 左右边框对应点连接
        
        Args:
            camera_idx: 相机索引 (0-3)
            
        Returns:
            tuple: (horizontal_lines, vertical_lines)
                - horizontal_lines: 水平分割线列表，每条线由两个点组成 [(x1,y1), (x2,y2)]
                - vertical_lines: 垂直分割线列表，每条线由两个点组成 [(x1,y1), (x2,y2)]
        """
        if self.all_calibration_data is None:
            return [], []
            
        camera_data = self.all_calibration_data[camera_idx]
        top_border, bottom_border, left_border, right_border = camera_data
        
        # 计算垂直分割线：上下边框对应点连接
        vertical_lines = []
        min_len = min(len(top_border), len(bottom_border))
        for i in range(min_len):
            top_point = top_border[i]
            bottom_point = bottom_border[i]
            vertical_lines.append([top_point, bottom_point])
        
        # 计算水平分割线：左右边框对应点连接
        horizontal_lines = []
        min_len = min(len(left_border), len(right_border))
        for i in range(min_len):
            left_point = left_border[i]
            right_point = right_border[i]
            horizontal_lines.append([left_point, right_point])
        
        return horizontal_lines, vertical_lines

    def determine_grid_cell(self, camera_idx: int, x: float, y: float) -> Optional[List[int]]:
        """
        判定点击点位于哪个网格单元

        Args:
            camera_idx: 相机索引 (0-3)
            x: 点击点x坐标
            y: 点击点y坐标

        Returns:
            [col, row] 网格索引 或 None（如果不在任何网格内）
        """
        # 第一步：计算分割线
        horizontal_lines, vertical_lines = self._calculate_dividing_lines(camera_idx)
        
        print(f"相机{camera_idx}: 水平线{len(horizontal_lines)}条, 垂直线{len(vertical_lines)}条")
        print("水平线：", end="")
        for i, line in enumerate(horizontal_lines):
            print(f" 线{i}:({line[0][0]:.1f},{line[0][1]:.1f})-({line[1][0]:.1f},{line[1][1]:.1f})", end="")
        print()
        print("垂直线：", end="")
        for i, line in enumerate(vertical_lines):
            print(f" 线{i}:({line[0][0]:.1f},{line[0][1]:.1f})-({line[1][0]:.1f},{line[1][1]:.1f})", end="")
        print()
          # 第二步：判定点在哪个网格内
        return self._determine_grid_position(x, y, horizontal_lines, vertical_lines)

    def get_grid_info(self, camera_idx: int) -> dict:
        """
        获取指定相机的网格信息

        Args:
            camera_idx: 相机索引

        Returns:
            网格信息字典
        """
        if (self.all_calibration_data is None or
            camera_idx < 0 or
            camera_idx >= len(self.all_calibration_data)):
            return {}

        camera_data = self.all_calibration_data[camera_idx]
        
        # 计算网格数量
        h_cells = len(camera_data[0]) - 1 if len(camera_data[0]) > 1 else 0  # 水平网格数量
        v_cells = len(camera_data[2]) - 1 if len(camera_data[2]) > 1 else 0  # 垂直网格数量

        return {
            'camera_idx': camera_idx,
            'top_border_points': len(camera_data[0]),
            'bottom_border_points': len(camera_data[1]),
            'left_border_points': len(camera_data[2]),
            'right_border_points': len(camera_data[3]),
            'horizontal_cells': h_cells if h_cells > 0 else 0,
            'vertical_cells': v_cells if v_cells > 0 else 0,
            'total_cells': h_cells * v_cells if h_cells > 0 and v_cells > 0 else 0
        }
    
    def print_border_points(self, camera_idx: int):
        """
        打印指定相机的四条边界上的所有点
        
        Args:
            camera_idx: 相机索引 (0-3)
        """
        if self.all_calibration_data is None:
            print("错误: 标定数据未加载")
            return
            
        if camera_idx < 0 or camera_idx >= len(self.all_calibration_data):
            print(f"错误: 相机索引{camera_idx}无效，应该在0-3范围内")
            return
            
        camera_data = self.all_calibration_data[camera_idx]
        border_names = ["上边框", "下边框", "左边框", "右边框"]
        
        print(f"=== 相机 {camera_idx} 的边界点信息 ===")
        
        for border_idx, (border_name, border_points) in enumerate(zip(border_names, camera_data)):
            print(f"\n{border_name} ({len(border_points)} 个点):")
            
            if not border_points:
                print("  (无点数据)")
                continue
                
            for i, point in enumerate(border_points):
                print(f"  点{i}: ({point[0]:.2f}, {point[1]:.2f})")
                
            # 计算边界范围
            if border_points:
                x_coords = [p[0] for p in border_points]
                y_coords = [p[1] for p in border_points]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                print(f"  X范围: {min_x:.2f} ~ {max_x:.2f}")
                print(f"  Y范围: {min_y:.2f} ~ {max_y:.2f}")
        
        # 打印网格信息
        grid_info = self.get_grid_info(camera_idx)
        print(f"\n=== 网格信息 ===")
        print(f"水平网格数: {grid_info['horizontal_cells']}")
        print(f"垂直网格数: {grid_info['vertical_cells']}")
        print(f"总网格数: {grid_info['total_cells']}")

    def _determine_grid_position(self, x: float, y: float, horizontal_lines: List, vertical_lines: List) -> Optional[List[int]]:
        """
        第二步：判定点在哪个网格内
        
        Args:
            x, y: 待判定的点坐标
            horizontal_lines: 水平分割线列表
            vertical_lines: 垂直分割线列表
            
        Returns:
            [col, row] 网格索引 或 None（如果不在任何网格内）
        """
        if not horizontal_lines or not vertical_lines:
            print("错误: 分割线数据为空")
            return None
        
        # 判定点在哪两条垂直线之间（决定列索引）
        col = self._find_position_between_lines(x, y, vertical_lines, is_vertical=True)
        if col is None:
            print(f"点({x:.2f}, {y:.2f}) 不在任何垂直分割线之间")
            return None
        
        # 判定点在哪两条水平线之间（决定行索引）
        row = self._find_position_between_lines(x, y, horizontal_lines, is_vertical=False)
        if row is None:
            print(f"点({x:.2f}, {y:.2f}) 不在任何水平分割线之间")
            return None
        
        print(f"点({x:.2f}, {y:.2f}) 位于网格[{col}, {row}]")
        return [col, row]
    
    def _find_position_between_lines(self, x: float, y: float, lines: List, is_vertical: bool) -> Optional[int]:
        """
        判定点在哪两条线之间
        
        Args:
            x, y: 待判定的点坐标
            lines: 分割线列表，每条线由两个端点组成
            is_vertical: True表示垂直线，False表示水平线
            
        Returns:
            线段索引（0到len(lines)-1）或None（如果不在任何线段之间）
        """
        for i in range(len(lines) - 1):
            line1 = lines[i]
            line2 = lines[i + 1]
            
            if is_vertical:
                # 垂直线：判断点是否在两条垂直线之间
                if self._point_between_vertical_lines(x, y, line1, line2):
                    return i
            else:
                # 水平线：判断点是否在两条水平线之间
                if self._point_between_horizontal_lines(x, y, line1, line2):
                    return i
        
        return None
    
    def _point_between_vertical_lines(self, x: float, y: float, line1: List, line2: List) -> bool:
        """
        判断点是否在两条垂直线之间
        
        Args:
            x, y: 待判定的点坐标
            line1, line2: 两条垂直线，每条线由两个端点组成
            
        Returns:
            True如果点在两条线之间
        """
        # 计算点到两条垂直线的相对位置
        # 对于垂直线，我们需要判断点的x坐标相对于线的x坐标位置
          # 获取两条线在点的y坐标处的x坐标
        x1_at_y = self._get_x_at_y(line1, y)
        x2_at_y = self._get_x_at_y(line2, y)
        
        if x1_at_y is None or x2_at_y is None:
            return False
          # 确保x1 <= x2
        if x1_at_y > x2_at_y:
            x1_at_y, x2_at_y = x2_at_y, x1_at_y
        
        # 判断点是否在两条线之间
        result = x1_at_y <= x <= x2_at_y
        return result
    
    def _point_between_horizontal_lines(self, x: float, y: float, line1: List, line2: List) -> bool:
        """
        判断点是否在两条水平线之间
        
        Args:
            x, y: 待判定的点坐标
            line1, line2: 两条水平线，每条线由两个端点组成
            
        Returns:
            True如果点在两条线之间
        """
        # 计算点到两条水平线的相对位置
        # 对于水平线，我们需要判断点的y坐标相对于线的y坐标位置
        
        # 获取两条线在点的x坐标处的y坐标
        y1_at_x = self._get_y_at_x(line1, x)
        y2_at_x = self._get_y_at_x(line2, x)
        
        if y1_at_x is None or y2_at_x is None:
            return False
        
        # 确保y1 <= y2
        if y1_at_x > y2_at_x:
            y1_at_x, y2_at_x = y2_at_x, y1_at_x
          # 判断点是否在两条线之间
        result = y1_at_x <= y <= y2_at_x
        return result
    
    def _get_x_at_y(self, line: List, y: float) -> Optional[float]:
        """
        获取直线在指定y坐标处的x坐标
        
        Args:
            line: 直线的两个端点 [(x1, y1), (x2, y2)]
            y: 指定的y坐标
            
        Returns:
            x坐标值或None（如果直线不经过该y坐标）
        """
        (x1, y1), (x2, y2) = line
        
        # 如果是水平线
        if y1 == y2:
            if y == y1:
                return (x1 + x2) / 2  # 返回中点x坐标
            else:
                return None
        
        # 线性插值计算x坐标
        t = (y - y1) / (y2 - y1)
        x = x1 + t * (x2 - x1)
        return x
    
    def _get_y_at_x(self, line: List, x: float) -> Optional[float]:
        """
        获取直线在指定x坐标处的y坐标
        
        Args:
            line: 直线的两个端点 [(x1, y1), (x2, y2)]
            x: 指定的x坐标
            
        Returns:
            y坐标值或None（如果直线不经过该x坐标）
        """
        (x1, y1), (x2, y2) = line
        
        # 如果是垂直线
        if x1 == x2:
            if x == x1:
                return (y1 + y2) / 2  # 返回中点y坐标
            else:
                return None        # 如果x不在线段的x范围内，我们延长直线进行计算
        # 注释掉原来的范围限制，允许直线延长
        # if not (min(x1, x2) <= x <= max(x1, x2)):
        #     return None
        
        # 线性插值计算y坐标
        t = (x - x1) / (x2 - x1)
        y = y1 + t * (y2 - y1)
        return y

    # ...existing code...
def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description='泳池网格判定工具')
    parser.add_argument('camera_idx', type=int, help='相机索引 (0-3)')
    parser.add_argument('x', type=float, help='点击点x坐标')
    parser.add_argument('y', type=float, help='点击点y坐标')
    parser.add_argument('-c', '--calibration', help='标定文件路径', default='calibration_data.txt')
    parser.add_argument('--info', action='store_true', help='显示网格信息')
    parser.add_argument('--borders', action='store_true', help='显示边界点信息')

    args = parser.parse_args()

    try:
        grid_determiner = GridDeterminer(args.calibration)
        grid_determiner.load_calibration_data()

        if args.info:
            info = grid_determiner.get_grid_info(args.camera_idx)
            print("网格信息:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            return
            
        if args.borders:
            grid_determiner.print_border_points(args.camera_idx)
            return

        result = grid_determiner.determine_grid_cell(args.camera_idx, args.x, args.y)
        
        if result is not None:
            print(f"点 ({args.x}, {args.y}) 位于网格单元: {result}")
        else:
            print(f"点 ({args.x}, {args.y}) 不在任何网格单元内")

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


# Web服务调用接口
def determine_grid_cell_from_data(camera_idx: int, x: float, y: float,
                                 calibration_data: List) -> Optional[List[int]]:
    """
    Web服务调用的简化接口

    Args:
        camera_idx: 相机索引
        x: x坐标
        y: y坐标
        calibration_data: 标定数据

    Returns:
        [col, row] 网格索引 或 None
    """
    try:
        grid_determiner = GridDeterminer()
        grid_determiner.set_calibration_data(calibration_data)
        return grid_determiner.determine_grid_cell(camera_idx, x, y)
    except Exception as e:
        print(f"Error in determine_grid_cell_from_data: {e}")
        return None


if __name__ == "__main__":
    # 如果有命令行参数则运行main，否则运行简单测试
    if len(sys.argv) > 1:
        main()
    else:
        # 简单的测试函数
        def run_test():
            print("--- 网格判定修正测试 ---")
            calibration_file_path = r"E:\mlvision\swimAD\multi_view_v1\temp\test_5.31.json"

            try:
                if not os.path.exists(calibration_file_path):
                    print(f"标定文件不存在: {calibration_file_path}")
                    return

                with open(calibration_file_path, 'r', encoding='utf-8') as f:
                    calibration_data = json.load(f)

                # 创建网格判定器
                grid_determiner = GridDeterminer()
                grid_determiner.set_calibration_data(calibration_data)
                
                # 查看标定数据信息
                for camera_idx in range(4):
                    grid_info = grid_determiner.get_grid_info(camera_idx)
                    print(f"\n相机{camera_idx}的网格信息:")
                    for key, value in grid_info.items():
                        print(f"  {key}: {value}")
                  # 测试网格判定功能
                test_points = [
                    (400.0, 230.0),  # 可能在[0,0]
                    (600.0, 220.0),  # 可能在[0,1]或[1,0]
                    (806.4, 215.0),  # 已知在[1,1]
                    (900.0, 250.0),  # 可能在[1,1]
                    (300.0, 200.0),  # 边界测试
                    (815.0, 277.0),  # 已知不在网格内
                ]
                
                print(f"\n=== 测试分割线计算 ===")
                for x, y in test_points:
                    print(f"\n测试点({x}, {y}):")
                    result = grid_determiner.determine_grid_cell(0, x, y)
                    print(f"  结果: {result}")

            except Exception as e:
                print(f"测试失败: {e}")
                import traceback
                traceback.print_exc()

        run_test()

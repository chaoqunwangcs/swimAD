# -*- coding: utf-8 -*-
"""
游泳池车道校准数据生成和可视化Web服务
Flask后端API服务
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import json
import base64
import io
import os
from PIL import Image
import numpy as np
from anti_distortion import AntiDistortion, process_image_bytes
from grid_determine import GridDeterminer, determine_grid_cell_from_data

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局配置
UPLOAD_FOLDER = 'uploads'
CALIBRATION_FILE = 'calibration_data.txt'

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 全局变量存储会话数据
session_data = {
    'calibration_data': None,
    'anti_distortion_params': None,  # 存储反畸变参数
    'current_camera': 0,
    'images': {}  # 存储上传的图像
}


@app.route('/')
def index():
    """主页面"""
    return render_template('index2.html')


@app.route('/api/upload_anti_distortion_params', methods=['POST'])
def upload_anti_distortion_params():
    """
    反畸变参数上传API端点
    
    接收:
    - anti_distortion_file: JSON文件包含反畸变参数
    
    返回:
    - success: 布尔值
    - message: 操作消息
    """
    try:
        if 'anti_distortion_file' not in request.files:
            return jsonify({
                'success': False,
                'message': '没有接收到反畸变参数文件'
            }), 400
        
        file = request.files['anti_distortion_file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': '文件名为空'
            }), 400
        
        # 读取并解析JSON文件
        file_content = file.read().decode('utf-8')
        anti_distortion_params = json.loads(file_content)
        
        # 验证参数格式
        if not validate_anti_distortion_params(anti_distortion_params):
            return jsonify({
                'success': False,
                'message': '反畸变参数文件格式错误'
            }), 400
        
        # 存储到会话数据
        session_data['anti_distortion_params'] = anti_distortion_params
        
        return jsonify({
            'success': True,
            'message': '反畸变参数上传成功'
        })
    
    except json.JSONDecodeError:
        return jsonify({
            'success': False,
            'message': 'JSON文件格式错误'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'处理错误: {str(e)}'
        }), 500


def validate_anti_distortion_params(params):
    """
    验证反畸变参数格式
    
    参数:
    - params: 待验证的参数字典
    
    返回:
    - bool: 验证结果
    """
    required_views = ['view1', 'view2', 'view3', 'view4']
    required_keys = ['p1', 'p2', 'k1', 'k2', 'k3', 'fx_ratio', 'fy_ratio']
    
    for view in required_views:
        if view not in params:
            return False
        for key in required_keys:
            if key not in params[view] or not isinstance(params[view][key], (int, float)):
                return False
    return True


def get_anti_distortion_params_for_view(view_index):
    """
    根据view索引获取对应的反畸变参数
    
    参数:
    - view_index: 相机索引 (0-3)
    
    返回:
    - dict: 对应view的反畸变参数，如果没有则返回None
    """
    if session_data['anti_distortion_params'] is None:
        return None
    
    view_name = f'view{view_index + 1}'  # 0->view1, 1->view2, etc.
    return session_data['anti_distortion_params'].get(view_name)


@app.route('/api/undistort_image', methods=['POST'])
def undistort_image():
    """
    图像反畸变API端点
    
    接收:
    - image: base64编码的图像数据或文件上传
    - camera_idx: 相机索引 (0-3, 可选)
    
    返回:
    - success: 布尔值
    - message: 操作消息    - undistorted_image: base64编码的反畸变图像（如果成功）
    """
    try:
        # 获取相机索引 - 安全地处理不同的请求格式
        camera_idx = 0  # 默认值
        
        # 优先从form数据获取
        if request.form.get('camera_idx'):
            camera_idx = int(request.form.get('camera_idx', 0))
        # 如果是JSON请求，从JSON获取
        elif request.is_json and request.json and 'camera_idx' in request.json:
            camera_idx = int(request.json.get('camera_idx', 0))
          # 获取对应相机的反畸变参数
        anti_distortion_params = get_anti_distortion_params_for_view(camera_idx)
        
        # 调试信息：记录使用的相机索引和参数来源
        print(f"Debug: camera_idx={camera_idx}, using {'uploaded' if anti_distortion_params else 'default'} params")
        
        # 检查请求中是否有图像数据
        image_data = None
        
        if 'image' in request.files:
            # 文件上传方式
            file = request.files['image']
            if file.filename != '':
                image_data = file.read()
        elif request.is_json and request.json and 'image_base64' in request.json:
            # base64方式
            base64_str = request.json['image_base64']
            if base64_str.startswith('data:image/'):
                # 移除数据URL前缀
                base64_str = base64_str.split(',')[1]
            image_data = base64.b64decode(base64_str)
        else:
            return jsonify({
                'success': False,
                'message': '没有接收到图像数据'
            }), 400
        
        if not image_data:
            return jsonify({
                'success': False,
                'message': '图像数据为空'
            }), 400
          # 调用反畸变处理，传递反畸变参数
        if anti_distortion_params:
            undistorted_data = process_image_bytes(image_data, anti_distortion_params)
        else:
            # 如果没有参数，使用默认参数（与anti_distortion.py中的默认参数一致）
            default_params = {
                'p1': 0,
                'p2': 0, 
                'k1': -0.5353,
                'k2': 0.2875,
                'k3': -0.0906,
                'fx_ratio': 1.33,
                'fy_ratio': 1.33
            }
            undistorted_data = process_image_bytes(image_data, default_params)
        
        if undistorted_data is None:
            return jsonify({
                'success': False,
                'message': '图像反畸变处理失败'
            }), 500
          # 将处理后的图像转换为base64
        undistorted_base64 = base64.b64encode(undistorted_data).decode('utf-8')
        
        return jsonify({
            'success': True,
            'message': '图像反畸变处理成功',
            'undistorted_image': f'data:image/png;base64,{undistorted_base64}'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'处理错误: {str(e)}'
        }), 500


@app.route('/api/determine_grid_cell', methods=['POST'])
def determine_grid_cell():
    """
    网格单元判定API端点
    
    接收:
    - camera_idx: 相机索引 (0-3)
    - x: 点击的x坐标
    - y: 点击的y坐标
    - calibration_data: 标定数据 (可选，如果不提供则使用会话数据)
    
    返回:
    - success: 布尔值
    - message: 操作消息
    - grid_cell: [i, j] 网格单元索引（如果成功）
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': '没有接收到请求数据'
            }), 400
        
        # 提取参数
        camera_idx = data.get('camera_idx')
        x = data.get('x')
        y = data.get('y')
        calibration_data = data.get('calibration_data')
        
        # 验证必要参数
        if camera_idx is None or x is None or y is None:
            return jsonify({
                'success': False,
                'message': '缺少必要参数: camera_idx, x, y'
            }), 400
        
        # 使用提供的标定数据或会话数据
        if calibration_data is None:
            calibration_data = session_data.get('calibration_data')
        
        if calibration_data is None:
            return jsonify({
                'success': False,
                'message': '没有可用的标定数据'
            }), 400
        
        # 调用网格判定
        result = determine_grid_cell_from_data(camera_idx, x, y, calibration_data)
        
        if result is not None:
            return jsonify({
                'success': True,
                'message': f'点 ({x}, {y}) 位于网格单元 {result}',
                'grid_cell': result,
                'camera_idx': camera_idx,
                'coordinates': {'x': x, 'y': y}
            })
        else:
            return jsonify({
                'success': False,
                'message': f'点 ({x}, {y}) 不在任何网格单元内',
                'camera_idx': camera_idx,
                'coordinates': {'x': x, 'y': y}
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'处理错误: {str(e)}'
        }), 500


@app.route('/api/save_calibration', methods=['POST'])
def save_calibration():
    """
    保存标定数据API端点
    
    接收:
    - calibration_data: 标定数据
    - filename: 保存文件名 (可选)
    
    返回:
    - success: 布尔值
    - message: 操作消息
    - filename: 保存的文件名
    """
    try:
        data = request.get_json()
        
        if not data or 'calibration_data' not in data:
            return jsonify({
                'success': False,
                'message': '没有接收到标定数据'
            }), 400
        
        calibration_data = data['calibration_data']
        filename = data.get('filename', CALIBRATION_FILE)
        
        # 保存到文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, ensure_ascii=False, indent=2)
        
        # 更新会话数据
        session_data['calibration_data'] = calibration_data
        
        return jsonify({
            'success': True,
            'message': f'标定数据已保存到 {filename}',
            'filename': filename
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'保存失败: {str(e)}'
        }), 500


@app.route('/api/load_calibration', methods=['GET', 'POST'])
def load_calibration():
    """
    加载标定数据API端点
    
    GET: 从默认文件加载
    POST: 从上传的文件加载
    
    返回:
    - success: 布尔值
    - message: 操作消息
    - calibration_data: 标定数据（如果成功）
    """
    try:
        if request.method == 'GET':
            # 从默认文件加载
            if os.path.exists(CALIBRATION_FILE):
                with open(CALIBRATION_FILE, 'r', encoding='utf-8') as f:
                    calibration_data = json.load(f)
                
                session_data['calibration_data'] = calibration_data
                
                return jsonify({
                    'success': True,
                    'message': f'从 {CALIBRATION_FILE} 加载标定数据成功',
                    'calibration_data': calibration_data
                })
            else:
                return jsonify({
                    'success': False,
                    'message': f'默认标定文件 {CALIBRATION_FILE} 不存在'
                }), 404
        
        elif request.method == 'POST':
            # 从上传文件加载
            if 'file' in request.files:
                file = request.files['file']
                if file.filename != '':
                    content = file.read().decode('utf-8')
                    calibration_data = json.loads(content)
                    
                    session_data['calibration_data'] = calibration_data
                    
                    return jsonify({
                        'success': True,
                        'message': f'从上传文件 {file.filename} 加载标定数据成功',
                        'calibration_data': calibration_data
                    })
            
            return jsonify({
                'success': False,
                'message': '没有接收到文件'
            }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'加载失败: {str(e)}'
        }), 500


@app.route('/api/get_grid_info/<int:camera_idx>')
def get_grid_info(camera_idx):
    """
    获取网格信息API端点
    
    参数:
    - camera_idx: 相机索引
    
    返回:
    - success: 布尔值
    - message: 操作消息
    - grid_info: 网格信息（如果成功）
    """
    try:
        calibration_data = session_data.get('calibration_data')
        
        if calibration_data is None:
            return jsonify({
                'success': False,
                'message': '没有可用的标定数据'
            }), 400
        
        grid_determiner = GridDeterminer()
        grid_determiner.set_calibration_data(calibration_data)
        
        grid_info = grid_determiner.get_grid_info(camera_idx)
        
        return jsonify({
            'success': True,
            'message': '获取网格信息成功',
            'grid_info': grid_info
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取网格信息失败: {str(e)}'
        }), 500


@app.route('/api/session_info')
def session_info():
    """
    获取会话信息API端点
    
    返回:
    - has_calibration_data: 是否有标定数据
    - has_anti_distortion_params: 是否有反畸变参数
    - current_camera: 当前相机索引
    - images_count: 上传的图像数量
    """
    return jsonify({
        'has_calibration_data': session_data['calibration_data'] is not None,
        'has_anti_distortion_params': session_data['anti_distortion_params'] is not None,
        'current_camera': session_data['current_camera'],
        'images_count': len(session_data['images']),
        'calibration_file_exists': os.path.exists(CALIBRATION_FILE)
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'message': '接口不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'message': '服务器内部错误'}), 500


if __name__ == '__main__':
    import webbrowser
    import threading
    import time
    
    def open_browser():
        """延迟打开浏览器"""
        time.sleep(1.5)  # 等待服务器启动
        url = "http://localhost:5000"
        print(f"🌐 正在打开浏览器: {url}")
        webbrowser.open(url)
    
    print("=" * 60)
    print("🏊 启动游泳池车道校准Web服务...")
    print("服务地址: http://localhost:5000")
    print("API端点:")
    print("  POST /api/undistort_image - 图像反畸变")
    print("  POST /api/determine_grid_cell - 网格单元判定")
    print("  POST /api/save_calibration - 保存标定数据")
    print("  GET/POST /api/load_calibration - 加载标定数据")
    print("  GET /api/get_grid_info/<camera_idx> - 获取网格信息")
    print("  GET /api/session_info - 获取会话信息")
    print("=" * 60)
    
    try:
        # 在新线程中延迟打开浏览器
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # 启动Flask服务器
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except Exception as e:
        print(f"❌ 启动服务器时发生错误: {e}")
        import traceback
        traceback.print_exc()

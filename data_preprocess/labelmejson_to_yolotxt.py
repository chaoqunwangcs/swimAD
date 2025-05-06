import json
import os
import logging

'''
intput：
G:\project\20250423游泳池数据最新标签修改返回\  <-- 这是你的输入基础目录 (json_base_dir)
│
│   (脚本不会处理直接放在这里的 .json 文件)
│
├───中午\                      <-- 这是基础目录下的 *第一层* 子目录 (Depth 1)
│   │
│   │   (脚本不会处理直接放在这里的 .json 文件)
│   │
│   └───1\                     <-- 这是 "中午" 目录下的子目录，也就是基础目录下的 *第二层* 子目录 (Depth 2)
│       │
│       │   frame_0001.json    <-- **你的 .json 文件必须放在这里**
│       │   ...
│

label第一位为跟踪id
在目标文件夹中，将label分类为正常和异常。
label中第二位或者第三位置信度为0的，视作异常label。
label第四位作为category id

'''
# --- 配置日志 ---
# 设置日志记录级别和格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 配置路径 ---
# 输入 JSON 文件的根目录 (脚本将查找其下两层的 .json 文件)
json_base_dir = r"G:\project\20250423游泳池数据最新标签修改返回" # 包含两层子目录的根路径
# 输出目录
normal_txt_dir = r"G:\project\20250423游泳池数据最新标签修改返回__processed\yolo_label\label_test"   
abnormal_txt_dir = r"G:\project\20250423游泳池数据最新标签修改返回__processed\yolo_label\label_test" # 存放异常标签的目录(置信度l2=0 or l3=0)

# --- 标签解析函数 ---
def _parse_custom_label(label_str):
    """
    解析 LabelMe 自定义标签字符串 "l1,l2,l3,l4" 或 "l1，l2，l3，l4"。
    处理半角 (,) 和全角 (，) 逗号。
    如果有效，则返回整数元组 (label_1, label_2, label_3, label_4)。
    如果解析失败或格式不正确（例如，不是 4 个部分），则返回 None。
    *** 不验证各部分是否为单个数字。***
    """
    if not label_str:
        logging.warning("遇到空标签字符串。")
        return None

    # 将全角逗号替换为半角逗号
    normalized_label_str = label_str.replace('，', ',')

    parts = normalized_label_str.split(',')
    if len(parts) != 4:
        # 记录原始字符串以便更好地调试
        logging.warning(f"标签字符串 '{label_str}' (normalized: '{normalized_label_str}') 没有 4 个部分。跳过。")
        return None
    try:
        # 将部分转换为整数
        labels = tuple(int(p.strip()) for p in parts)
        return labels # 如果所有 4 个部分都成功转换为 int，则返回元组
    except ValueError:
        # 记录原始字符串以便更好地调试
        logging.warning(f"无法将标签 '{label_str}' (normalized: '{normalized_label_str}') 的部分转换为整数。跳过。")
        return None

# --- 主转换函数 ---
def convert_json_to_yolo_v3(json_base_dir, normal_txt_dir, abnormal_txt_dir):
    """
    将嵌套目录中的 JSON 标注文件转换为 YOLO 格式的 TXT 文件，
    并根据解析后的标签第二位或第三位是否为零分离正常和异常标签。

    Args:
        json_base_dir (str): 包含 JSON 文件子目录的根目录路径。
        normal_txt_dir (str): 保存正常 YOLO TXT 文件的目录路径。
        abnormal_txt_dir (str): 保存异常 YOLO TXT 文件的目录路径。
    """
    # 确保输出目录存在
    try:
        os.makedirs(normal_txt_dir, exist_ok=True)
        os.makedirs(abnormal_txt_dir, exist_ok=True)
        logging.info(f"确保输出目录存在: {normal_txt_dir}, {abnormal_txt_dir}")
    except OSError as e:
        logging.error(f"创建输出目录时出错: {e}")
        return

    processed_files_count = 0 # 成功处理（至少生成一个输出文件）的 JSON 文件计数
    error_files_count = 0     # 处理出错或完全没有有效标注的 JSON 文件计数
    total_json_files_found = 0 # 找到的总 JSON 文件数

    # 遍历根目录及其子目录
    logging.info(f"开始在 {json_base_dir} 中搜索 JSON 文件...")
    for root, dirs, files in os.walk(json_base_dir):
        # 计算当前目录相对于基准目录的深度
        # depth = root[len(json_base_dir):].count(os.sep)
        # 根据你的描述 "还有两层子文件"，这里我们不过滤深度，直接处理找到的所有 json
        # if depth > 2: # 如果需要严格限制深度，可以取消注释此检查
        #     continue

        for filename in files:
            if filename.lower().endswith('.json'):
                total_json_files_found += 1
                json_filepath = os.path.join(root, filename)
                # 使用原始 JSON 文件名（不含扩展名）作为 TXT 文件名
                base_filename = os.path.splitext(filename)[0]
                normal_txt_filepath = os.path.join(normal_txt_dir, base_filename + '.txt')
                abnormal_txt_filepath = os.path.join(abnormal_txt_dir, base_filename + '.txt')

                logging.debug(f"正在处理: {json_filepath}") # 使用 debug 级别记录每个文件

                has_valid_annotation = False # 标记当前 JSON 是否包含任何有效标注
                normal_annotations = []   # 存储正常标注
                abnormal_annotations = [] # 存储异常标注

                try:
                    # 打开并读取 JSON 文件
                    with open(json_filepath, 'r', encoding='utf-8') as f_json:
                        data = json.load(f_json)

                    # 提取图像尺寸
                    img_height = data.get('imageHeight')
                    img_width = data.get('imageWidth')
                    shapes = data.get('shapes', [])

                    # 验证图像尺寸
                    if img_height is None or img_width is None or not isinstance(img_height, (int, float)) or not isinstance(img_width, (int, float)) or img_height <= 0 or img_width <= 0:
                        logging.warning(f"跳过 {filename}: 无效或缺失的 'imageHeight' 或 'imageWidth'。")
                        error_files_count += 1
                        continue # 处理下一个 JSON 文件

                    # 处理每个标注形状
                    for shape in shapes:
                        label_str = shape.get('label')
                        points = shape.get('points')
                        shape_type = shape.get('shape_type')

                        # 验证基本形状信息
                        if shape_type != 'rectangle' or label_str is None or points is None or len(points) != 2:
                            logging.debug(f"在 {filename} 中跳过无效或非矩形标注: {shape}")
                            continue

                        # 解析四部分标签
                        parsed_labels = _parse_custom_label(label_str)
                        if parsed_labels is None:
                            # 解析失败的日志已在 _parse_custom_label 中记录
                            continue # 跳过此标注

                        # 使用解析后的第一个标签作为类别 ID
                        class_id = parsed_labels[3]

                        # 解析坐标点
                        try:
                            x1, y1 = map(float, points[0])
                            x2, y2 = map(float, points[1])
                        except (ValueError, TypeError):
                            logging.warning(f"在 {filename} 中跳过包含无效坐标点的标注: {points}")
                            continue

                        # 计算 YOLO 格式值
                        box_width = abs(x1 - x2)
                        box_height = abs(y1 - y2)
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        norm_center_x = round(center_x / img_width, 6)
                        norm_center_y = round(center_y / img_height, 6)
                        norm_width = round(box_width / img_width, 6)
                        norm_height = round(box_height / img_height, 6)

                        # 检查归一化值范围
                        if not (0 <= norm_center_x <= 1 and 0 <= norm_center_y <= 1 and 0 <= norm_width <= 1 and 0 <= norm_height <= 1):
                            logging.warning(f"在 {filename} 中跳过归一化后值越界的标注: "
                                            f"[{norm_center_x}, {norm_center_y}, {norm_width}, {norm_height}]")
                            continue

                        # 格式化 YOLO 字符串
                        yolo_line = f"{class_id} {norm_center_x} {norm_center_y} {norm_width} {norm_height}"
                        has_valid_annotation = True # 标记已找到至少一个有效标注

                        # --- 修改点: 判断是正常还是异常标注 ---
                        # 检查解析后的标签的第二位 (索引1) 或第三位 (索引2) 是否为 0
                        if parsed_labels[1] == 0 or parsed_labels[2] == 0:
                            abnormal_annotations.append(yolo_line)
                            logging.debug(f"  -> 添加到异常 (基于标签 l2 或 l3 为 0): {yolo_line}")
                        else:
                            normal_annotations.append(yolo_line)
                            logging.debug(f"  -> 添加到正常: {yolo_line}")

                    # --- 文件写入 ---
                    file_written = False # 标记是否为当前 JSON 写入了任何文件
                    # 写入正常标注文件
                    if normal_annotations:
                        try:
                            with open(normal_txt_filepath, 'w', encoding='utf-8') as f_txt:
                                f_txt.write("\n".join(normal_annotations) + "\n")
                            logging.debug(f"已写入正常标注到: {normal_txt_filepath}")
                            file_written = True
                        except IOError as e:
                             logging.error(f"写入正常标注文件 {normal_txt_filepath} 时出错: {e}")

                    # 写入异常标注文件
                    if abnormal_annotations:
                        try:
                            with open(abnormal_txt_filepath, 'w', encoding='utf-8') as f_txt:
                                f_txt.write("\n".join(abnormal_annotations) + "\n")
                            logging.debug(f"已写入异常标注到: {abnormal_txt_filepath}")
                            file_written = True
                        except IOError as e:
                            logging.error(f"写入异常标注文件 {abnormal_txt_filepath} 时出错: {e}")

                    # 更新计数器
                    if file_written:
                        processed_files_count += 1
                    elif has_valid_annotation:
                        # 有有效标注但写入失败
                        error_files_count += 1
                        logging.error(f"处理 {filename} 时找到有效标注但未能成功写入任何文件。")
                    else:
                        # 没有找到任何有效标注
                        logging.info(f"在 {filename} 中未找到有效标注，未创建 TXT 文件。")
                        # 将其计为“错误/跳过”，因为它没有产生输出
                        error_files_count += 1


                # --- 异常处理 ---
                except json.JSONDecodeError:
                    logging.error(f"解码 {filename} 的 JSON 时出错。跳过。")
                    error_files_count += 1
                except FileNotFoundError: # 虽然 os.walk 找到的文件理论上存在，但以防万一
                    logging.error(f"尝试打开文件时未找到 {json_filepath}。跳过。")
                    error_files_count += 1
                except Exception as e:
                    logging.error(f"处理 {filename} 时发生意外错误: {e}", exc_info=True) # 添加 exc_info 获取 traceback
                    error_files_count += 1

    # --- 输出转换摘要 ---
    logging.info(f"--- 转换摘要 ---")
    logging.info(f"在 {json_base_dir} 及其子目录中共找到 {total_json_files_found} 个 JSON 文件。")
    logging.info(f"成功处理（至少生成一个输出文件）的 JSON 文件数: {processed_files_count}")
    logging.info(f"处理出错或无有效标注的 JSON 文件数: {error_files_count}")
    logging.info(f"正常 TXT 文件保存在: {normal_txt_dir}")
    logging.info(f"异常 TXT 文件保存在: {abnormal_txt_dir}")


# --- 脚本执行入口 ---
if __name__ == "__main__":
    # 使用脚本开头定义的示例路径运行转换
    convert_json_to_yolo_v3(json_base_dir, normal_txt_dir, abnormal_txt_dir)
    # 提示：你可以将 json_base_dir, normal_txt_dir, abnormal_txt_dir
    #      替换为你实际使用的路径。

import os
import json
from PIL import Image # 用于读取图片尺寸
# import argparse # 如果不使用命令行参数，可以注释掉
from tqdm import tqdm # 可选，用于显示进度条
import glob         # 用于查找标签文件

# --- 配置常量 (宏定义) ---

IMG_DIRECTORY = r"G:\project\20250423游泳池数据最新标签修改返回__processed\yolo_label\yolo_data\images\val"  # <--- 修改: 包含图片的目录路径
LABEL_DIRECTORY = r"G:\project\detection_model\result_pred_v11L_250_with_score\labels" # <--- 包含 YOLO .txt 标签文件的目录路径
OUTPUT_JSON = r"G:\project\detection_model\result_pred_v11L_250_with_score\cocolabel\pred.json" # <--- 输出 COCO JSON 文件的保存路径
# --- CLASS_NAMES_PATH 不再需要 ---
# --------------------------

def discover_yolo_classes(label_dir):
    """
    扫描标签目录中的所有 .txt 文件，发现所有唯一的 YOLO 类别 ID。

    Args:
        label_dir (str): 包含 YOLO .txt 标签文件的目录路径。

    Returns:
        list: 包含所有唯一、已排序的 YOLO 类别 ID 的列表。 None 如果出错。
    """
    discovered_yolo_ids = set()
    print(f"\n开始扫描标签目录以发现类别 ID: {label_dir}")
    try:
        # 使用 glob 查找所有 .txt 文件，更健壮
        label_files = glob.glob(os.path.join(label_dir, '*.txt'))
        if not label_files:
            print(f"警告：在目录 {label_dir} 中找不到任何 .txt 标签文件。")
            return [] # 返回空列表

        for label_path in tqdm(label_files, desc="扫描标签文件"):
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 1: # 至少需要有类别 ID
                        try:
                            yolo_class_id = int(parts[0])
                            discovered_yolo_ids.add(yolo_class_id)
                        except ValueError:
                             # 忽略无法解析为整数的第一部分
                             pass
            except Exception as e:
                 print(f"\n警告：读取标签文件 {label_path} 时出错: {e}。跳过此文件。")

        sorted_yolo_ids = sorted(list(discovered_yolo_ids))
        print(f"扫描完成。发现的唯一 YOLO 类别 ID: {sorted_yolo_ids}")
        return sorted_yolo_ids

    except Exception as e:
        print(f"错误：扫描标签目录 {label_dir} 时发生严重错误: {e}")
        return None


def yolo_to_coco(img_dir, label_dir, output_json_path):
    """
    将 YOLO 格式的标注转换为 COCO JSON 格式，自动检测类别。

    Args:
        img_dir (str): 包含图片的目录路径。
        label_dir (str): 包含 YOLO .txt 标签文件的目录路径。
        output_json_path (str): 输出 COCO JSON 文件的保存路径。
    """
    # --- 1. 初始化 COCO 数据结构 ---
    coco_output = {
        "info": {
            "description": "从 YOLO 转换的数据集 (自动检测类别)",
            "url": "",
            "version": "1.0",
            "year": 2025, # 或当前年份
            "contributor": "YOLO to COCO Script",
            "date_created": "" # 或当前日期
        },
        "licenses": [], # 如果适用，添加许可证信息
        "images": [],
        "annotations": [],
        "categories": []
    }

    # --- 2. 自动发现类别并填充 categories 字段 ---
    sorted_yolo_ids = discover_yolo_classes(label_dir)

    if sorted_yolo_ids is None: # 如果 discover_yolo_classes 过程中出错
        print("错误：无法发现类别 ID，转换中止。")
        return
    if not sorted_yolo_ids:
         print("警告：未在标签文件中发现任何有效的类别 ID。生成的 COCO 文件可能不完整或无效。")
         # 可以选择在这里停止，或者继续生成一个没有类别和标注的 COCO 文件

    # 创建从 YOLO 类别 ID (发现的整数) 到 COCO 类别 ID (1-based) 的映射
    yolo_class_id_to_coco_id = {}
    print("\n正在生成 COCO 类别信息...")
    for i, yolo_id in enumerate(sorted_yolo_ids):
        # COCO 类别 ID 从 1 开始
        coco_category_id = i + 1
        coco_output["categories"].append({
            "id": coco_category_id,
            "name": f"class_{yolo_id}", # 使用通用名称 "class_YOLOID"
            "supercategory": "object" # 可以根据需要修改或留空
        })
        yolo_class_id_to_coco_id[yolo_id] = coco_category_id

    if coco_output["categories"]:
        print("COCO 类别信息已生成。")
        print("类别 ID 映射 (YOLO ID -> COCO ID):", yolo_class_id_to_coco_id)
    else:
        print("未生成任何 COCO 类别信息。")


    # --- 3. 处理图片和标注 ---
    image_id_counter = 1      # COCO 图片 ID 从 1 开始
    annotation_id_counter = 1 # COCO 标注 ID 从 1 开始
    valid_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'} # 支持的图片格式

    print(f"\n正在扫描图片目录: {img_dir}")
    try:
        image_files = [f for f in os.listdir(img_dir)
                       if os.path.splitext(f)[1].lower() in valid_image_extensions]
    except FileNotFoundError:
        print(f"错误：找不到图片目录 {img_dir}")
        return
    except Exception as e:
        print(f"错误：访问图片目录时出错: {e}")
        return

    print(f"找到 {len(image_files)} 个潜在的图片文件。")

    if not image_files:
        print("错误：在图片目录中没有找到有效的图片文件。")
        return

    # 使用 tqdm 显示进度条
    for image_filename in tqdm(image_files, desc="处理图片和标签"):
        image_path = os.path.join(img_dir, image_filename)
        # 构造对应的标签文件名 (与图片同名，扩展名为 .txt)
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)

        # --- 获取图片尺寸 ---
        try:
            # 使用 Pillow 打开图片获取宽高
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except FileNotFoundError:
            # 如果图片文件不存在，打印警告并跳过
            print(f"\n警告：找不到图片文件 {image_path}。跳过此图片。")
            continue
        except Exception as e:
            # 如果读取图片时发生其他错误，打印警告并跳过
            print(f"\n警告：无法读取图片 {image_path}。错误: {e}。跳过此图片。")
            continue

        # --- 将图片信息添加到 COCO 输出 ---
        coco_output["images"].append({
            "id": image_id_counter,
            "width": img_width,
            "height": img_height,
            "file_name": image_filename, # 保留原始文件名
            "license": 0, # 如果适用，添加许可证 ID
            "flickr_url": "", # 可选
            "coco_url": "",   # 可选
            "date_captured": "" # 可选
        })

        # --- 处理对应的标签文件 ---
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    # 支持带置信度的6字段行，只取前5个字段用于bbox等，score单独提取
                    score = None
                    if len(parts) == 6:
                        score = float(parts[5])
                        parts = parts[:5]
                    # 确保每行有 5 个部分: class_id, x_center, y_center, width, height
                    if len(parts) == 5:
                        try:
                            # 解析 YOLO 格式数据
                            yolo_class_id = int(parts[0]) # 这是实际在文件中找到的 ID
                            x_center_norm = float(parts[1])
                            y_center_norm = float(parts[2])
                            width_norm = float(parts[3])
                            height_norm = float(parts[4])

                            # --- 将 YOLO 坐标转换为 COCO 边界框格式 [x_min, y_min, width, height] ---
                            bbox_width = width_norm * img_width
                            bbox_height = height_norm * img_height
                            x_min = (x_center_norm * img_width) - (bbox_width / 2)
                            y_min = (y_center_norm * img_height) - (bbox_height / 2)
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            bbox_width = min(img_width - x_min, bbox_width)
                            bbox_height = min(img_height - y_min, bbox_height)

                            # --- 获取 COCO 类别 ID (使用动态生成的映射) ---
                            if yolo_class_id not in yolo_class_id_to_coco_id:
                                # 这个情况理论上不应发生，因为我们是基于发现的 ID 生成映射的
                                # 但以防万一添加检查
                                print(f"\n警告：在标签文件 {label_filename} 中发现未被记录的 YOLO 类别 ID {yolo_class_id}。跳过此标注。")
                                continue # 跳过这个无效的标注
                            coco_cat_id = yolo_class_id_to_coco_id[yolo_class_id]

                            # --- 添加标注信息到 COCO 输出 ---
                            ann = {
                                "id": annotation_id_counter,       # 唯一的标注 ID
                                "image_id": image_id_counter,      # 对应的图片 ID
                                "category_id": coco_cat_id,        # 对应的类别 ID
                                "segmentation": [],                # YOLO 通常不提供分割信息，留空
                                "area": bbox_width * bbox_height,  # 边界框面积
                                "bbox": [round(x_min, 2), round(y_min, 2), round(bbox_width, 2), round(bbox_height, 2)], # COCO 格式的边界框
                                "iscrowd": 0                       # 0 表示非拥挤目标
                            }
                            if score is not None:
                                ann["score"] = score
                            coco_output["annotations"].append(ann)
                            annotation_id_counter += 1 # 增加标注 ID 计数器
                        except ValueError:
                             print(f"\n警告：标签文件 {label_filename} 中行 '{line.strip()}' 包含无效数字格式。跳过此行。")
                        except IndexError:
                             print(f"\n警告：标签文件 {label_filename} 中行 '{line.strip()}' 格式不正确。跳过此行。")
                    elif line.strip(): # 如果行不为空但部分数量不对
                        print(f"\n警告：标签文件 {label_filename} 中行 '{line.strip()}' 格式无效 (预期 5 个部分)。跳过此行。")
            except Exception as e:
                # 如果处理标签文件时出错，打印警告
                print(f"\n警告：处理标签文件 {label_path} 时出错。错误: {e}")

        # 为下一张图片增加图片 ID 计数器
        image_id_counter += 1

    # --- 4. 保存 COCO JSON 文件 ---
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_json_path)
        if output_dir: # 检查目录名是否为空 (如果输出就在当前目录)
             os.makedirs(output_dir, exist_ok=True)

        print(f"\n正在将结果写入 JSON 文件: {output_json_path}")
        # 将 COCO 数据结构写入 JSON 文件，使用 indent 使其更易读
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_output, f, indent=4, ensure_ascii=False) # ensure_ascii=False 支持中文等字符

        print(f"\n转换成功完成！")
        print(f"总共处理了 {image_id_counter - 1} 张图片。")
        print(f"总共创建了 {annotation_id_counter - 1} 条标注。")
        print(f"COCO JSON 文件已保存到: {output_json_path}")
    except Exception as e:
        print(f"\n错误：保存 COCO JSON 文件时出错: {e}")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 检查定义的路径是否存在（可选但推荐）
    if not os.path.isdir(IMG_DIRECTORY):
        print(f"错误：图片目录不存在: {IMG_DIRECTORY}")
    elif not os.path.isdir(LABEL_DIRECTORY):
        print(f"错误：标签目录不存在: {LABEL_DIRECTORY}")
    # 不再检查 CLASS_NAMES_PATH
    else:
        # 确保输出目录存在
        output_dir_main = os.path.dirname(OUTPUT_JSON)
        if output_dir_main:
            os.makedirs(output_dir_main, exist_ok=True)

        # 调用转换函数，不再需要类别文件路径
        yolo_to_coco(IMG_DIRECTORY, LABEL_DIRECTORY, OUTPUT_JSON)

    # --- 如果希望使用命令行参数（不再需要 --classes 参数） ---
    # import argparse
    # parser = argparse.ArgumentParser(description="将 YOLO 格式的标注转换为 COCO JSON 格式 (自动检测类别)。")
    # parser.add_argument("--img_dir", default=IMG_DIRECTORY, help=f"包含图片的目录路径 (默认: {IMG_DIRECTORY})。")
    # parser.add_argument("--label_dir", default=LABEL_DIRECTORY, help=f"包含 YOLO .txt 标签文件的目录路径 (默认: {LABEL_DIRECTORY})。")
    # parser.add_argument("--output", default=OUTPUT_JSON, help=f"输出 COCO JSON 文件的路径 (默认: {OUTPUT_JSON})。")
    # args = parser.parse_args()
    #
    # # 确保输出目录存在
    # output_dir_arg = os.path.dirname(args.output)
    # if output_dir_arg:
    #     os.makedirs(output_dir_arg, exist_ok=True)
    #
    # yolo_to_coco(args.img_dir, args.label_dir, args.output)


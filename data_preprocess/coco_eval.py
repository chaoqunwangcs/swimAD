# -*- coding: utf-8 -*-
"""
COCO目标检测评估脚本 - 清理优化版本 (支持argparse)
功能：
1. COCO格式的目标检测评估
2. 生成P-R曲线图（按置信度和面积分类）
3. 生成混淆矩阵
4. 支持指定类别评估
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import traceback
import argparse # 导入 argparse

# COCO面积阈值定义 (保持为全局常量)
AREA_THRESHOLDS = {
    "all": (0, float('inf')),
    "small": (0, 32**2),
    "medium": (32**2, 96**2),
    "large": (96**2, float('inf'))
}

def parse_arguments():
  """
  使用 argparse 解析命令行参数
  """
  parser = argparse.ArgumentParser(description="COCO Object Detection Evaluation Script")
  parser.add_argument("-g", "--gt_file", type=str, required=True, 
                      help="Path to the ground truth COCO JSON file (e.g., instances_val_gt.json)")
  parser.add_argument("-p", "--pred_file", type=str, required=True,
                      help="Path to the prediction COCO JSON file (e.g., pred.json)")
  parser.add_argument("-o", "--results_dir", type=str, default="./results",
                      help="Directory to save evaluation results and plots (default: ./results)")
  parser.add_argument("--iou_type", type=str, default='bbox', choices=['bbox', 'segm'],
                      help="Evaluation type: 'bbox' or 'segm' (default: bbox)")
  parser.add_argument("--category_ids", type=int, nargs='+', default=None, # 默认评估所有类别
                      help="List of category IDs to evaluate (e.g., 1 2 3). If not provided, evaluates all categories.")
  # parser.add_argument("--score_thresholds_main", type=float, nargs='+', default=[0.1],
  #                     help="List of score thresholds to iterate over in the main evaluation loop (default: [0.1])")

  args = parser.parse_args()
  return args

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    
    Args:
        box1, box2: [x, y, w, h] 格式的边界框
        
    Returns:
        float: IoU值
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 转换为 (x1, y1, x2, y2) 格式
    box1_coords = [x1, y1, x1 + w1, y1 + h1]
    box2_coords = [x2, y2, x2 + w2, y2 + h2]
    
    # 计算交集
    xi1 = max(box1_coords[0], box2_coords[0])
    yi1 = max(box1_coords[1], box2_coords[1])
    xi2 = min(box1_coords[2], box2_coords[2])
    yi2 = min(box1_coords[3], box2_coords[3])
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def get_bbox_area(bbox):
    """计算边界框面积"""
    return bbox[2] * bbox[3]

def filter_by_area(boxes, area_filter):
    """根据面积过滤边界框"""
    min_area, max_area = AREA_THRESHOLDS[area_filter]
    filtered = []
    for box in boxes:
        area = get_bbox_area(box['bbox'])
        if min_area <= area < max_area:
            filtered.append(box)
    return filtered

def calculate_precision_recall_for_category(gt_data, detections_list, category_id, area_filter, iou_threshold=0.5):
    """
    计算特定类别和面积的精度和召回率
    
    Returns:
        tuple: (precision, recall)
    """
    # 按图像组织GT和预测
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)
    
    for ann in gt_data['annotations']:
        if ann['category_id'] == category_id:
            gt_by_image[ann['image_id']].append(ann)
    
    for det in detections_list:
        if det['category_id'] == category_id:
            pred_by_image[det['image_id']].append(det)
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # 获取所有图像ID
    all_image_ids = set(gt_by_image.keys()) | set(pred_by_image.keys())
    
    for img_id in all_image_ids:
        gt_boxes = filter_by_area(gt_by_image.get(img_id, []), area_filter)
        pred_boxes = filter_by_area(pred_by_image.get(img_id, []), area_filter)
        
        matched_gt = set()
        
        # 匹配预测框到GT框
        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1
        
        # 未匹配的GT为false negatives
        false_negatives += len(gt_boxes) - len(matched_gt)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    return precision, recall

def plot_precision_recall_curves(gt_data, detections_list, save_dir, conf_thresholds=None, area_filter="all"):
    """绘制精度和召回率曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    if conf_thresholds is None:
        conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # 获取类别信息
    categories = {cat['id']: cat['name'] for cat in gt_data['categories']}
    cat_ids = sorted(categories.keys())
    cat_names = [categories[cat_id] for cat_id in cat_ids]
    
    min_area, max_area = AREA_THRESHOLDS[area_filter]
    
    def filter_by_area_and_conf(detections, conf_thresh):
        filtered = []
        for det in detections:
            if det.get('score', 1.0) >= conf_thresh:
                area = get_bbox_area(det['bbox'])
                if min_area <= area < max_area:
                    filtered.append(det)
        return filtered
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 绘制精度曲线
    for cat_id, cat_name in zip(cat_ids, cat_names):
        precisions = []
        recalls = []
        
        for conf_thresh in conf_thresholds:
            filtered_detections = filter_by_area_and_conf(detections_list, conf_thresh)
            precision, recall = calculate_precision_recall_for_category(gt_data, filtered_detections, cat_id, area_filter)
            precisions.append(precision)
            recalls.append(recall)
        
        ax1.plot(conf_thresholds, precisions, label=f'{cat_name}', marker='o', linewidth=2, markersize=6)
        ax2.plot(conf_thresholds, recalls, label=f'{cat_name}', marker='s', linewidth=2, markersize=6)
    
    # 设置精度图
    ax1.set_xlabel('Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title(f'Precision vs Confidence (Area: {area_filter})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([min(conf_thresholds), max(conf_thresholds)])
    ax1.set_ylim([0, 1])
    
    # 设置召回率图
    ax2.set_xlabel('Confidence Threshold', fontsize=12)
    ax2.set_ylabel('Recall', fontsize=12)
    ax2.set_title(f'Recall vs Confidence (Area: {area_filter})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([min(conf_thresholds), max(conf_thresholds)])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"precision_recall_curves_area_{area_filter}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"精度召回率曲线已保存到 {os.path.join(save_dir, f'precision_recall_curves_area_{area_filter}.png')}")

def create_confusion_matrix(gt_data, detections_list, save_dir, iou_threshold=0.5, area_filter="all"):
    """创建并保存混淆矩阵"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取类别信息
    categories = {cat['id']: cat['name'] for cat in gt_data['categories']}
    cat_ids = sorted(categories.keys())
    cat_names = [categories[cat_id] for cat_id in cat_ids]
    
    # 添加背景类
    background_id = (max(cat_ids) + 1) if cat_ids else 0 # Handle case with no categories
    cat_names_with_bg = cat_names + ['Background']
    
    # 按图像组织GT和预测
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)
    
    for ann in gt_data['annotations']:
        gt_by_image[ann['image_id']].append(ann)
    
    for det in detections_list:
        pred_by_image[det['image_id']].append(det)
    
    y_true = []
    y_pred = []
    
    # 获取所有图像ID
    all_image_ids = set(gt_by_image.keys()) | set(pred_by_image.keys())
    
    for img_id in all_image_ids:
        gt_boxes = filter_by_area(gt_by_image.get(img_id, []), area_filter)
        pred_boxes = filter_by_area(pred_by_image.get(img_id, []), area_filter)
        
        if not gt_boxes and not pred_boxes:
            continue
            
        matched_gt = set()
        # matched_pred = set() # Not strictly needed for y_true, y_pred collection
        
        # 匹配预测框到GT框
        for pred_idx, pred in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            best_gt_cat = None
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_gt_cat = gt['category_id']
            
            if best_gt_idx >= 0:
                # 找到匹配的GT框
                matched_gt.add(best_gt_idx)
                # matched_pred.add(pred_idx)
                y_true.append(best_gt_cat)
                y_pred.append(pred['category_id'])
            else:
                # 误检（预测为某类但没有匹配的GT）
                y_true.append(background_id)
                y_pred.append(pred['category_id'])
        
        # 未匹配的GT（漏检）
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                y_true.append(gt['category_id'])
                y_pred.append(background_id)
    
    # 检查是否有数据
    if not y_true or not y_pred:
        print(f"警告: 没有找到有效的匹配数据来生成混淆矩阵 (IoU={iou_threshold}, area={area_filter})")
        return None, cat_names_with_bg
    
    # 创建混淆矩阵
    labels_for_cm = cat_ids + [background_id]
    # Ensure all actual and predicted category IDs are in labels_for_cm
    # This handles cases where a category ID might appear in predictions but not in gt_data['categories']
    # or vice-versa, though ideally gt_data['categories'] is the superset.
    all_present_ids = sorted(list(set(y_true) | set(y_pred)))
    
    # Ensure background_id is the largest for consistent plotting
    if background_id not in all_present_ids and all_present_ids:
         all_present_ids.append(max(all_present_ids) + 1 if background_id < max(all_present_ids) else background_id)
    elif not all_present_ids: # No data
        all_present_ids = [background_id]


    # Map actual category IDs to names for tick labels, including background
    final_tick_labels_map = {cat_id: categories.get(cat_id, f"UnknownCatID_{cat_id}") for cat_id in all_present_ids if cat_id != background_id}
    final_tick_labels_map[background_id] = "Background"
    
    # Sort labels for confusion matrix based on all_present_ids to ensure consistency
    # And get corresponding names for ticks
    sorted_cm_labels = sorted(all_present_ids)
    final_tick_names = [final_tick_labels_map[l] for l in sorted_cm_labels]

    cm = confusion_matrix(y_true, y_pred, labels=sorted_cm_labels)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(max(12, len(final_tick_names)), max(10, len(final_tick_names)-2))) # Adjust size based on num labels
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix @ IoU={iou_threshold} | Area={area_filter}', fontsize=14)
    plt.colorbar(im)
    
    # 添加文本标注
    thresh = cm.max() / 2. if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=10)
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    
    plt.xticks(range(len(final_tick_names)), final_tick_names, rotation=45, ha='right')
    plt.yticks(range(len(final_tick_names)), final_tick_names, rotation=0)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_iou_{iou_threshold:.2f}_area_{area_filter}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"混淆矩阵已保存到 {os.path.join(save_dir, f'confusion_matrix_iou_{iou_threshold:.2f}_area_{area_filter}.png')}")
    
    # 输出统计信息
    print_confusion_matrix_stats(cm, sorted_cm_labels, final_tick_labels_map, iou_threshold, area_filter)
    
    return cm, final_tick_names # Return names corresponding to sorted_cm_labels

def print_confusion_matrix_stats(cm, cm_labels, label_map, iou_threshold, area_filter):
    """打印混淆矩阵统计信息"""
    print(f"\n混淆矩阵统计 (IoU={iou_threshold}, area={area_filter}):")
    print(f"总样本数: {cm.sum()}")
    
    for i, label_id in enumerate(cm_labels):
        label_name = label_map.get(label_id, f"ID_{label_id}")
        if i < cm.shape[0] and i < cm.shape[1]: # Ensure index is within bounds
            true_positives = cm[i, i]
            false_positives = cm[:, i].sum() - true_positives
            false_negatives = cm[i, :].sum() - true_positives
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            print(f"{label_name}: TP={true_positives}, FP={false_positives}, FN={false_negatives}, "
                  f"Precision={precision:.3f}, Recall={recall:.3f}")

def evaluate_coco_with_visualization(gt_file_path, dt_file_path, results_output_dir, 
                                     eval_type='bbox', category_ids_to_eval=None):
    """
    执行COCO评估并生成可视化图表
    
    Args:
        gt_file_path: 真实标注文件路径
        dt_file_path: 检测结果文件路径
        results_output_dir: 结果输出目录
        eval_type: 评估类型 ('bbox' 或 'segm')
        category_ids_to_eval: 要评估的类别ID列表
    """
    try:
        # 验证文件存在
        if not os.path.exists(gt_file_path):
            print(f"错误: 真实标注文件不存在: {gt_file_path}")
            return
        if not os.path.exists(dt_file_path):
            print(f"错误: 检测结果文件不存在: {dt_file_path}")
            return
        
        # 创建结果目录
        os.makedirs(results_output_dir, exist_ok=True)

        # 加载真实标注
        print(f"正在加载真实标注: {gt_file_path}")
        with open(gt_file_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        cocoGt = COCO()
        cocoGt.dataset = gt_data
        cocoGt.createIndex()
        print(f"真实标注加载完成，图像数量: {len(cocoGt.getImgIds())}")
        
        # 加载检测结果
        print(f"正在加载检测结果: {dt_file_path}")
        with open(dt_file_path, 'r', encoding='utf-8') as f:
            detections_data = json.load(f)
        
        # 处理检测结果格式
        if isinstance(detections_data, dict) and 'annotations' in detections_data:
            detections_list = detections_data['annotations']
        elif isinstance(detections_data, list):
            detections_list = detections_data
        else:
            raise ValueError("检测结果格式错误，应为列表或包含 'annotations' 键的字典")
        
        # 添加缺失的score字段 (如果检测结果中没有score，默认为1.0)
        for det in detections_list:
            if 'score' not in det:
                det['score'] = 1.0
        
        print(f"检测结果加载完成，检测数量: {len(detections_list)}")
        
        # 执行COCO评估
        cocoDt = cocoGt.loadRes(detections_list) # 使用完整的检测列表
        cocoEval = COCOeval(cocoGt, cocoDt, iouType=eval_type)
        
        # 设置评估类别
        if category_ids_to_eval:
            all_gt_cat_ids = cocoGt.getCatIds()
            valid_cat_ids = [cat_id for cat_id in category_ids_to_eval if cat_id in all_gt_cat_ids]
            if valid_cat_ids:
                cocoEval.params.catIds = valid_cat_ids
                print(f"将评估限制在类别ID: {valid_cat_ids}")
            else:
                print(f"警告: 指定的类别ID {category_ids_to_eval} 在真实标注中不存在或为空。将评估所有类别。")
        else:
            print("未指定特定类别ID，将评估所有类别。")

        # 设置自定义IoU阈值 (COCOeval默认使用 np.linspace(.5, 0.95, 10))
        # 这里可以按需修改，例如：
        # iou_thresholds_for_eval = np.linspace(0.4, 0.9, 6)  # 0.4到0.9，步长0.1
        # cocoEval.params.iouThrs = iou_thresholds_for_eval
        
        print("正在执行COCO评估...")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize() # 打印标准的COCO评估结果
        
        # 生成可视化图表
        print("\n正在生成可视化图表...")
        
        # 置信度阈值 (用于P-R曲线，与COCOeval内部的阈值不同)
        conf_thresholds_for_pr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # 面积过滤器
        area_filters = ["all", "small", "medium", "large"]
        
        # 生成P-R曲线
        for area_filter in area_filters:
            print(f"生成 {area_filter} 面积的P-R曲线")
            plot_precision_recall_curves(gt_data, detections_list, # 使用完整的检测列表
                                         save_dir=results_output_dir,
                                         conf_thresholds=conf_thresholds_for_pr,
                                         area_filter=area_filter)
        
        # 生成混淆矩阵
        iou_thresholds_for_cm = [0.3, 0.5, 0.7] # 用于混淆矩阵的IoU阈值
        for iou_thresh_cm in iou_thresholds_for_cm:
            for area_filter in area_filters:
                print(f"生成IoU={iou_thresh_cm}, area={area_filter}的混淆矩阵")
                try:
                    create_confusion_matrix(gt_data, detections_list, # 使用完整的检测列表
                                            save_dir=results_output_dir,
                                            iou_threshold=iou_thresh_cm,
                                            area_filter=area_filter)
                except Exception as e:
                    print(f"警告: 无法生成混淆矩阵 (IoU={iou_thresh_cm}, area={area_filter}): {e}")
                    traceback.print_exc()
        
        print(f"\n所有图表已保存到: {results_output_dir}")
        
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        traceback.print_exc()

def main():
    """主函数"""
    args = parse_arguments() # 解析命令行参数

    print("COCO目标检测评估脚本 - 清理优化版本 (支持argparse)")
    print("=" * 60)
    print(f"Ground Truth 文件: {args.gt_file}")
    print(f"Prediction 文件: {args.pred_file}")
    print(f"结果保存目录: {args.results_dir}")
    print(f"IoU 类型: {args.iou_type}")
    if args.category_ids:
        print(f"评估特定类别 IDs: {args.category_ids}")
    else:
        print("评估所有类别")
    print("=" * 60)
    
    # --- 注意：以下循环是原脚本中用于不同置信度阈值过滤检测结果的逻辑 ---
    # --- COCOeval 本身会处理不同置信度的检测，通常不需要在此处预过滤 ---
    # --- 如果确实需要基于特定置信度阈值进行多次独立评估，可以保留此逻辑 ---
    # --- 但对于标准COCO评估和P-R曲线生成，通常传入所有检测（带分数）即可 ---
    
    # 示例：如果只想对原始（未经过外部置信度过滤）的检测文件进行一次评估
    # score_thresholds_loop = [0.0] # 表示不过滤，或使用一个象征性的阈值
    # 如果你的 pred_file 已经是过滤后的，或者你想评估原始 pred_file，就用 0.0
    
    # 原脚本的逻辑是为每个score_thresholds创建一个临时文件并评估
    # 这里我们简化为直接评估传入的pred_file，因为COCOeval和P-R曲线会处理score
    # 如果你的pred_file本身不包含低置信度的检测，那也是可以的。
    # 如果你想复现原脚本的“为每个thresh过滤并评估”的行为，可以取消注释下面的循环
    
    # score_thresholds_to_iterate = args.score_thresholds_main # 从argparse获取
    score_thresholds_to_iterate = [0.0] # 简化：对原始（或已过滤的）pred_file评估一次

    for thresh in score_thresholds_to_iterate:
        current_pred_file = args.pred_file
        temp_pred_file_path = None

        if thresh > 0.0: # 仅当需要基于此处的阈值过滤时才创建临时文件
            print(f"\n==== 注意: 如果需要，将预过滤检测结果 (score > {thresh}) ====")
            print(f"====      (标准COCO评估通常不需要此步骤，它内部处理分数) ====")
            
            with open(args.pred_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            if isinstance(loaded_data, dict) and 'annotations' in loaded_data:
                detections_to_filter = loaded_data['annotations']
            elif isinstance(loaded_data, list):
                detections_to_filter = loaded_data
            else:
                raise ValueError("检测结果格式错误")
            
            filtered_detections = [d for d in detections_to_filter if d.get('score', 1.0) > thresh]
            
            # 确保临时文件目录存在 (例如，在 results_dir 下创建临时目录)
            temp_dir = os.path.join(args.results_dir, "temp_preds")
            os.makedirs(temp_dir, exist_ok=True)
            temp_pred_file_path = os.path.join(temp_dir, f"temp_pred_score_gt_{int(thresh*100)}.json")
            
            with open(temp_pred_file_path, 'w', encoding='utf-8') as f:
                # 如果原始文件是字典格式，临时文件也保存为字典格式
                if isinstance(loaded_data, dict) and 'annotations' in loaded_data:
                    temp_data_to_save = loaded_data.copy() # 复制其他元数据如 'images', 'categories'
                    temp_data_to_save['annotations'] = filtered_detections
                    json.dump(temp_data_to_save, f)
                else: # 否则直接保存列表
                    json.dump(filtered_detections, f)
            current_pred_file = temp_pred_file_path
            print(f"已创建临时过滤检测文件: {current_pred_file} (包含 {len(filtered_detections)} 条检测)")

        try:
            # 执行评估
            evaluate_coco_with_visualization(
                gt_file_path=args.gt_file, 
                dt_file_path=current_pred_file,
                results_output_dir=args.results_dir,
                eval_type=args.iou_type,
                category_ids_to_eval=args.category_ids
            )
        finally:
            # 清理临时文件
            if temp_pred_file_path and os.path.exists(temp_pred_file_path):
                try:
                    os.remove(temp_pred_file_path)
                    print(f"已删除临时文件: {temp_pred_file_path}")
                except OSError as e:
                    print(f"删除临时文件失败 {temp_pred_file_path}: {e}")
    
    print("\n评估完成！")

if __name__ == "__main__":
    main()

## coco_eval.py

**Args:**

1. **COCO评估：** 

2. **P-R曲线生成：** 按照不同的置信度阈值和目标面积（全部、小、中、大物体）绘制精确率-召回率（Precision-Recall）曲线图

3. **混淆矩阵生成：** 为不同的IoU阈值和目标面积生成混淆矩阵。

   

**example**

```
python evaluate_coco.py  --gt_file /path/to/your/ground_truth_annotations.json   --pred_file /path/to/your/model_detection_results.json  --results_dir ./logs
    

```


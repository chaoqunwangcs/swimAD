# 任务：泳池场景溺水检测

## 任务描述
任务输入: 多视角图像（4个视角），2K分辨率，25Hz

任务输出：溺水人员检测结果

## 整体推理流程
1. - [ ] 多视角相机数据读取
2. - [ ] 多视角联合溺水检测
    1. - [x] 单视角多目标跟踪
    2. - [x] 多视角联合跟踪
    3. - [ ] 溺水判定
3. - [ ] 输出处理

### 多视角相机数据读取（李航）
1. - [ ] 需要更新

### 单视角多目标跟踪（李健）
1. - [x] [数据标注](data_annotation.md)

2. 跟踪模型训练
    1. - [x] 环境搭建

        ```
        conda create -n swimAD python=3.9 -y
        conda activate swimAD
        pip install -r requirements.txt
        cd boxmot
        pip install .
        ```
       
    2. - [x] [数据预处理](data_preprocess/data_preprocess.md)
       
    3. - [x] 检测模型训练（YOLO）(需要先预处理数据)
    
        ```
        cd yolo_train   # update to the detector when align the performance
        python convert_data.py --data-version dataset_v20250506  # only once, preprocess the training structure, split the training and val set

        python train.py --cfg cfgs/model_yolo11l_v20250604.yaml --name yolov11l_swimAD_v0604_bs64_e250_op_auto --ckpt ../ckpts/yolo11l.pt --device 2,3 --batch 64
        ```

        新进来一批数据进行训练，日志记录在[此处](yolo_train/exp.md)
    4. - [x] 检测模型推理和评测
    
        ```
        cd yolo_train
        python val.py --data-cfg cfgs/data_swimAD_v20250506.yaml --ckpt ./runs/detect/yolov11l_swimAD/weights/best.pt   # load the trained model and data config
        ```
    
    5. - [x] 多目标跟踪训练（）
    
        '''
        暂时没训练reid_model或者end-to-end的trackor，暂时用基于无参的方法和pretrained reid_model, 后续可以训练一版。采用boxmot框架，均是detection+reid范式。
        '''
    
    6. - [x] 多目标跟踪推理和评测 （）
       
        1. -[x] 推理
            1. -[x] 无参跟踪器(ocsort/bytetrack/)，利用--tracking-method指定跟踪器即可

            ```
            python -m tracking.track --source `video_path`[webcam(0)/.mp4/.jpg/path/url] --yolo-model `yolo_ckpt_path` --tracking-method `track_method` --save-txt --device `GPU_id`
            ```

            2. -[x] 有参跟踪器(boosttrack/botsort/strongsort/deepocsort/imprassoc)，利用--tracking-method指定跟踪器，利用--reid-model制定reid参数[lmbn_n_cuhk03_d.pt/osnet_x0_25_market1501.pt/osnet_x1_0_msmt17.pt].

            ```
            python -m tracking.track --source `video_path`[webcam(0)/.mp4/.jpg/path/url] --yolo-model `yolo_ckpt_path` --tracking-method `track_method` --reid-model osnet_x0_25_market1501.pt --save-txt --device `GPU_id`
            ```
        
        2. -[ ] 评测
       
           文件格式要求:
       
           - --source和--custom-gt-folder下都为序列,各序列名字对应相同.
           - 在--source的序列中放图片;--custom-gt-folder的对应序列中放mot标注文件和seqinfo.ini文件.(seqinfo.ini文件已写好于./swimAD/assets,拷贝以使用)
           - mot标注文件要求命名为normal_seq.txt(由labelme_to_mot.py生成的mot文件)
           - <a href="https://drive.google.com/file/d/110us0NPPlGSJuowNxJ_tgrU8eXLkPn4E/view?usp=drive_link" download>dataset_v20250506的annotation_mot_gt文件下载</a>
       
            例如：
        ```
        python -m tracking.val  --source "..images\dataset_v20250506\下午"  --custom-gt-folder "..\annotation_mot_gt\dataset_v20250506\下午" --yolo-model ..\yolov11L_run2_epoch250_batchsize64_imgsize640\weights\yolo11L_epoch250.pt  --tracking-method ocsort --device 0 
        ```


    7. -[x] 单视角溺水检测（基于[规则](rules.md)）
    
        ```
        python -m tracking.swimAD --source `video_path`[webcam(0)/.mp4/.jpg/path/url] --yolo-model `yolo_ckpt_path` --tracking-method ocsort --save-video `save_video_path` --device 0
        ```
        例如：
        ```
        python -m tracking.swimAD --source ../dataset/dataset_v20250506/noon/1/ --yolo-model ../ckpts/yolo11L_epoch250.pt --tracking-method ocsort --save-video swimAD_example.mp4 --show-trajectories --device 0 --log-level DEBUG
        ```
        此处，利用DEBUG LOG记录每一个ID物体的信息，默认存放在logs/time.log中，包含图像路径，物体ID，最远/最近移动距离，10秒内运动距离，类别列表，位置列表等信息
    
    8 -[x] 多视角溺水检测
    ```
    python -m tracking.track_v0702 --source ../dataset/dataset_v20250506/afternoon --yolo-model $checkpoint_path$ --tracking-method ocsort --save-video --device 1 --save-video --save --show-trajectories --log-path log.json --metrics metric1,metric2,...
    ```
    例如：
    ```
    python -m tracking.track_v0702 --source ../dataset/dataset_v20250506/afternoon --yolo-model /home/chaoqunwang/swimAD/yolo_train/runs/detect/11L_v0630+0630shuffle_bs64/weights/best.pt --tracking-method ocsort --save-video --device 1 --save-video --save --show-trajectories --log-path log.json --metrics min_dist,max_dist
    ```
    输出各个指标，可视化用于总结规则
    
    #### TODO List
    1. - [ ] 代码库修改，代码格式对齐一下
    2. - [ ] 用pretrain VL model处理第一批数据，提升检测准确率
    3. - [ ] 设计多视角联合检测问题，主要难点是不同视角下物体的对应关系，应该需要相机参数（位置，角度），也许多视角投影至3D空间进行统一，因为是在2D空间标注的，这个想法需要验证
    4. - [ ] 设计多视角联合跟踪问题
    5. - [x] 基于单视角检测结果，设计rule-based规则，判断是否溺水
    6. - [ ] 更新rules规则，大家可以写在[规则](rules.md)文件中
    7. - [x] 需要重新设计box投影
    8. - [ ] 由于按照投影后的区域划分，中间区域投影不准确，需要换个方式，里面原始区域。所以得标注四个视角的四个区域，并查看边缘部分投影后是否会跳变
    9. - [ ] 由于跳变问题，是否需要每个视角设置一个tracker，用5个tracker进行判断
    10. -[ ] 统计推理速度



### 多视角多目标跟踪（）
1. - [ ] 多视角联合检测

2. - [ ] 多视角联合跟踪

   **指令example**

   ```
   python -m tracking.track_mul --source1 "G:\project\images\test6.2_multiview\1" --source4 "G:\project\images\test6.2_multiview\4" --yolo-model "G:\project\detection_model\yolov11L_run2_epoch250_batchsize64_imgsize640\weights\yolo11L_epoch250.pt" --tracking-method ocsort   --device 0  --show-trajectories  --save
   ```

   (重映射的逻辑还没写,多视角相关文件在swimAD\boxmot\multiview_tool)






### 后处理溺水判定（）
1. -[ ] 后处理规则判定

### 输出处理（李航）




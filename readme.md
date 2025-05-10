# 任务：泳池场景溺水检测

## 任务描述
任务输入: 多视角图像（4个视角），2K分辨率，25Hz

任务输出：溺水人员检测结果

## 整体推理流程
1. - [ ] 多视角相机数据读取
2. - [ ] 多视角联合溺水检测
    1. - [ ] 单视角多目标跟踪
    2. - [ ] 多视角联合跟踪
    3. - [ ] 溺水判定
3. - [ ] 输出处理

### 多视角相机数据读取（李航）
1. - [ ] 需要更新

### 单视角多目标跟踪（李健）
1. - [ ] [数据标注](data_annotation.md)

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
       
    3. - [x] 检测模型训练（YOLO）
    
       ```
       \detector文件夹
       ```
    
       **`train.py`**
    
       - **作用**: yolo训练启动文件
    
       **`train_config.yaml`**
    
       - **作用**: yolo训练配置文件
    
       
    
    4. - [x] 检测模型推理和评测
    
        '''
        脚本
        '''
    
    5. - [x] 多目标跟踪训练（）
    
        '''
        脚本
        '''
    
    6. - [x] 多目标跟踪推理和评测 （）
       
        '''inference

        python -m tracking.track --source $video_path$[webcam(0)/.mp4/.jpg/path/url] --yolo-model $yolo_ckpt_path$ --tracking-method $track_method$[ocsort/deepocsort/...] --save-txt --device 0
        
        python -m tracking.track --source ../dataset/dataset_v20250506/noon/1/ --yolo-model ../ckpts/yolo11L_epoch250.pt --tracking-method ocsort --save-txt --device 0
        '''

    #### TODO List
    1. - [ ] 代码库修改，代码格式对齐一下
    2. - [ ] 用pretrain VL model处理第一批数据，提升检测准确率
    3. - [ ] 设计多视角联合检测问题，主要难点是不同视角下物体的对应关系，应该需要相机参数（位置，角度），也许多视角投影至3D空间进行统一，因为是在2D空间标注的，这个想法需要验证
    4. - [ ] 设计多视角联合跟踪问题
    5. - [ ] 基于单视角检测结果，设计rule-based规则，判断是否溺水



### 多视角多目标跟踪（）
1. - [ ] 多视角联合检测

2. - [ ] 多视角联合跟踪


### 后处理溺水判定（）
1. -[ ] 后处理规则判定

### 输出处理（李航）




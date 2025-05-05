## envs

#pip install boxmot会卸载掉当前带gpu版本的torch！先装了boxmot再装torch，否则无法调用gpu

Python Version: 3.10

```
pip install boxmot

pip install git+https://github.com/mikel-brostrom/ultralytics.git

git clone https://github.com/mikel-brostrom/boxmot.git
```

PyTorch Version: 2.2.1+cu121

torchvision: 0.17.1+cu121



***环境创建示例***

```
conda create --prefix F:\anaconda\envs\boxmot python=3.10 
conda activate boxmot
$env:PYTHONNOUSERSITE=1

pip install boxmot
pip install git+https://github.com/mikel-brostrom/ultralytics.git
git clone https://github.com/mikel-brostrom/boxmot.git

pip3 install torch torchvision  --index-url https://download.pytorch.org/whl/cu121
```



***环境测试查看命令***

````
python -c "import torch; print(torch.cuda.is_available())"

python -c "import torch; import torchvision; print(f'torch: {torch.__version__}\ntorchvision: {torchvision.__version__}')"
````



## detector_pt_files

<a href="https://drive.google.com/file/d/1TV1zhoLgOmN-rk6OALuJQokUof-o7Q6R/view?usp=drive_link" download>yolo8L_240epoch</a>

<a href="https://drive.google.com/file/d/1S7UFK2qqtnmknSQ1PuSWZTBHiajgdZUp/view?usp=drive_link" download>yolo11L_250epoch</a>



## example

### 检测跟踪

```
cd boxmot

python -m tracking.track --source "G:\project\泳池data\video\clip.mp4" --yolo-model "G:\project\detection_model\yolov11L_run2_epoch250_batchsize64_imgsize640\weights\yolo11L_epoch250.pt" --tracking-method ocsort --save --save-txt --device 0
```

--source 指定输入源

--yolo-model 

--tracking-method

--save 保存一个带有可视化追踪结果

--save-txt 保存为MOT文本文件



### 检测跟踪后做mot eval

```
python -m tracking.val  --source "G:\project\20250423游泳池数据最新标签修改返回__processed\中午\test"    --yolo-model G:\project\detection_model\yolov11L_run2_epoch250_batchsize64_imgsize640\weights\yolo11L_epoch250.pt  --tracking-method ocsort --device 0 
```

***source文件格式***

G:\project\20250423游泳池数据最新标签修改返回__processed\中午\test\

​	<sequence

​		<gt

​			gt.txt

​		<img

​			.jpg





## 
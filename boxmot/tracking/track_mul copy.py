# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import os
import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from .detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

import pdb


# ID重映射功能
class IDMapper:
    def __init__(self):
        self.id_mapping = {}  # 原始ID -> 新ID的映射
        self.next_id = 1      # 下一个可用的新ID
        
    def get_mapped_id(self, original_id):
        """获取映射后的ID，如果是新ID则创建映射"""
        if original_id not in self.id_mapping:
            self.id_mapping[original_id] = self.next_id
            self.next_id += 1
        return self.id_mapping[original_id]
    
    def reset_mapping(self):
        """重置ID映射"""
        self.id_mapping = {}
        self.next_id = 1


def custom_id_processing(tracks, id_mapper, frame_count=None):
    """
    自定义ID处理逻辑
    
    Args:
        tracks: 跟踪结果 [x1, y1, x2, y2, track_id, conf, cls, ...]
        id_mapper: ID映射器实例
        frame_count: 当前帧数（可选）
    
    Returns:
        处理后的跟踪结果
    """
    if tracks is None or len(tracks) == 0:
        return tracks
    
    processed_tracks = []
    for track in tracks:
        if len(track) >= 5:  # 确保有track_id
            # 获取原始ID（通常在索引4的位置）
            original_id = int(track[4])            # 将所有ID设为0用于测试
            # new_id = id_mapper.get_mapped_id(original_id)
            new_id = 0
            # 创建新的track副本
            new_track = track.copy()
            new_track[4] = new_id
            
            # 可以在这里添加其他自定义逻辑
            # 例如：根据位置、时间等条件修改ID
            # if frame_count is not None and frame_count % 100 == 0:
            #     # 每100帧重新分配ID
            #     new_id = id_mapper.get_mapped_id(original_id + 1000)
            
            processed_tracks.append(new_track)
        else:
            processed_tracks.append(track)
    
    return processed_tracks


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):

    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)
    yolo = YOLO(
        args.yolo_model if is_ultralytics_model(args.yolo_model)
        else 'yolov8n.pt',
    )
    results = yolo.track(
   
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
       
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if not is_ultralytics_model(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        yolo_model = m(model=args.yolo_model, device=yolo.predictor.device,
                       args=yolo.predictor.args)
        yolo.predictor.model = yolo_model

        # If current model is YOLOX, change the preprocess and postprocess
        if not is_ultralytics_model(args.yolo_model):
            # add callback to save image paths for further processing
            yolo.add_callback(
                "on_predict_batch_start",
                lambda p: yolo_model.update_im_paths(p)
            )
            yolo.predictor.preprocess = (
                lambda imgs: yolo_model.preprocess(im=imgs))
            yolo.predictor.postprocess = (
                lambda preds, im, im0s:
                yolo_model.postprocess(preds=preds, im=im, im0s=im0s))

    # store custom args in predictor
    yolo.predictor.custom_args = args    # 初始化ID映射器
    id_mapper = IDMapper()
    frame_count = 0

    # 初始化保存相关变量
    all_imgs = []
    save_dir = None
    if args.save or args.save_video:
        save_dir = Path(args.project) / args.name
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出将保存到: {save_dir}")

    for r in results:
        frame_count += 1
        
        # 检查是否需要重置ID映射
        if hasattr(args, 'reset_id_interval') and args.reset_id_interval > 0:
            if frame_count % args.reset_id_interval == 0:
                id_mapper.reset_mapping()
                print(f"Frame {frame_count}: ID mapping reset")
        
        # 获取原始跟踪结果
        if hasattr(r, 'boxes') and r.boxes is not None:
            # 从results中提取跟踪信息
            if hasattr(r.boxes, 'id') and r.boxes.id is not None:
                # 构建tracks格式: [x1, y1, x2, y2, id, conf, cls]
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                
                tracks = []
                for i in range(len(boxes)):
                    track = [
                        boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],  # x1,y1,x2,y2
                        ids[i],      # track_id
                        confs[i],    # confidence
                        classes[i]   # class
                    ]
                    tracks.append(track)
                
                # 强制将所有ID设为0（测试用）
                processed_tracks = custom_id_processing(tracks, id_mapper, frame_count)
                
                # 将处理后的ID写回到results中
                if processed_tracks and len(processed_tracks) > 0:
                    new_ids = [track[4] for track in processed_tracks]                    # 创建新的boxes张量，包含更新的ID
                    if r.boxes is not None and len(r.boxes) > 0:
                        # 获取原始boxes数据并克隆避免就地更新错误                        if len(new_ids) == len(r.boxes):
                            # 记录修改前的ID（第5列，索引4）
                            original_ids = r.boxes.data[:, 4].cpu().numpy()
                            
                            # 克隆数据并修改ID列（第5列，索引4）
                            new_data = r.boxes.data.clone()
                            new_data[:, 4] = torch.tensor(new_ids, dtype=new_data.dtype).to(new_data.device)
                            r.boxes.data = new_data
                            
                            # 验证修改后的ID
                            modified_ids = r.boxes.data[:, 4].cpu().numpy()
                            
                            # 保存调试信息到文件
                            debug_file_path = os.path.join(args.project, args.name, 'id_debug.txt')
                            os.makedirs(os.path.dirname(debug_file_path), exist_ok=True)
                            
                            with open(debug_file_path, 'a', encoding='utf-8') as f:
                                f.write(f"\n=== Frame {frame_count} ===\n")
                                f.write(f"检测到目标数量: {len(new_ids)}\n")
                                f.write(f"原始IDs: {original_ids}\n")
                                f.write(f"修改后的IDs: {modified_ids}\n")
                                f.write(f"预期的new_ids: {new_ids}\n")
                                
                                # 验证r.boxes.id属性（如果存在）
                                if hasattr(r.boxes, 'id') and r.boxes.id is not None:
                                    boxes_id_values = r.boxes.id.cpu().numpy()
                                    f.write(f"r.boxes.id属性值: {boxes_id_values}\n")
                                
                                # 验证修改是否成功
                                modification_success = all(id_val == 0 for id_val in modified_ids)
                                f.write(f"ID修改成功: {modification_success}\n")
                                f.write("-" * 50 + "\n")
                    
                    # 打印ID映射信息
                    if frame_count % 30 == 0:  # 每30帧打印一次
                        print(f"Frame {frame_count}: 所有ID已设为0，检测到 {len(new_ids)} 个目标")
        
        # 绘制结果
        img = r.orig_img.copy()
        
        # 如果有检测结果，直接在图像上绘制
        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
            annotator = Annotator(img, line_width=args.line_width)
            
            for i, box in enumerate(r.boxes.xyxy):
                # 获取框坐标
                x1, y1, x2, y2 = box.cpu().numpy()
                  # 获取ID（从第5列，索引4获取修改后的track_id）
                track_id = int(r.boxes.data[i, 4].cpu().numpy()) if r.boxes.data.shape[1] > 4 else 0
                
                # 获取置信度和类别
                conf = r.boxes.conf[i].cpu().numpy() if hasattr(r.boxes, 'conf') else 0.0
                cls = int(r.boxes.cls[i].cpu().numpy()) if hasattr(r.boxes, 'cls') else 0
                
                # 构建标签
                label = f"ID:{track_id}"
                if args.show_conf:
                    label += f" {conf:.2f}"
                if args.show_labels and hasattr(r, 'names'):
                    label += f" {r.names[cls]}"
                  # 绘制框和标签 - 使用类别ID决定颜色
                annotator.box_label((x1, y1, x2, y2), label, color=colors(cls, True))
            
            img = annotator.result()
        else:
            # 如果没有修改后的results，使用原始tracker绘制
            img = yolo.predictor.trackers[0].plot_results(r.orig_img, args.show_trajectories)        # 保存结果
        if args.save or args.save_video:
            # 创建保存目录
            save_dir = Path(args.project) / args.name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存单张图片（如果启用了save）
            if args.save:
                img_path = save_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(img_path), img)
            
            # 收集用于视频保存的图片
            if args.save_video:
                all_imgs.append(img)

        if args.show is True:
            cv2.imshow('BoxMOT', img)     
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break

    # 保存视频（如果收集了图片）
    if args.save_video and len(all_imgs) > 0:
        save_dir = Path(args.project) / args.name
        save_dir.mkdir(parents=True, exist_ok=True)
        video_path = save_dir / 'tracking_output.mp4'
        
        frame_size = all_imgs[0].shape[1], all_imgs[0].shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # 使用更合理的帧率
        out = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
        for img in all_imgs:
            out.write(img)
        out.release()
        print(f"视频已保存到: {video_path}")


def parse_opt():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc, boosttrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=None,
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    parser.add_argument('--save-video', action='store_true',
                        help='save video tracking results in .mp4 format')

    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show confidences')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--id-remapping', action='store_true',
                        help='enable custom ID remapping')
    parser.add_argument('--reset-id-interval', type=int, default=0,
                        help='reset ID mapping every N frames (0 = disabled)')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)

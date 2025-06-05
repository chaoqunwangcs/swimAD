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
            original_id = int(track[4])
            # 将所有ID设为0用于测试
            # new_id = id_mapper.get_mapped_id(original_id)
            new_id = 0
            # 创建新的track副本
            new_track = track.copy()
            new_track[4] = new_id
            
            processed_tracks.append(new_track)
        else:
            processed_tracks.append(track)
    
    return processed_tracks


def process_single_frame(result, id_mapper, source_name, frame_count):
    """处理单个帧的逻辑"""
    if hasattr(result, 'boxes') and result.boxes is not None:
        if hasattr(result.boxes, 'id') and result.boxes.id is not None:
            # 构建tracks格式: [x1, y1, x2, y2, id, conf, cls]
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
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
                new_ids = [track[4] for track in processed_tracks]
                if result.boxes is not None and len(result.boxes) > 0:
                    if len(new_ids) == len(result.boxes):
                        # 克隆数据并修改ID列（第5列，索引4）
                        new_data = result.boxes.data.clone()
                        new_data[:, 4] = torch.tensor(new_ids, dtype=new_data.dtype).to(new_data.device)
                        result.boxes.data = new_data
    
    return result


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


def render_frame(result, args, source_name):
    """渲染单个帧"""
    img = result.orig_img.copy()
    
    # 如果有检测结果，直接在图像上绘制
    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
        annotator = Annotator(img, line_width=args.line_width)
        
        for i, box in enumerate(result.boxes.xyxy):
            # 获取框坐标
            x1, y1, x2, y2 = box.cpu().numpy()
            # 获取ID（从第5列，索引4获取修改后的track_id）
            track_id = int(result.boxes.data[i, 4].cpu().numpy()) if result.boxes.data.shape[1] > 4 else 0
            
            # 获取置信度和类别
            conf = result.boxes.conf[i].cpu().numpy() if hasattr(result.boxes, 'conf') else 0.0
            cls = int(result.boxes.cls[i].cpu().numpy()) if hasattr(result.boxes, 'cls') else 0
            
            # 构建标签
            label = f"ID:{track_id}"
            if args.show_conf:
                label += f" {conf:.2f}"
            if args.show_labels and hasattr(result, 'names'):
                label += f" {result.names[cls]}"
            
            # 绘制框和标签 - 使用类别ID决定颜色
            annotator.box_label((x1, y1, x2, y2), label, color=colors(cls, True))
        
        img = annotator.result()
    
    return img


@torch.no_grad()
def run_dual_source(args):
    """双源同步处理"""
    # 创建两个独立的YOLO实例
    yolo1 = YOLO(args.yolo_model if is_ultralytics_model(args.yolo_model) else 'yolov8n.pt')
    yolo4 = YOLO(args.yolo_model if is_ultralytics_model(args.yolo_model) else 'yolov8n.pt')
    
    # 获取两个同步的结果流
    results1 = yolo1.track(
        source=args.source1,
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
        name=f"{args.name}_source1",
        classes=args.classes,
        imgsz=args.imgsz or default_imgsz(args.yolo_model),
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )
    
    results4 = yolo4.track(
        source=args.source4,
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
        name=f"{args.name}_source4",
        classes=args.classes,
        imgsz=args.imgsz or default_imgsz(args.yolo_model),
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )
    
    # 添加回调
    yolo1.add_callback('on_predict_start', partial(on_predict_start, persist=True))
    yolo4.add_callback('on_predict_start', partial(on_predict_start, persist=True))
    
    # 设置自定义参数
    yolo1.predictor.custom_args = args
    yolo4.predictor.custom_args = args
    
    # 初始化两个独立的ID映射器
    id_mapper1 = IDMapper()
    id_mapper4 = IDMapper()
    frame_count = 0
    
    # 初始化保存目录
    if args.save or args.save_video:
        save_dir1 = Path(args.project) / f"{args.name}_source1"
        save_dir4 = Path(args.project) / f"{args.name}_source4"
        save_dir1.mkdir(parents=True, exist_ok=True)
        save_dir4.mkdir(parents=True, exist_ok=True)
        print(f"Source1输出将保存到: {save_dir1}")
        print(f"Source4输出将保存到: {save_dir4}")
    
    all_imgs1 = []
    all_imgs4 = []
    
    # 关键：zip确保帧同步
    for r1, r4 in zip(results1, results4):
        frame_count += 1
        
        # 串行处理source1
        processed_r1 = process_single_frame(r1, id_mapper1, "source1", frame_count)
        
        # 串行处理source4  
        processed_r4 = process_single_frame(r4, id_mapper4, "source4", frame_count)
        
        # 渲染两个帧
        img1 = render_frame(processed_r1, args, "source1")
        img4 = render_frame(processed_r4, args, "source4")
        
        # 保存结果
        if args.save or args.save_video:
            if args.save:
                # 保存单张图片到各自目录
                img_path1 = save_dir1 / f"frame_{frame_count:06d}.jpg"
                img_path4 = save_dir4 / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(img_path1), img1)
                cv2.imwrite(str(img_path4), img4)
            
            # 收集用于视频保存的图片
            if args.save_video:
                all_imgs1.append(img1)
                all_imgs4.append(img4)
        
        # 显示
        if args.show:
            # 并排显示两个视频
            combined_img = np.hstack([img1, img4])
            cv2.imshow('BoxMOT - Source1 | Source4', combined_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break
        
        # 打印进度
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: 双源处理完成")
    
    # 保存视频
    if args.save_video:
        if len(all_imgs1) > 0:
            video_path1 = save_dir1 / 'tracking_output.mp4'
            frame_size = all_imgs1[0].shape[1], all_imgs1[0].shape[0]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            out1 = cv2.VideoWriter(str(video_path1), fourcc, fps, frame_size)
            for img in all_imgs1:
                out1.write(img)
            out1.release()
            print(f"Source1视频已保存到: {video_path1}")
        
        if len(all_imgs4) > 0:
            video_path4 = save_dir4 / 'tracking_output.mp4'
            frame_size = all_imgs4[0].shape[1], all_imgs4[0].shape[0]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            out4 = cv2.VideoWriter(str(video_path4), fourcc, fps, frame_size)
            for img in all_imgs4:
                out4.write(img)
            out4.release()
            print(f"Source4视频已保存到: {video_path4}")


@torch.no_grad()
def run_single_source(args):
    """单源处理（原有逻辑）"""
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
    yolo.predictor.custom_args = args
    
    # 初始化ID映射器
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
        
        # 处理单帧
        processed_r = process_single_frame(r, id_mapper, "single", frame_count)
        
        # 渲染
        img = render_frame(processed_r, args, "single")
        
        # 保存和显示
        if args.save or args.save_video:
            if args.save:
                img_path = save_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(img_path), img)
            
            if args.save_video:
                all_imgs.append(img)

        if args.show is True:
            cv2.imshow('BoxMOT', img)     
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break

    # 保存视频
    if args.save_video and len(all_imgs) > 0:
        video_path = save_dir / 'tracking_output.mp4'
        frame_size = all_imgs[0].shape[1], all_imgs[0].shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        out = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
        for img in all_imgs:
            out.write(img)
        out.release()
        print(f"视频已保存到: {video_path}")


@torch.no_grad()
def run(args):
    """主运行函数 - 自动检测单源或双源模式"""
    # 检查是否是双源模式
    if hasattr(args, 'source1') and hasattr(args, 'source4') and args.source1 and args.source4:
        print("检测到双源模式，启动双源同步处理...")
        return run_dual_source(args)
    else:
        print("使用单源模式...")
        return run_single_source(args)


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
    parser.add_argument('--source1', type=str, default=None,
                        help='first source for dual tracking')
    parser.add_argument('--source4', type=str, default=None,
                        help='fourth source for dual tracking')
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

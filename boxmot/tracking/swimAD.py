# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import os
import argparse
import cv2
import numpy as np
import json
from functools import partial
from pathlib import Path
from datetime import datetime
now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS

from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

from boxmot.utils import logger as LOGGER

import pdb


def plot_id(img, frame_id):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (255, 255, 255)  # 白色
    font_thickness = 2

    height, width = img.shape[:2]

    # 计算文本的位置（右上角）
    line = f"FrameID: {frame_id:04d}"
    text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
    text_x = width - text_size[0] - 10  # 距离右侧边缘10像素
    text_y = text_size[1] + 10  # 距离顶部边缘10像素

    # 在图像上绘制文本
    cv2.putText(img, line, (text_x, text_y), font, font_scale, font_color, font_thickness)

    return img

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

def convert_np(obj):
    """递归将 numpy 类型转为 Python 原生类型"""
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

@torch.no_grad()
def run(args):

    LOGGER.add(args.log_path, format="{time} - {level} - {message}", level=args.log_level, rotation="100 MB")

    # 新增：初始化json数据结构
    json_data = {
        "video_id": str(args.source),
        "time_str": now,
        "frame_data": {}
    }

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
        save=args.save,
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
        m = get_yolo_inferer(args.yolo_model)
        yolo_model = m(model=args.yolo_model, device=yolo.predictor.device,
                       args=yolo.predictor.args)
        yolo.predictor.model = yolo_model

        if not is_ultralytics_model(args.yolo_model):
            yolo.add_callback(
                "on_predict_batch_start",
                lambda p: yolo_model.update_im_paths(p)
            )
            yolo.predictor.preprocess = (
                lambda imgs: yolo_model.preprocess(im=imgs))
            yolo.predictor.postprocess = (
                lambda preds, im, im0s:
                yolo_model.postprocess(preds=preds, im=im, im0s=im0s))

    yolo.predictor.custom_args = args

    if args.save_video is not None:
        all_imgs = []

    try:
        for frame_id, r in enumerate(results):
            img = yolo.predictor.trackers[0].plot_plain_results(r.orig_img, args.show_trajectories, fontscale=2)
            img = plot_id(img, frame_id+1)
            cv2.imwrite('aa.jpg', img)
            AD_list, info_list = yolo.predictor.trackers[0].detect_AD()
            if len(AD_list) > 0:
                for AD in AD_list:
                    img = yolo.predictor.trackers[0].plot_AD_results(r.orig_img, args.show_trajectories, AD_list)

            if args.save_video is not None:
                all_imgs.append(img)

            if args.show is True:
                cv2.imshow('BoxMOT', img)     
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') or key == ord('q'):
                    break

            # update the info into log file
            if len(info_list) > 0:
                LOGGER.debug(f"Image path: {r.path}")
                frame_key = f"{frame_id+1:04d}"
                json_data["frame_data"][frame_key] = {}
                for object_info in info_list:
                    for rule_name, rule_info in object_info.items():
                        obj_key = f"obj{rule_info['id']}"
                        cls_list = rule_info.get("cls_list", [])
                        class_label = cls_list[-1] if len(cls_list) > 0 else None
                        # 保证所有指标都写入
                        bbox = rule_info.get("bbox")
                        json_data["frame_data"][frame_key][obj_key] = {
                            "class_label": class_label,
                            "min_dist": rule_info.get("min_dist"),
                            "max_dist": rule_info.get("max_dist"),
                            "move_dist": rule_info.get("move_dist"),
                            "avg_scale": rule_info.get("avg_scale"),
                            "is_AD": rule_info.get("is_AD"),
                            "condition_A_triggered": rule_info.get("condition_A_triggered"),
                            "condition_B_triggered": rule_info.get("condition_B_triggered"),
                            "final_ema_magnitude": rule_info.get("final_ema_magnitude"),
                            "cos_theta_values": rule_info.get("cos_theta_values"),
                            "displacement_magnitudes_list": rule_info.get("displacement_magnitudes_list"),
                            "cls_flag_triggered": rule_info.get("cls_flag_triggered"),
                            "scale_list": rule_info.get("scale_list"),
                            "cls_list": rule_info.get("cls_list"),
                            "traj_len": rule_info.get("traj_len"),
                            "id": rule_info.get("id"),
                            "new_movement_drowning_flag": rule_info.get("new_movement_drowning_flag"),
                            "bbox_left_top": [bbox[0], bbox[1]] if bbox else None,         # 左上角 [x1, y1]
                            "bbox_right_bottom": [bbox[2], bbox[3]] if bbox else None      # 右下角 [x2, y2]
                        }
    finally:
        # 保存json文件，递归转换numpy类型
        if args.json_path:
            os.makedirs(os.path.dirname(args.json_path), exist_ok=True)
            with open(args.json_path, "w", encoding="utf-8") as f:
                json.dump(convert_np(json_data), f, ensure_ascii=False, indent=2)
            print(f"Saved json results to {args.json_path}")

        # 保存视频
        if args.save_video is not None and 'all_imgs' in locals() and len(all_imgs) > 0:
            os.makedirs(args.project, exist_ok=True)
            video_path = os.path.join(args.project, args.save_video)
            frame_size = all_imgs[0].shape[1], all_imgs[0].shape[0]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 2
            out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
            for img in all_imgs:
                out.write(img)
            out.release()
            print(f"Saving video in {video_path}.")

        # 新增：保存图片包
        if args.save_imgs is not None and 'all_imgs' in locals() and len(all_imgs) > 0:
            img_dir = os.path.join(args.project, args.save_imgs)
            os.makedirs(img_dir, exist_ok=True)
            for idx, img in enumerate(all_imgs):
                img_path = os.path.join(img_dir, f"{idx+1:04d}.jpg")
                cv2.imwrite(img_path, img)
            print(f"Saved image package to {img_dir}")
def parse_opt():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-imgs', type=str, default=None,
                        help='save image package to this folder (relative to project)')
    parser.add_argument('--json-path', type=str, default=f"logs/{now}.json",
                        help='save json file path')
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
    parser.add_argument('--save-video', type=str, default=None,
                        help='save video tracking results path in .mp4 format')

    parser.add_argument('--log-path', type=str, default=f"logs/{now}.log",
                        help='save log file path')
    parser.add_argument('--log-level', type=str, default="INFO",
                        help='the log level')

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

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)

# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import os, glob
import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path
import math, copy

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
from ultralytics.data.utils import VID_FORMATS, IMG_FORMATS
from ultralytics.utils.plotting import save_one_box
from ultralytics.engine.results import Results, Boxes

from boxmot.multi_view_association.test_stream import MultiViewAssociationStream
from ultralytics.trackers.track import on_predict_postprocess_end


import pdb


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
    # pdb.set_trace()
    dirs = os.listdir(args.sources)
    view_nums = len(dirs)
    yolos = [YOLO(args.yolo_model) for _ in range(view_nums+1)]
    
    results = []
    for idx, yolo in enumerate(yolos):
        result = yolo.track(
            source=os.path.join(args.sources, dirs[idx % view_nums]),
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
            line_width=args.line_width,
        )
        

        yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

        # store custom args in predictor
        yolo.predictor.custom_args = args
        results.append(result)

    if args.save_video is True:
        all_imgs = []

    # pdb.set_trace()
    # setup main view tracker
    print('init the main view tracker...')
    # pdb.set_trace()
    main_view_predictor = yolos[-1].predictor
    main_view_predictor.setup_source(args.sources[0])
    on_predict_start(main_view_predictor)
    # main_view_predictor.trackers[0].w, main_view_predictor.trackers[0].h = 2100, 3100
    print('init the multi view associator...')
    assocaition = MultiViewAssociationStream(r'boxmot/multi_view_association/calibration_v1.json')
    
    print('running the multi view stream...')
    for (r1, r2, r3, r4) in zip(results[0], results[1], results[2], results[3]):
        view_data = [r1.boxes.data.cpu().numpy(), r2.boxes.data.cpu().numpy(), r3.boxes.data.cpu().numpy(), r4.boxes.data.cpu().numpy()]

        # filter the ashore person
        multi_view_data = [arr[arr[:, -1] != 0] for arr in view_data]
        main_view_data = assocaition.forward(multi_view_data)
        main_view_data = [[x[1].x1, x[1].y1, x[1].x2, x[1].y2, x[1].conf, x[1].cls_id] for x in main_view_data.association_data['MAIN']]
        main_view_data = np.stack(main_view_data)
        main_view_data[:, -1] = np.round(main_view_data[:, -1])

        # filter the object out of the range
        limits = [(0, 2100), (0, 3100), (0, 2100), (0, 3100), (0, 1), (1, 2)]
        bool_indices = np.ones(main_view_data.shape[0], dtype=bool)  # 初始全为 True
        for i, (min_val, max_val) in enumerate(limits):
            bool_indices &= (main_view_data[:, i] >= min_val) & (main_view_data[:, i] <= max_val) 
        main_view_data = main_view_data[bool_indices]
        
        new_img =  np.zeros((3100, 2100, 3), dtype=np.uint8)
        cv2.rectangle(new_img, (300, 300), (1800, 2800),  (0, 255, 0), 3)   # margin
        for box in main_view_data:
            box, conf, cls_id = box[:4], box[4], box[5]
            cv2.rectangle(new_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),  (0, 255, 0), 3)
            # cv2.imwrite('ab.jpg', new_img)
            # print(box-300)
            # pdb.set_trace()
        new_img = cv2.resize(new_img, (2100, 2880), interpolation=cv2.INTER_LINEAR)

        # pdb.set_trace()
        main_view_predictor.vid_path = ['aa.jpg']
        main_view_predictor.results = [
            Results(
                orig_img = np.zeros((3100, 2100, 3), dtype=np.uint8),
                path = 'aa.jpg',
                names = {0: 'ashore', 1: 'above', 2: 'under'},
                boxes = torch.as_tensor(main_view_data).to(torch.device(f"cuda:{args.device}")),
            )
        ]

        # pdb.set_trace()
        on_predict_postprocess_end(main_view_predictor)
        
        img1 = yolos[0].predictor.trackers[0].plot_results(r1.orig_img, args.show_trajectories, fontscale=1, thickness=4)
        img2 = yolos[1].predictor.trackers[0].plot_results(r2.orig_img, args.show_trajectories, fontscale=1, thickness=4)
        img3 = yolos[2].predictor.trackers[0].plot_results(r3.orig_img, args.show_trajectories, fontscale=1, thickness=4)
        img4 = yolos[3].predictor.trackers[0].plot_results(r4.orig_img, args.show_trajectories, fontscale=1, thickness=4)
        main_img = main_view_predictor.trackers[0].plot_results(np.zeros((3100, 2100, 3), dtype=np.uint8), args.show_trajectories, fontscale=1)

        
        main_img = cv2.resize(main_img, (2100, 2880), interpolation=cv2.INTER_LINEAR)
        img = np.hstack((np.vstack((np.hstack((img1, img2)), np.hstack((img3, img4)))),main_img,new_img))
        img = cv2.resize(img, None, fx=0.5,fy=0.5, interpolation=cv2.INTER_LINEAR)
        if args.save_video is True:
            all_imgs.append(img)

        print(main_view_data)
        cv2.imwrite('aa.jpg', img)
        pdb.set_trace()
        if args.show is True:
            cv2.imshow('BoxMOT', img)     
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break


    if args.save_video is True:
        os.makedirs(args.project, exist_ok=True)
        video_path = os.path.join(args.project, 'output_video.mp4')
        frame_size = all_imgs[0].shape[1], all_imgs[0].shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 2
        out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        for img in all_imgs:
            out.write(img)
        out.release()


def parse_opt():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc, boosttrack')
    parser.add_argument('--sources', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=None,
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--batch', type=int, default=1,
                        help='batch size. 4 for multi view inference')
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

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)

# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
import math
import os
import argparse
import cv2
import glob

import copy
from PIL import Image
import numpy as np
import threading
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple, Optional
from datetime import datetime
import torch
import json
import logging

from tracking.detection_entry import DetectionEntry
from tracking.drowning_detector import DrowningDetector
from tracking.track_lifecycle import TrackManagerController
from tracking.track_window_manager import TrackWindowManager

logger = logging.getLogger(__name__)

from ultralytics.data.augment import classify_transforms
from ultralytics.engine.predictor import STREAM_WARNING

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git',))  # install

from ultralytics import YOLO
from ultralytics.models import yolo
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS, FORMATS_HELP_MSG
from ultralytics.utils.plotting import save_one_box
from ultralytics.utils.checks import check_imgsz, check_imshow, check_requirements
from ultralytics.engine.results import Results
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.data.build import check_source
from ultralytics.utils.patches import imread
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from ultralytics.data.utils import IMG_FORMATS, PIN_MEMORY, VID_FORMATS
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    SETTINGS,
    callbacks,
    checks,
    emojis,
    yaml_load,
)

from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)

from boxmot.multi_view_association.test_stream_split import MultiViewAssociationStream, POOL_WIDTH, POOL_HEIGHT, \
    MARGIN_WIDTH, MARGIN_HEIGHT, MAIN_VIEW

import pdb


def plot_ids(
        result,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
        color_mode="class",
        txt_color=(255, 255, 255),
        ignore_ids=[0],
):
    """
    Plots detection results on an input RGB image.

    Args:
        conf (bool): Whether to plot detection confidence scores.
        line_width (float | None): Line width of bounding boxes. If None, scaled to image size.
        font_size (float | None): Font size for text. If None, scaled to image size.
        font (str): Font to use for text.
        pil (bool): Whether to return the image as a PIL Image.
        img (np.ndarray | None): Image to plot on. If None, uses original image.
        im_gpu (torch.Tensor | None): Normalized image on GPU for faster mask plotting.
        kpt_radius (int): Radius of drawn keypoints.
        kpt_line (bool): Whether to draw lines connecting keypoints.
        labels (bool): Whether to plot labels of bounding boxes.
        boxes (bool): Whether to plot bounding boxes.
        masks (bool): Whether to plot masks.
        probs (bool): Whether to plot classification probabilities.
        show (bool): Whether to display the annotated image.
        save (bool): Whether to save the annotated image.
        filename (str | None): Filename to save image if save is True.
        color_mode (bool): Specify the color mode, e.g., 'instance' or 'class'. Default to 'class'.
        txt_color (tuple[int, int, int]): Specify the RGB text color for classification task

    Returns:
        (np.ndarray): Annotated image as a numpy array.

    Examples:
        >>> results = model("image.jpg")
        >>> for result in results:
        >>>     im = result.plot()
        >>>     im.show()
    """
    assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
    if img is None and isinstance(result.orig_img, torch.Tensor):
        img = (result.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

    names = result.names
    is_obb = result.obb is not None
    pred_boxes, show_boxes = result.obb if is_obb else result.boxes, boxes
    pred_masks, show_masks = result.masks, masks
    pred_probs, show_probs = result.probs, probs
    annotator = Annotator(
        copy.deepcopy(result.orig_img if img is None else img),
        line_width,
        font_size,
        font,
        pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
        example=names,
    )

    # Plot Detect results
    if pred_boxes is not None and show_boxes:
        for i, d in enumerate(reversed(pred_boxes)):
            c, d_conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
            if c in ignore_ids:
                continue
            name = ("" if id is None else f"id:{id} ") + names[c]
            label = (f"{name} {d_conf:.2f}" if conf else name) if labels else None
            box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
            annotator.box_label(
                box,
                label,
                color=colors(
                    c
                    if color_mode == "class"
                    else id
                    if id is not None
                    else i
                    if color_mode == "instance"
                    else None,
                    True,
                ),
                rotated=is_obb,
            )
    # Show results
    if show:
        annotator.show(result.path)

    # Save results
    if save:
        annotator.save(filename)

    return annotator.im if pil else annotator.result()


class LoadMultiViewImagesAndVideos:
    """
    A class for loading and processing images and videos for YOLO object detection.

    This class manages the loading and pre-processing of image and video data from various sources, including
    single image files, video files, and lists of image and video paths.

    Attributes:
        files (List[str]): List of image and video file paths.
        nf (int): Total number of files (images and videos).
        video_flag (List[bool]): Flags indicating whether a file is a video (True) or an image (False).
        mode (str): Current mode, 'image' or 'video'.
        vid_stride (int): Stride for video frame-rate.
        bs (int): Batch size.
        cap (cv2.VideoCapture): Video capture object for OpenCV.
        frame (int): Frame counter for video.
        frames (int): Total number of frames in the video.
        count (int): Counter for iteration, initialized at 0 during __iter__().
        ni (int): Number of images.

    Methods:
        __init__: Initialize the LoadImagesAndVideos object.
        __iter__: Returns an iterator object for VideoStream or ImageFolder.
        __next__: Returns the next batch of images or video frames along with their paths and metadata.
        _new_video: Creates a new video capture object for the given path.
        __len__: Returns the number of batches in the object.

    Examples:
        >>> loader = LoadImagesAndVideos("path/to/data", batch=32, vid_stride=1)
        >>> for paths, imgs, info in loader:
        ...     # Process batch of images or video frames
        ...     pass

    Notes:
        - Supports various image formats including HEIC.
        - Handles both local files and directories.
        - Can read from a text file containing paths to images and videos.
    """

    def __init__(self, path, batch=1, vid_stride=1):
        """Initialize dataloader for images and videos, supporting various input formats."""
        parent = None
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            parent = Path(path).parent
            path = Path(path).read_text().splitlines()  # list of sources
        files = []

        self.views = os.listdir(path)
        self.num_v = len(self.views)

        files = {}

        for view in self.views:
            images = []
            view_path = os.path.join(path, view)
            a = str(Path(view_path).absolute())
            images.extend(sorted(glob.glob(os.path.join(a, "*.*"))))
            images = [x for x in images if x.split(".")[-1].lower() in IMG_FORMATS]
            files[view] = images

        ni, nv = len(images), 0

        self.files = files
        self.nf = ni + nv  # number of files
        self.ni = ni  # number of images
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "video" if ni == 0 else "image"  # default to video if no images
        self.vid_stride = vid_stride  # video frame-rate stride
        self.bs = batch
        self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f"No images or videos found in {path}. {FORMATS_HELP_MSG}")

    def __iter__(self):
        """Iterates through image/video files, yielding source paths, images, and metadata."""
        self.count = 0
        return self

    def __next__(self):
        """Returns the next batch of images or video frames with their paths and metadata."""
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs * self.num_v:
            if self.count >= self.nf:  # end of file list
                if imgs:
                    return paths, imgs, info  # return last partial batch
                else:
                    raise StopIteration

            for view in self.views:
                path = self.files[view][self.count]
                if self.video_flag[self.count]:
                    self.mode = "video"
                    if not self.cap or not self.cap.isOpened():
                        self._new_video(path)

                    success = False
                    for _ in range(self.vid_stride):
                        success = self.cap.grab()
                        if not success:
                            break  # end of video or failure

                    if success:
                        success, im0 = self.cap.retrieve()
                        if success:
                            self.frame += 1
                            paths.append(path)
                            imgs.append(im0)
                            info.append(f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: ")
                            if self.frame == self.frames:  # end of video
                                self.count += 1
                                self.cap.release()
                    else:
                        # Move to the next file if the current video ended or failed to open
                        self.count += 1
                        if self.cap:
                            self.cap.release()
                        if self.count < self.nf:
                            self._new_video(self.files[self.count])
                else:
                    # Handle image files (including HEIC)
                    self.mode = "image"
                    if path.split(".")[-1].lower() == "heic":
                        # Load HEIC image using Pillow with pillow-heif
                        check_requirements("pillow-heif")

                        from pillow_heif import register_heif_opener

                        register_heif_opener()  # Register HEIF opener with Pillow
                        with Image.open(path) as img:
                            im0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # convert image to BGR nparray
                    else:
                        im0 = imread(path)  # BGR
                    if im0 is None:
                        LOGGER.warning(f"WARNING ⚠️ Image Read Error {path}")
                    else:
                        paths.append(path)
                        imgs.append(im0)
                        info.append(f"image {self.count + 1}/{self.nf} {path}: ")
            self.count += 1  # move to the next file
            if self.count >= self.ni:  # end of image list
                break

        return paths, imgs, info

    def _new_video(self, path):
        """Creates a new video capture object for the given path and initializes video-related attributes."""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {path}")
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self):
        """Returns the number of files (images and videos) in the dataset."""
        return math.ceil(self.nf / self.bs)  # number of batches


def load_inference_source(source=None, batch=1, vid_stride=1, buffer=False):
    """
    Load an inference source for object detection and apply necessary transformations.

    Args:
        source (str | Path | torch.Tensor | PIL.Image | np.ndarray, optional): The input source for inference.
        batch (int, optional): Batch size for dataloaders.
        vid_stride (int, optional): The frame interval for video sources.
        buffer (bool, optional): Whether stream frames will be buffered.

    Returns:
        (Dataset): A dataset object for the specified input source with attached source_type attribute.
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # dataloader
    dataset = LoadMultiViewImagesAndVideos(source, batch=batch, vid_stride=vid_stride)
    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset


class MyDetectionPredictor(yolo.detect.DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the BasePredictor class.

        Args:
            cfg (str | dict): Path to a configuration file or a configuration dictionary.
            overrides (dict | None): Configuration overrides.
            _callbacks (dict | None): Dictionary of callback functions.
        """
        # super().__init__(cfg, overrides, _callbacks)

        if 'calibration' in overrides:
            cfg.calibration = overrides['calibration']
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.associator = None
        self.object_map = {}
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}  # dict of {save_path: video_writer, ...}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # for automatic thread-safe inference
        callbacks.add_integration_callbacks(self)

    def setup_associator(self, calibration):
        self.associator = MultiViewAssociationStream(calibration)

    def setup_source(self, source):
        """
        Set up source and inference mode.

        Args:
            source (str | Path | List[str] | List[Path] | List[np.ndarray] | np.ndarray | torch.Tensor):
                Source for inference.
        """
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.imgsz[0], crop_fraction=self.args.crop_fraction),
            )
            if self.args.task == "classify"
            else None
        )
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
        )
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (
                self.source_type.stream
                or self.source_type.screenshot
                or len(self.dataset) > 1000  # many images
                or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """
        Stream real-time inference on camera feed and save results to file.

        Args:
            source (str | Path | List[str] | List[Path] | List[np.ndarray] | np.ndarray | torch.Tensor | None):
                Source for inference.
            model (str | Path | torch.nn.Module | None): Model for inference.
            *args (Any): Additional arguments for the inference method.
            **kwargs (Any): Additional keyword arguments for the inference method.

        Yields:
            (ultralytics.engine.results.Results): Results objects.
        """
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        if not hasattr(self, 'associator') or not self.associator:
            self.setup_associator(self.args.calibration)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(imgsz=(
                    self.dataset.num_v if self.model.pt or self.model.triton else self.dataset.bs * self.dataset.num_v,
                    3,
                    *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s[i] += self.write_results(i, Path(paths[i]), im, s)

                # Print batch results
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from [self.results]

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        # Print final results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), 3, *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")


class MYYOLO(YOLO):
    def track(
            self,
            source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
            stream: bool = False,
            persist: bool = False,
            **kwargs: Any,
    ) -> List[Results]:
        """
        Conducts object tracking on the specified input source using the registered trackers.

        This method performs object tracking using the model's predictors and optionally registered trackers. It handles
        various input sources such as file paths or video streams, and supports customization through keyword arguments.
        The method registers trackers if not already present and can persist them between calls.

        Args:
            source (Union[str, Path, int, List, Tuple, np.ndarray, torch.Tensor], optional): Input source for object
                tracking. Can be a file path, URL, or video stream.
            stream (bool): If True, treats the input source as a continuous video stream.
            persist (bool): If True, persists trackers between different calls to this method.
            **kwargs (Any): Additional keyword arguments for configuring the tracking process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of tracking results, each a Results object.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.track(source="path/to/video.mp4", show=True)
            >>> for r in results:
            ...     print(r.boxes.id)  # print tracking IDs

        Notes:
            - This method sets a default confidence threshold of 0.1 for ByteTrack-based tracking.
            - The tracking mode is explicitly set in the keyword arguments.
            - Batch size is set to 1 for tracking in videos.
        """
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs["batch"] = kwargs.get("batch") or 1  # batch-size 1 for tracking in videos
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": MyDetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }

    def predict(
            self,
            source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
            stream: bool = False,
            predictor=None,
            **kwargs: Any,
    ) -> List[Results]:
        """
        Performs predictions on the given image source using the YOLO model.

        This method facilitates the prediction process, allowing various configurations through keyword arguments.
        It supports predictions with custom predictors or the default predictor method. The method handles different
        types of image sources and can operate in a streaming mode.

        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple): The source
                of the image(s) to make predictions on. Accepts various types including file paths, URLs, PIL
                images, numpy arrays, and torch tensors.
            stream (bool): If True, treats the input source as a continuous stream for predictions.
            predictor (BasePredictor | None): An instance of a custom predictor class for making predictions.
                If None, the method uses a default predictor.
            **kwargs (Any): Additional keyword arguments for configuring the prediction process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, each encapsulated in a
                Results object.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.predict(source="path/to/image.jpg", conf=0.25)
            >>> for r in results:
            ...     print(r.boxes.data)  # print detection bounding boxes

        Notes:
            - If 'source' is not provided, it defaults to the ASSETS constant with a warning.
            - The method sets up a new predictor if not already present and updates its arguments with each call.
            - For SAM-type models, 'prompts' can be passed as a keyword argument.
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )
        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict"}  # method defaults
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right
        prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor:
            self.predictor = (predictor or self._smart_load("predictor"))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)


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


def on_predict_postprocess_end(predictor: Union[object, MyDetectionPredictor], persist: bool = False) -> None:
    """
    Postprocess YOLO predictions across multiple views, perform association,
    construct object map, and update tracker.

    Args:
        predictor: Detection predictor object
        persist: Whether to keep existing tracker states

    多视角检测后处理流程：
    - 判断是否为OBB任务以及是否为流式模式
    - 检查是否已初始化追踪器
    - 解析每个视角的检测结果并过滤
    - 执行多视角目标关联（MultiViewAssociation）
    - 构造 DetectionEntry 对象以封装检测项
    - 构建并更新 frame_object_map（key 为 bbox，value 为 view_name 和原始框）
    - 构建主视角结果图像并调用 tracker 进行更新
    - 将最终结果追加至 predictor.results
    """
    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"
    logger.debug(f"Task: {'OBB' if is_obb else 'BBox'}, stream: {is_stream}")

    # 检查是否存在追踪器
    if not hasattr(predictor, 'trackers'):
        logger.warning("Trackers not initialized.")
        return

    # 检查是否有预测结果
    if not predictor.results:
        logger.warning("No results found in predictor.")
        return

    # 提取每个视角的检测框信息（并剔除 cls_id == 0 的 ashore person）
    detections_by_view = extract_detections_by_view(predictor, is_obb)
    if len(detections_by_view) != len(predictor.associator.views):
        logger.warning(
            f"Views mismatch: {len(detections_by_view)} dets vs {len(predictor.associator.views)} associator views.")

    # 多视角关联，输出格式为 [ [bbox, view_name, original_bbox], ... ]
    multi_view_detections = associate_multi_view_detections(predictor, detections_by_view)

    # 构建 DetectionEntry 列表
    entries = build_detection_entries(multi_view_detections)

    # 构建主视角图像（canvas）并封装为 Results 对象
    main_view_result = build_main_view_result(predictor, entries)

    # 更新 tracker 状态，并将最终结果写入 predictor
    update_tracker_with_entries(
        tracker=predictor.trackers[0], entries=entries, image=main_view_result.orig_img,
    )

    # 用 track.history_observations[-1] 与 entry.key() 精确匹配，填入 track_id
    fill_detection_entry_track_ids_by_history(entries, predictor.trackers[0].active_tracks)

    # 构建 frame_object_map：key = bbox_str，value = (view_name, original_bbox)
    frame_object_map = build_object_map(entries)
    predictor.object_map.update(frame_object_map)
    logger.debug(f"Updated object_map with {len(frame_object_map)} entries.")

    # 构造并写入主视角结果
    finalize_main_view_result(predictor=predictor, result=main_view_result, entries=entries, is_obb=is_obb)
    frame_index = predictor.seen - 1
    for entry in entries:
        logger.debug(f"[Entry] key={entry.key()}, track_id={entry.track_id}")
        if entry.track_id:
            g_track_manager.update(entry.track_id, entry, frame_index=frame_index)

    # 检测异常轨迹
    # TODO: 发出警告
    check_drowning_alerts(g_track_manager)


def check_drowning_alerts(track_manager) -> None:
    """
    执行溺水检测并使用结构化日志输出，仅对异常轨迹进行记录。
    """
    detector = DrowningDetector(use_original_bbox=True)
    abnormal = track_manager.detect_abnormal_tracks(detector.as_hook_fn())

    for track_id, result in abnormal.items():
        if not result or not result.get("alarm", default=False):
            continue
        logger.warning(f"[ALARM] Track {track_id} triggered drowning alert:")
        for rule in result.get("triggered_rules", []):
            rule_id = rule["rule_id"]
            rule_name = rule["rule_name"]
            value = rule["value"]
            value_str = _format_rule_value(value)
            logger.warning(f"{track_id} - [{rule_id}] {rule_name} -> {value_str}")
        # TODO: 一个广播式的异常通知接口


def _format_rule_value(value) -> str:
    """
    结构化格式化规则值（支持 float, list/tuple, 其他类型）
    """
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in value) + ")"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def extract_detections_by_view(predictor: MyDetectionPredictor, is_obb: bool) -> List[Dict[str, Any]]:
    """
    Extract bounding box detections from each view result.
    Filters out detections with class_id == 0.

    Args:
        predictor: Detection predictor object
        is_obb: Whether to extract oriented bounding boxes (OBB)

    Returns:
        List of detection dictionaries with image path and filtered boxes
    """
    detections = []
    for result in predictor.results:
        try:
            raw_boxes = result.obb.data if is_obb else result.boxes.data
            det_array = raw_boxes.cpu().numpy()

            if det_array.ndim != 2 or det_array.shape[1] < 6:
                logger.warning(f"Invalid det shape in {result.path}: {det_array.shape}")
                continue

            filtered_det = det_array[det_array[:, -1] != 0]
            detections.append({
                'image_path': result.path,
                'det': filtered_det
            })
            logger.debug(f"{result.path}: {len(filtered_det)} detections kept.")
        except Exception as e:
            logger.error(f"Error reading detection from {result.path}: {e}", exc_info=True)
    return detections


def associate_multi_view_detections(
        predictor: MyDetectionPredictor, detections: List[Dict[str, Any]],
) -> List[List[Any]]:
    """
    Perform multi-view association using the predictor's associator.

    Args:
        predictor: Detection predictor object
        detections: List of per-view detection dicts

    Returns:
        List of associated detections across views
    """
    try:
        associated = predictor.associator.forward(detections)
        logger.info(f"Associated {len(associated)} multi-view detections.")
        return associated
    except Exception as e:
        logger.error(f"Multi-view association failed: {e}", exc_info=True)
        return []


def build_detection_entries(detections: List[List[Any]]) -> List[DetectionEntry]:
    """
    构造 DetectionEntry

    Args:
        detections: List of (projected_box, view_name, original_bbox)

    Returns:
        List[DetectionEntry]
    """
    return [
        DetectionEntry(
            box_data=box_data,
            view_name=view_name,
            original_bbox=original_bbox
        )
        for box_data, view_name, original_bbox in detections
    ]


def build_object_map(entries: List[DetectionEntry]) -> Dict[str, Tuple[str, Tuple[float, float, float, float]]]:
    """
    Construct a mapping from detection box key to (view_name, original_bbox).

    Args:
        entries: List of DetectionEntry objects

    Returns:
        Object map for the current frame
    """
    object_map = {}
    for entry in entries:
        key = entry.key()
        if key in object_map:
            logger.warning(f"Duplicate entry key: {key}")
        object_map[key] = (entry.view_name, entry.original_bbox)
    return object_map


def build_main_view_result(predictor: MyDetectionPredictor, entries: List[DetectionEntry]) -> Results:
    """
    Build the main view result canvas with boxes for YOLO tracker.

    Args:
        predictor: Detection predictor object
        entries: List of DetectionEntry objects

    Returns:
        Results object with image, path, and detection tensor
    """
    det_array = np.array([entry.box_data() for entry in entries])
    if len(det_array) == 0:
        logger.info("Empty detection array. Filling with shape (0, 6).")
        det_array = np.zeros((0, 6))

    canvas = np.zeros((2 * MARGIN_HEIGHT + POOL_HEIGHT, 2 * MARGIN_WIDTH + POOL_WIDTH, 3), dtype=np.uint8)
    return Results(
        canvas,
        path=f'main_view_{predictor.seen:05d}.jpg',
        names=predictor.model.names,
        boxes=torch.as_tensor(det_array)
    )


def update_tracker_with_entries(
        tracker: Any,
        entries: List[DetectionEntry],
        image: np.ndarray,
) -> None:
    """
    使用当前帧的 DetectionEntry 更新 tracker 状态（不返回结果）
    """
    try:
        det_array = np.array([entry.box_data() for entry in entries])
        if len(det_array) > 0:
            tracks = tracker.update(det_array, image)
            if not isinstance(tracks, (list, tuple)):
                logger.warning(f"Tracker update did not return list/tuple. Got: {type(tracks)}")
            elif len(tracks) == 0:
                logger.info("Tracker update returned empty tracks.")
            else:
                logger.debug(f"{len(tracks)} tracks updated.")
        else:
            logger.debug("No detections, skipped tracker update.")
    except Exception as e:
        logger.error(f"Tracker update failed: {e}", exc_info=True)


def finalize_main_view_result(
        predictor, result, entries: List[DetectionEntry], is_obb: bool = False
) -> None:
    """
    对主视角结果进行最终封装（tensor赋值 + 写入 predictor.results）
    """
    predictor.results.append(result)

    det_array = np.array([entry.box_data() for entry in entries])
    update_tensor = torch.as_tensor(det_array)

    key = "obb" if is_obb else "boxes"
    result.update(**{key: update_tensor})


def update_tracker_and_results(
        predictor: MyDetectionPredictor, main_result: Results, entries: List[DetectionEntry], is_obb: bool,
) -> None:
    """
    Update the tracker with new detections and append results to predictor.

    Args:
        predictor: Detection predictor object
        main_result: Main Results object
        entries: List of DetectionEntry objects
        is_obb: Whether this is an OBB task
    """
    det_array = []
    try:
        det_array = np.array([entry.box_data() for entry in entries])
        if len(det_array) > 0:
            tracks = predictor.trackers[0].update(det_array, main_result.orig_img)
            logger.debug(f"{len(tracks)} tracks updated.")
        else:
            logger.debug("Skipping tracker update (no detections).")
    except Exception as e:
        logger.error(f"Tracker update failed: {e}", exc_info=True)

    predictor.results.append(main_result)
    update_args = {"obb" if is_obb else "boxes": torch.as_tensor(det_array)}
    predictor.results[-1].update(**update_args)
    logger.info("Main view result finalized and appended.")


def fill_detection_entry_track_ids_by_history(
        entries: List[DetectionEntry],
        active_tracks: List[Any],
) -> None:
    """
    用 tracker 的 history_observations[-1] 构造 key，
    填入 DetectionEntry.track_id

    Args:
        entries: 当前帧的 DetectionEntry 列表
        active_tracks: tracker.active_tracks 列表
    """
    entry_dict = {entry.key(): entry for entry in entries}
    matched, unmatched = 0, 0

    for track in active_tracks:
        if not track.history_observations:
            continue
        box = track.history_observations[-1][:4]
        key = f"{int(box[0])}_{int(box[1])}_{int(box[2])}_{int(box[3])}"
        if key in entry_dict:
            entry_dict[key].track_id = track.id
            matched += 1
        else:
            unmatched += 1

    logger.debug(f"[fill_track_id_by_history] Matched: {matched}, Unmatched: {unmatched}")


@torch.no_grad()
def run(args):
    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)

    yolo = MYYOLO(args.yolo_model)  # add default callbacks

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
        line_width=args.line_width,
        calibration=args.calibration
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))
    yolo.callbacks['on_predict_postprocess_end'].pop()
    yolo.add_callback('on_predict_postprocess_end', partial(on_predict_postprocess_end, persist=False))
    # store custom args in predictor
    yolo.predictor.custom_args = args
    # used to save the log
    save_results = {
        'video_id': args.source,
        'time_str': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'frame_data': {}
    }

    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)

    root = os.path.join('runs', yolo.predictor.args.name)
    os.makedirs(root, exist_ok=True)
    out = None
    if args.save_video:
        video_path = os.path.join(root, 'video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        fps = 10
    for idx, r in enumerate(results):
        print(idx, r)
        # if idx > 100:
        #    break
        # each view det result
        img_view1 = plot_ids(r[0], line_width=yolo.predictor.args.line_width, boxes=yolo.predictor.args.show_boxes,
                             conf=yolo.predictor.args.show_conf, labels=yolo.predictor.args.show_labels,
                             im_gpu=r[0].orig_img)
        img_view2 = plot_ids(r[1], line_width=yolo.predictor.args.line_width, boxes=yolo.predictor.args.show_boxes,
                             conf=yolo.predictor.args.show_conf, labels=yolo.predictor.args.show_labels,
                             im_gpu=r[1].orig_img)
        img_view3 = plot_ids(r[2], line_width=yolo.predictor.args.line_width, boxes=yolo.predictor.args.show_boxes,
                             conf=yolo.predictor.args.show_conf, labels=yolo.predictor.args.show_labels,
                             im_gpu=r[2].orig_img)
        img_view4 = plot_ids(r[3], line_width=yolo.predictor.args.line_width, boxes=yolo.predictor.args.show_boxes,
                             conf=yolo.predictor.args.show_conf, labels=yolo.predictor.args.show_labels,
                             im_gpu=r[3].orig_img)

        # main view det result
        main_det = plot_ids(r[4], line_width=yolo.predictor.args.line_width, boxes=yolo.predictor.args.show_boxes,
                            conf=yolo.predictor.args.show_conf, labels=yolo.predictor.args.show_labels,
                            im_gpu=r[4].orig_img)
        cv2.rectangle(main_det, (MARGIN_WIDTH, MARGIN_HEIGHT), (MARGIN_WIDTH + POOL_WIDTH, MARGIN_HEIGHT + POOL_HEIGHT),
                      (0, 255, 0), 3)  # plot the pool boundary
        main_det = cv2.resize(main_det, (main_det.shape[1], img_view1.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        cv2.putText(main_det, 'Det', (int(main_det.shape[1] / 2.0 - 150), int(MARGIN_HEIGHT / 1.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), thickness=5)

        # main view track result
        main_track = yolo.predictor.trackers[0].plot_multi_view_results(r[4].orig_img, args.show_trajectories,
                                                                        fontscale=3, thickness=5)
        cv2.rectangle(main_track, (MARGIN_WIDTH, MARGIN_HEIGHT),
                      (MARGIN_WIDTH + POOL_WIDTH, MARGIN_HEIGHT + POOL_HEIGHT), (0, 255, 0),
                      3)  # plot the pool boundary
        main_track = cv2.resize(main_track, (main_track.shape[1], img_view1.shape[0] * 2),
                                interpolation=cv2.INTER_LINEAR)
        cv2.putText(main_track, 'Track', (int(main_track.shape[1] / 2.0 - 150), int(MARGIN_HEIGHT / 1.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), thickness=5)

        # merge together
        resize_factor = 0.5
        canvas = np.hstack(
            (np.vstack((np.hstack((img_view1, img_view2)), np.hstack((img_view3, img_view4)))), main_det, main_track))
        cv2.putText(canvas, f'{idx:04d}', ((canvas.shape[1] - 600), int(MARGIN_HEIGHT / 1.5)), cv2.FONT_HERSHEY_SIMPLEX,
                    5, (255, 255, 255), thickness=5)
        canvas = cv2.resize(canvas, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)

        # pdb.set_trace()
        # swim AD rules:
        swimAD_results = yolo.predictor.trackers[0].detect_AD_v2(yolo.predictor.object_map, args.metrics)

        # remap the box to the saved video
        for obj_id in swimAD_results.keys():
            view = swimAD_results[obj_id]['view']
            x1, y1 = swimAD_results[obj_id]['bbox_left_top']
            x2, y2 = swimAD_results[obj_id]['bbox_right_bottom']
            x1, y1, x2, y2 = x1 * resize_factor, y1 * resize_factor, x2 * resize_factor, y2 * resize_factor
            new_img_width, new_img_height = r[0].orig_shape[1] * resize_factor, r[0].orig_shape[0] * resize_factor
            if view == '2':
                x1, x2 = x1 + new_img_width, x2 + new_img_width
            if view == '3':
                y1, y2 = y1 + new_img_height, y2 + new_img_height
            if view == '4':
                x1, y1, x2, y2 = x1 + new_img_width, y1 + new_img_height, x2 + new_img_width, y2 + new_img_height
            if x1 > 2560 or x2 > 2560:
                pdb.set_trace()
            if y1 > 1440 or y2 > 1440:
                pdb.set_trace()
            swimAD_results[obj_id]['bbox_left_top'] = [x1, y1]
            swimAD_results[obj_id]['bbox_right_bottom'] = [x2, y2]

        save_results['frame_data'][f"{idx:04d}"] = swimAD_results

        if args.save:
            root = os.path.join('runs', yolo.predictor.args.name)
            os.makedirs(root, exist_ok=True)
            # 改为保存为 webp 并指定质量（0-100）
            cv2.imwrite(
                os.path.join(root, f'img_{idx:05d}.webp'),
                canvas,
                [int(cv2.IMWRITE_WEBP_QUALITY), 95]
            )
            # cv2.imwrite(os.path.join(root, f'img_{idx:05d}.jpg'), canvas)
            # pdb.set_trace()
        all_images = []
        if args.save_video:
            all_images.append(canvas)
            if out is None:
                h, w = canvas.shape[:2]
                out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            out.write(canvas)
        if args.show is True:
            cv2.imshow('BoxMOT', canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break

        # loop 结束后释放
    # 推理结束时手动关闭
    track_controller.shutdown()
    if out:
        out.release()
        print(f"Finish saving video to {video_path}.")

    with open(args.log_path, "w", encoding="utf-8") as file:
        json.dump(save_results, file, ensure_ascii=False, indent=4)
        print(f'Finish saving log to {args.log_path}')

    if args.save_video is not None:
        root = os.path.join('runs', yolo.predictor.args.name)
        os.makedirs(root, exist_ok=True)
        video_path = os.path.join(root, 'video.mp4')
        print(f'saving video to {video_path}...')
        frame_size = all_images[0].shape[1], all_images[0].shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 2
        out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        for img in all_images:
            out.write(img)
        out.release()
        print(f"Finish saving video to {video_path}.")


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
    parser.add_argument('--calibration', default='boxmot/multi_view_association/region_calibration_data_v1.json',
                        help='the annotated calibration file, used for multi view association')
    parser.add_argument('--metrics', type=str, default='min_dist,max_dist,class_label',
                        help='the metrics used for AD detection in the swimming pool')
    parser.add_argument('--log-path', type=str, default='log.json',
                        help='the path to save the tracking log')

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
    opt.metrics = opt.metrics.split(',')
    track_controller = TrackManagerController(ttl=10, snapshot_interval=30, frame_index=30)
    g_track_manager = track_controller.manager
    run(opt)

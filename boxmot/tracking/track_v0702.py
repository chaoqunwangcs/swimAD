# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

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
from typing import Any, Dict, List, Union
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
from ultralytics.models import yolo
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box
from ultralytics.utils.checks import check_imgsz, check_imshow
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

from boxmot.multi_view_association.test_stream_split import MultiViewAssociationStream

import pdb

MAIN_VIEW = 'MAIN'
HEIGHT = 1440
WIDTH = 2560

POOL_WIDTH = 1500
POOL_HEIGHT = 2500
MARGIN_WIDTH = 300
MARGIN_HEIGHT = 300

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
            raise FileNotFoundError(f"No images or videos found in {p}. {FORMATS_HELP_MSG}")

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
        
        if not hasattr(self, 'associator'):
            self.setup_associator(self.args.calibration)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(imgsz=(self.dataset.num_v if self.model.pt or self.model.triton else self.dataset.bs*self.dataset.num_v, 3, *self.imgsz))
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


def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """
    Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Postprocess predictions and update with tracking
        >>> predictor = YourPredictorClass()
        >>> on_predict_postprocess_end(predictor, persist=True)
    """
    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"

    dets = []
    for i, result in enumerate(predictor.results):
        det = (result.obb.data if is_obb else result.boxes.data).cpu().numpy()
        # filter the ashore person, who's cls_id == 0
        det = det[det[:, -1] != 0]
        det = {'image_path': result.path, 'det': det}
        dets.append(det)

    # if '0015' in result.path:
    #     pdb.set_trace()
    # N * 6, x1, y1, x2, y2, conf, cls_id
    multi_view_det = predictor.associator.forward(dets)
    det = np.array(multi_view_det)
    main_results = Results(
        np.zeros((2*MARGIN_HEIGHT+POOL_HEIGHT, 2*MARGIN_WIDTH+POOL_WIDTH, 3), dtype=np.uint8),
        path='aa.jpg',
        names = predictor.model.names,
        boxes = torch.as_tensor(det)
    )

    if len(det) > 0:
        tracks = predictor.trackers[0].update(det, main_results.orig_img)
    
    predictor.results.append(main_results)
    update_args = {"obb" if is_obb else "boxes": torch.as_tensor(det)}
    predictor.results[-1].update(**update_args)
    # if len(tracks) > 0:
    #     idx = tracks[:, -1].astype(int)
    #     predictor.results.append(main_results[idx])

    #     update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
    #     predictor.results[-1].update(**update_args)

    # if len(det) == 0 or len(tracks) == 0:
    #     predictor.results.append(main_results)
    #     update_args = {"obb" if is_obb else "boxes": torch.as_tensor(det)}
    #     predictor.results[-1].update(**update_args)




@torch.no_grad()
def run(args):

    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)

    yolo = MYYOLO(args.yolo_model)    # add default callbacks
    
    
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

    if args.save_video is not None:
        all_images = []
    
    for idx, r in enumerate(results):
        if idx > 100:
            break
        # each view det result
        img_view1 = plot_ids(r[0], line_width=yolo.predictor.args.line_width,boxes=yolo.predictor.args.show_boxes,conf=yolo.predictor.args.show_conf,labels=yolo.predictor.args.show_labels,im_gpu=r[0].orig_img)
        img_view2 = plot_ids(r[1], line_width=yolo.predictor.args.line_width,boxes=yolo.predictor.args.show_boxes,conf=yolo.predictor.args.show_conf,labels=yolo.predictor.args.show_labels,im_gpu=r[1].orig_img)
        img_view3 = plot_ids(r[2], line_width=yolo.predictor.args.line_width,boxes=yolo.predictor.args.show_boxes,conf=yolo.predictor.args.show_conf,labels=yolo.predictor.args.show_labels,im_gpu=r[2].orig_img)
        img_view4 = plot_ids(r[3], line_width=yolo.predictor.args.line_width,boxes=yolo.predictor.args.show_boxes,conf=yolo.predictor.args.show_conf,labels=yolo.predictor.args.show_labels,im_gpu=r[3].orig_img)

        # main view det result
        main_det = plot_ids(r[4], line_width=yolo.predictor.args.line_width,boxes=yolo.predictor.args.show_boxes,conf=yolo.predictor.args.show_conf,labels=yolo.predictor.args.show_labels,im_gpu=r[4].orig_img)
        cv2.rectangle(main_det, (MARGIN_WIDTH, MARGIN_HEIGHT), (MARGIN_WIDTH+POOL_WIDTH,MARGIN_HEIGHT+POOL_HEIGHT),(0,255,0),3) # plot the pool boundary
        main_det = cv2.resize(main_det, (main_det.shape[1], img_view1.shape[0]*2), interpolation=cv2.INTER_LINEAR)
        cv2.putText(main_det, 'Det', (int(main_det.shape[1]/2.0-150), int(MARGIN_HEIGHT/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), thickness=5)

        # main view track result
        main_track = yolo.predictor.trackers[0].plot_multi_view_results(r[4].orig_img, args.show_trajectories, fontscale=3, thickness=5)
        cv2.rectangle(main_track, (MARGIN_WIDTH, MARGIN_HEIGHT), (MARGIN_WIDTH+POOL_WIDTH,MARGIN_HEIGHT+POOL_HEIGHT),(0,255,0),3) # plot the pool boundary
        main_track = cv2.resize(main_track, (main_track.shape[1], img_view1.shape[0]*2), interpolation=cv2.INTER_LINEAR)
        cv2.putText(main_track, 'Track', (int(main_track.shape[1]/2.0-150), int(MARGIN_HEIGHT/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), thickness=5)

        # merge together
        canvas = np.hstack((np.vstack((np.hstack((img_view1, img_view2)), np.hstack((img_view3, img_view4)))),main_det, main_track))
        cv2.putText(canvas, f'{idx:05d}', ((canvas.shape[1]-600), int(MARGIN_HEIGHT/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), thickness=5)
        canvas = cv2.resize(canvas, None, fx=0.4,fy=0.4, interpolation=cv2.INTER_LINEAR)
        
        if args.save:
            root = os.path.join('runs', yolo.predictor.args.name)
            os.makedirs(root, exist_ok=True)
            cv2.imwrite(os.path.join(root, f'img_{idx:05d}.jpg'), canvas)
            # pdb.set_trace()

        if args.save_video is not None:
            all_images.append(canvas)

        if args.show is True:
            cv2.imshow('BoxMOT', canvas)     
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break
    
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
    parser.add_argument('--calibration', default='boxmot/multi_view_association/calibration_v1.json',
                        help='the annotated calibration file, used for multi view association')

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

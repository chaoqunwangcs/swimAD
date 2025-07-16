import os
from pathlib import Path
import numpy as np
import random
import pdb

from boxmot.multi_view_association.test_stream_split import View, Box

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS
from ultralytics.utils.torch_utils import de_parallel
from torch.utils.data import DataLoader
from ultralytics.utils.checks import check_version
from ultralytics.data.augment import Compose, Format, Mosaic, RandomPerspective, CopyPaste, MixUp, RandomHSV, RandomFlip, LetterBox
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from utils import load_dataset_cache_file, get_hash

from ultralytics.models.yolo.detect.train import DetectionTrainer
from torchvision import transforms
import torch
DATASET_CACHE_VERSION = "1.0.3"

DEFAULT_MEAN = (0.37913725, 0.49507477, 0.51744888)
DEFAULT_STD = (0.24070154, 0.20103757, 0.19606746)

class Albumentations:
    """
    Albumentations transformations for image augmentation.

    This class applies various image transformations using the Albumentations library. It includes operations such as
    Blur, Median Blur, conversion to grayscale, Contrast Limited Adaptive Histogram Equalization (CLAHE), random changes
    in brightness and contrast, RandomGamma, and image quality reduction through compression.

    Attributes:
        p (float): Probability of applying the transformations.
        transform (albumentations.Compose): Composed Albumentations transforms.
        contains_spatial (bool): Indicates if the transforms include spatial operations.

    Methods:
        __call__: Applies the Albumentations transformations to the input labels.

    Examples:
        >>> transform = Albumentations(p=0.5)
        >>> augmented_labels = transform(labels)

    Notes:
        - The Albumentations package must be installed to use this class.
        - If the package is not installed or an error occurs during initialization, the transform will be set to None.
        - Spatial transforms are handled differently and require special processing for bounding boxes.
    """

    def __init__(self, p=1.0):
        """
        Initialize the Albumentations transform object for YOLO bbox formatted parameters.

        This class applies various image augmentations using the Albumentations library, including Blur, Median Blur,
        conversion to grayscale, Contrast Limited Adaptive Histogram Equalization, random changes of brightness and
        contrast, RandomGamma, and image quality reduction through compression.

        Args:
            p (float): Probability of applying the augmentations. Must be between 0 and 1.

        Attributes:
            p (float): Probability of applying the augmentations.
            transform (albumentations.Compose): Composed Albumentations transforms.
            contains_spatial (bool): Indicates if the transforms include spatial transformations.

        Raises:
            ImportError: If the Albumentations package is not installed.
            Exception: For any other errors during initialization.

        Examples:
            >>> transform = Albumentations(p=0.5)
            >>> augmented = transform(image=image, bboxes=bboxes, class_labels=classes)
            >>> augmented_image = augmented["image"]
            >>> augmented_bboxes = augmented["bboxes"]

        Notes:
            - Requires Albumentations version 1.0.3 or higher.
            - Spatial transforms are handled differently to ensure bbox compatibility.
            - Some transforms are applied with very low probability (0.01) by default.
        """
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")
        # pdb.set_trace()
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            # List of possible spatial transforms
            spatial_transforms = {
                "Affine",
                "BBoxSafeRandomCrop",
                "CenterCrop",
                "CoarseDropout",
                "Crop",
                "CropAndPad",
                "CropNonEmptyMaskIfExists",
                "D4",
                "ElasticTransform",
                "Flip",
                "GridDistortion",
                "GridDropout",
                "HorizontalFlip",
                "Lambda",
                "LongestMaxSize",
                "MaskDropout",
                "MixUp",
                "Morphological",
                "NoOp",
                "OpticalDistortion",
                "PadIfNeeded",
                "Perspective",
                "PiecewiseAffine",
                "PixelDropout",
                "RandomCrop",
                "RandomCropFromBorders",
                "RandomGridShuffle",
                "RandomResizedCrop",
                "RandomRotate90",
                "RandomScale",
                "RandomSizedBBoxSafeCrop",
                "RandomSizedCrop",
                "Resize",
                "Rotate",
                "SafeRotate",
                "ShiftScaleRotate",
                "SmallestMaxSize",
                "Transpose",
                "VerticalFlip",
                "XYMasking",
            }  # from https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms
            # Transforms
            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.5),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.ImageCompression(quality_lower=75, quality_upper=100, p=0.01),
            ]

            # Compose transforms
            self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
            self.transform = (
                A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
                if self.contains_spatial
                else A.Compose(T)
            )
            if hasattr(self.transform, "set_random_seed"):
                # Required for deterministic transforms in albumentations>=1.4.21
                self.transform.set_random_seed(torch.initial_seed())
            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels):
        """
        Applies Albumentations transformations to input labels.

        This method applies a series of image augmentations using the Albumentations library. It can perform both
        spatial and non-spatial transformations on the input image and its corresponding labels.

        Args:
            labels (dict): A dictionary containing image data and annotations. Expected keys are:
                - 'img': numpy.ndarray representing the image
                - 'cls': numpy.ndarray of class labels
                - 'instances': object containing bounding boxes and other instance information

        Returns:
            (dict): The input dictionary with augmented image and updated annotations.

        Examples:
            >>> transform = Albumentations(p=0.5)
            >>> labels = {
            ...     "img": np.random.rand(640, 640, 3),
            ...     "cls": np.array([0, 1]),
            ...     "instances": Instances(bboxes=np.array([[0, 0, 1, 1], [0.5, 0.5, 0.8, 0.8]])),
            ... }
            >>> augmented = transform(labels)
            >>> assert augmented["img"].shape == (640, 640, 3)

        Notes:
            - The method applies transformations with probability self.p.
            - Spatial transforms update bounding boxes, while non-spatial transforms only modify the image.
            - Requires the Albumentations library to be installed.
        """
        # pdb.set_trace()
        if self.transform is None or random.random() > self.p:
            return labels

        if self.contains_spatial:
            cls = labels["cls"]
            if len(cls):
                im = labels["img"]
                labels["instances"].convert_bbox("xywh")
                labels["instances"].normalize(*im.shape[:2][::-1])
                bboxes = labels["instances"].bboxes
                # TODO: add supports of segments and keypoints
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # transformed
                if len(new["class_labels"]) > 0:  # skip update if no bbox in new im
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"])
                    bboxes = np.array(new["bboxes"], dtype=np.float32)
                labels["instances"].update(bboxes=bboxes)
        else:
            labels["img"] = self.transform(image=labels["img"])["image"]  # transformed

        return labels

class Normalization:
    def __init__(self):
        self.mean = DEFAULT_MEAN
        self.std = DEFAULT_STD
        self.normalize_transform = Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(self.mean), std=torch.tensor(self.std))
        ])
    
    def __call__(self, labels):
        # pdb.set_trace()
        labels['img'] = np.array(self.normalize_transform(labels['img']).permute(1,2,0)) * 255
        return labels
        

def v8_transforms(dataset, imgsz, hyp, stretch=False):
    """
    Applies a series of image transformations for training.

    This function creates a composition of image augmentation techniques to prepare images for YOLO training.
    It includes operations such as mosaic, copy-paste, random perspective, mixup, and various color adjustments.

    Args:
        dataset (Dataset): The dataset object containing image data and annotations.
        imgsz (int): The target image size for resizing.
        hyp (Namespace): A dictionary of hyperparameters controlling various aspects of the transformations.
        stretch (bool): If True, applies stretching to the image. If False, uses LetterBox resizing.

    Returns:
        (Compose): A composition of image transformations to be applied to the dataset.

    Examples:
        >>> from ultralytics.data.dataset import YOLODataset
        >>> from ultralytics.utils import IterableSimpleNamespace
        >>> dataset = YOLODataset(img_path="path/to/images", imgsz=640)
        >>> hyp = IterableSimpleNamespace(mosaic=1.0, copy_paste=0.5, degrees=10.0, translate=0.2, scale=0.9)
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
        >>> augmented_data = transforms(dataset[0])
    """
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
    )

    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")
    # import pdb; pdb.set_trace()
    import torchvision.transforms as T 
    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
            # Normalization(),
        ]
    )  # transforms


class CustomDataset(YOLODataset):
    def get_img_files(self, img_path):
        # for swimAD data
        self.img2label_dict = dict()
        try:
            f = []
            if isinstance(img_path, str):
                img_path = [img_path]
            for path in img_path:
                p = Path(path)
                assert p.is_file()    # save the img_path and corresponding label path # absolute path
                with open(p, encoding="utf-8") as t:
                    items = t.read().strip().splitlines()
                for item in items:
                    try:
                        file_path, anno_path = item.split(',')
                    except:
                        pdb.set_trace()
                    file_path = file_path.strip()
                    anno_path = anno_path.strip()
                    if not (Path(file_path).is_file() and Path(anno_path).is_file()):
                        continue
                    # assert Path(file_path).is_file() and Path(anno_path).is_file()
                    f.append(file_path)
                    self.img2label_dict[file_path] = anno_path

            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
        return im_files

    def get_labels(self):
        """
        Returns dictionary of labels for YOLO training.

        This method loads labels from disk or cache, verifies their integrity, and prepares them for training.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        # pdb.set_trace()
        self.label_files = [self.img2label_dict[x] for x in self.im_files]
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """
        Builds and appends transforms to the list.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms.
        """
        # import pdb; pdb.set_trace()
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms


def build_custom_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    dataset = CustomDataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )

class CustomTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_custom_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

class CustomValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, region='all'):
        """
        Initialize detection validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (Any, optional): Progress bar for displaying progress.
            args (dict, optional): Arguments for the validator.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.1, 0.95, 18)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        self.region = region
        if self.args.save_hybrid and self.args.task == "detect":
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )
        self.init_views()

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 10) % ("Class", "Images", "Instances", "Box(P", "R", "mAP10", "mAP20", "mAP30", "mAP40", "mAP50", "mAP50-95)")

    def print_results(self):
        """Print training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * (len(self.metrics.keys) + 4)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *[self.metrics.box.mp, self.metrics.box.mr, self.metrics.box.all_ap[:,0].mean(), self.metrics.box.all_ap[:,2].mean(), self.metrics.box.all_ap[:,4].mean(), self.metrics.box.all_ap[:,6].mean(), self.metrics.box.all_ap[:,8].mean(), self.metrics.box.all_ap[:,8:].mean()]))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *[self.metrics.box.p[i], self.metrics.box.r[i],  self.metrics.box.all_ap[:,0][i], self.metrics.box.all_ap[:,2][i], self.metrics.box.all_ap[:,4][i], self.metrics.box.all_ap[:,6][i], self.metrics.box.all_ap[:,8][i], self.metrics.box.all_ap[:,8:].mean(1)[i]])
                )

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def init_views(self):
        view1 = View('1', p1=0, p2=0, k1=-0.5353, k2=0.2875, k3=-0.0906, fx=1621.9, fy=1856.1, cx=1116.3, cy=742.9178, fx_ratio=1.33, fy_ratio=1.33, grid_info=None)
        view2 = View('2', p1=0, p2=0, k1=-0.5153, k2=0.2845, k3=-0.0906, fx=1621.9, fy=1856.1, cx=1116.3, cy=742.9178, fx_ratio=1.34, fy_ratio=1.34, grid_info=None)
        view3 = View('3', p1=0, p2=0, k1=-0.5253, k2=0.2845, k3=-0.0906, fx=1621.9, fy=1856.1, cx=1116.3, cy=742.9178, fx_ratio=1.34, fy_ratio=1.34, grid_info=None)
        view4 = View('4', p1=0, p2=0, k1=-0.5253, k2=0.2875, k3=-0.0906, fx=1621.9, fy=1856.1, cx=1116.3, cy=742.9178, fx_ratio=1.34, fy_ratio=1.34, grid_info=None)
        self.views = {'1': view1, '2': view2, '3': view3, '4': view4}

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_custom_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def update_metrics(self, preds, batch):
        """
        Update metrics with new predictions and ground truth.

        Args:
            preds (List[torch.Tensor]): List of predictions from the model.
            batch (dict): Batch data containing ground truth.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)

            # pdb.set_trace()
            if self.region != 'all':
                ## filter the object out of region of the ground-truth
                view_name = str(batch['im_file'][si].split('/')[-2])
                clses, bboxes = pbatch.pop("cls"), pbatch.pop("bbox")
                device = bboxes.device
                clses, bboxes = clses.cpu().numpy(), bboxes.cpu().numpy()
                cls_gt, bbox_gt = [], []

                for obj_id in range(bboxes.shape[0]):
                    x1, y1, x2, y2 = bboxes[obj_id]
                    c = clses[obj_id]
                    if Box(x1, y1, x2, y2, 1.0, c, self.views[view_name], self.views[view_name], None, is_distorted=True).is_keep() == (self.region == 'region'):
                        bbox_gt.append([x1, y1, x2, y2])
                        cls_gt.append(c)

                if len(bbox_gt) == 0:
                    bbox_gt = np.zeros((0, 4))
                else:
                    bbox_gt = np.array(bbox_gt)
                bbox_gt = torch.as_tensor(bbox_gt).to(device)
                cls_gt = torch.as_tensor(np.array(cls_gt)).to(device)
                pbatch['cls'], pbatch['bbox'] = cls_gt, bbox_gt
                npr = len(bbox_gt)
                ## filter the object out of region of the ground-truth

            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)

            if self.region != 'all':
                ## filter the object out of region of the prediction
                bboxes = predn
                device = predn.device
                bboxes = bboxes.cpu().numpy()
                bbox_pred = []

                for obj_id in range(bboxes.shape[0]):
                    x1, y1, x2, y2, conf, c = bboxes[obj_id]
                    if Box(x1, y1, x2, y2, 1.0, c, self.views[view_name], self.views[view_name], None, is_distorted=True).is_keep() == (self.region == 'region'):
                        bbox_pred.append([x1, y1, x2, y2, conf, c])

                if len(bbox_pred) == 0:
                    bbox_pred = np.zeros((0, 6))
                else:
                    bbox_pred = np.array(bbox_pred)

                predn = torch.as_tensor(bbox_pred).to(device)
                ## filter the object out of region of the prediction
            # pdb.set_trace()
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]
            
            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )



import os
from pathlib import Path
import pdb

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS
from ultralytics.utils.torch_utils import de_parallel
from torch.utils.data import DataLoader
from utils import load_dataset_cache_file, get_hash

from ultralytics.models.yolo.detect.train import DetectionTrainer
DATASET_CACHE_VERSION = "1.0.3"


class CustomDataset(YOLODataset):
    def get_img_files(self, img_path):
        # for swimAD data
        self.img2label_dict = dict()
        try:
            f = []
            p = Path(img_path)
            assert p.is_file()    # save the img_path and corresponding label path # absolute path
            with open(p, encoding="utf-8") as t:
                items = t.read().strip().splitlines()
            for item in items:
                file_path, anno_path = item.split(',')
                file_path = file_path.strip()
                anno_path = anno_path.strip()
                assert Path(file_path).is_file() and Path(anno_path).is_file()
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





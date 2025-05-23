# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm
import configparser
import shutil
import json
import queue
import select
import re
import os
import torch
from functools import partial
import threading
import sys
import copy
import concurrent.futures

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS, logger as LOGGER, EXAMPLES, DATA
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.torch_utils import select_device
from boxmot.utils.misc import increment_path
from boxmot.postprocessing.gsi import gsi

from ultralytics import YOLO
from ultralytics.data.loaders import LoadImagesAndVideos

from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)
from tracking.utils import convert_to_mot_format, write_mot_results, download_mot_eval_tools, download_mot_dataset, unzip_mot_dataset, eval_setup, split_dataset
from boxmot.appearance.reid.auto_backend import ReidAutoBackend

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install


def cleanup_mot17(data_dir, keep_detection='FRCNN'):
    """
    Cleans up the MOT17 dataset to resemble the MOT16 format by keeping only one detection folder per sequence.
    Skips sequences that have already been cleaned.

    Args:
    - data_dir (str): Path to the MOT17 train directory.
    - keep_detection (str): Detection type to keep (options: 'DPM', 'FRCNN', 'SDP'). Default is 'DPM'.
    """

    # Get all folders in the train directory
    all_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # Identify unique sequences by removing detection suffixes
    unique_sequences = set(seq.split('-')[0] + '-' + seq.split('-')[1] for seq in all_dirs)

    for seq in unique_sequences:
        # Directory path to the cleaned sequence
        cleaned_seq_dir = os.path.join(data_dir, seq)

        # Skip if the sequence is already cleaned
        if os.path.exists(cleaned_seq_dir):
            print(f"Sequence {seq} is already cleaned. Skipping.")
            continue

        # Directories for each detection method
        seq_dirs = [os.path.join(data_dir, d)
                    for d in all_dirs if d.startswith(seq)]

        # Directory path for the detection folder to keep
        keep_dir = os.path.join(data_dir, f"{seq}-{keep_detection}")

        if os.path.exists(keep_dir):
            # Move the directory to a new name (removing the detection suffix)
            shutil.move(keep_dir, cleaned_seq_dir)
            print(f"Moved {keep_dir} to {cleaned_seq_dir}")

            # Remove other detection directories
            for seq_dir in seq_dirs:
                if os.path.exists(seq_dir) and seq_dir != keep_dir:
                    shutil.rmtree(seq_dir)
                    print(f"Removed {seq_dir}")
        else:
            print(f"Directory for {seq} with {keep_detection} detection does not exist. Skipping.")

    print("MOT17 Cleanup completed!")


def prompt_overwrite(path_type: str, path: str, ci: bool = True) -> bool:
    """
    Prompts the user to confirm overwriting an existing file.

    Args:
        path_type (str): Type of the path (e.g., 'Detections and Embeddings', 'MOT Result').
        path (str): The path to check.
        ci (bool): If True, automatically reuse existing file without prompting (for CI environments).

    Returns:
        bool: True if user confirms to overwrite, False otherwise.
    """
    if ci:
        LOGGER.debug(f"{path_type} {path} already exists. Use existing due to no UI mode.")
        return False

    def input_with_timeout(prompt, timeout=20.0):
        print(prompt, end='', flush=True)

        result = []
        input_received = threading.Event()

        def get_input():
            user_input = sys.stdin.readline().strip().lower()
            result.append(user_input)
            input_received.set()

        input_thread = threading.Thread(target=get_input)
        input_thread.daemon = True  # Ensure thread does not prevent program exit
        input_thread.start()
        input_thread.join(timeout)

        if input_received.is_set():
            return result[0] in ['y', 'yes']
        else:
            print("\nNo response, not proceeding with overwrite...")
            return False

    return input_with_timeout(f"{path_type} {path} already exists. Overwrite? [y/N]: ")


def generate_dets_embs(args: argparse.Namespace, y: Path, source: Path, sequence_name_for_files: str) -> None:
    """
    Generates detections and embeddings for the specified 
    arguments, YOLO model and source.

    Args:
        args (Namespace): Parsed command line arguments.
        y (Path): Path to the YOLO model file.
        source (Path): Path to the source directory.
    """
    WEIGHTS.mkdir(parents=True, exist_ok=True)

    if args.imgsz is None:
        args.imgsz = default_imgsz(y)

    yolo = YOLO(
        y if is_ultralytics_model(y)
        else 'yolov8n.pt',
    )

    results = yolo(
        source=source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        stream=True,
        device=args.device,
        verbose=False,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
    )

    if not is_ultralytics_model(y):
        m = get_yolo_inferer(y)
        yolo_model = m(model=y, device=yolo.predictor.device,
                       args=yolo.predictor.args)
        yolo.predictor.model = yolo_model

        # If current model is YOLOX, change the preprocess and postprocess
        if is_yolox_model(y):
            # add callback to save image paths for further processing
            yolo.add_callback("on_predict_batch_start",
                              lambda p: yolo_model.update_im_paths(p))
            yolo.predictor.preprocess = (
                lambda im: yolo_model.preprocess(im=im))
            yolo.predictor.postprocess = (
                lambda preds, im, im0s:
                yolo_model.postprocess(preds=preds, im=im, im0s=im0s))

    reids = []
    for r in args.reid_model:
        reid_model = ReidAutoBackend(weights=r,
                                     device=yolo.predictor.device,
                                     half=args.half).model
        reids.append(reid_model)
        embs_path = args.project / 'dets_n_embs' / y.stem / 'embs' / r.stem / (sequence_name_for_files + '.txt')
        embs_path.parent.mkdir(parents=True, exist_ok=True)
        embs_path.touch(exist_ok=True)

        if os.path.getsize(embs_path) > 0:
            if prompt_overwrite('Embeddings file', embs_path, args.ci):
                open(embs_path, 'w').close()
            else:
                LOGGER.debug(f"Skipping overwrite for existing embeddings file {embs_path}")
        else:
            embs_path.touch(exist_ok=True)

    yolo.predictor.custom_args = args

    dets_path = args.project / 'dets_n_embs' / y.stem / 'dets' / (sequence_name_for_files + '.txt')
    dets_path.parent.mkdir(parents=True, exist_ok=True)
    dets_path.touch(exist_ok=True)

    if os.path.getsize(dets_path) > 0:
        if prompt_overwrite('Detections file', dets_path, args.ci):
            open(dets_path, 'w').close()
        else:
            LOGGER.debug(f"Skipping overwrite for existing detections file {dets_path}")
    else:
        dets_path.touch(exist_ok=True)

    # --- Start: Ensure files are cleared and headers written before loop ---
    # Clear/create detections file and write header
    with open(str(dets_path), 'wb') as f: # Use 'wb' to clear and write bytes
        # Simpler header: just the source path, use os.linesep for consistency
        header_line = f"# {str(source)}{os.linesep}".encode('utf-8')
        f.write(header_line)

    # Clear/create embeddings file and write header
    for reid_config_path in args.reid_model: # Iterate through configured reid models
        # Construct embs_path using sequence_name_for_files and current reid model's stem
        current_embs_path = args.project / 'dets_n_embs' / y.stem / 'embs' / reid_config_path.stem / (sequence_name_for_files + '.txt')
        current_embs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(current_embs_path), 'wb') as f: # Use 'wb' to clear and write bytes
            # Simpler header: just the source path and reid model, use os.linesep
            header_line = f"# Reid: {reid_config_path.stem}, Source: {str(source)}{os.linesep}".encode('utf-8')
            f.write(header_line)
    # --- End: Ensure files are cleared and headers written before loop ---

    for frame_idx, r in enumerate(tqdm(results, desc=f"Processing {sequence_name_for_files}")):
        nr_dets = len(r.boxes)
        # frame_idx here is 0-based from enumerate, results from yolo might have different frame indexing if it's from a video
        # The original code used frame_idx + 1 directly for MOT format, which is typical (1-based frames)
        # For dets file, it also used frame_idx + 1.
        # Let's ensure yolo.stream=True gives per-frame results, and frame_idx is the simple counter.
        # The problem is if `results` itself contains frame numbers, this enumerate `frame_idx` might be misleading.
        # However, yolo() with stream=True typically yields results per image/frame processed.
        # Let's assume this frame_idx from enumerate is fine for now for the .txt file.

        # Ensure current_frame_num_for_dets is 1-based for the detection file
        current_frame_num_for_dets = torch.full((1, 1), frame_idx + 1).repeat(nr_dets, 1)
        img = r.orig_img

        dets = np.concatenate(
            [
                current_frame_num_for_dets, # Use 1-based frame index for saving
                r.boxes.xyxy.to('cpu'),
                r.boxes.conf.unsqueeze(1).to('cpu'),
                r.boxes.cls.unsqueeze(1).to('cpu'),
            ], axis=1
        )

        # Filter dets with incorrect boxes: (x2 < x1 or y2 < y1)
        boxes = r.boxes.xyxy.to('cpu').numpy().round().astype(int)
        boxes_filter = ((np.maximum(0, boxes[:, 0]) < np.minimum(boxes[:, 2], img.shape[1])) &
                        (np.maximum(0, boxes[:, 1]) < np.minimum(boxes[:, 3], img.shape[0])))
        dets = dets[boxes_filter]

        # Skip frame if no valid detections after filtering
        if dets.shape[0] == 0:
            continue

        with open(str(dets_path), 'ab+') as f:
            np.savetxt(f, dets, fmt='%f')

        # Generate and save embeddings only if detections exist for the frame
        for reid, reid_model_name in zip(reids, args.reid_model):
            # Ensure embeddings are calculated for the same filtered detections
            embs = reid.get_features(dets[:, 1:5], img)

            # Check consistency within the frame (optional but good practice)
            if embs.shape[0] != dets.shape[0]:
                 LOGGER.warning(f"Frame {frame_idx + 1}: Mismatch between detections ({dets.shape[0]}) and generated embeddings ({embs.shape[0]}) for {reid_model_name.stem}. Skipping embedding save for this frame.")
                 continue # Skip saving embeddings for this specific Reid model/frame

            # Construct embs_path using sequence_name_for_files for saving this specific embedding
            current_embs_path_for_saving = args.project / "dets_n_embs" / y.stem / 'embs' / reid_model_name.stem / (sequence_name_for_files + '.txt')
            with open(str(current_embs_path_for_saving), 'ab+') as f:
                np.savetxt(f, embs, fmt='%f')


def generate_mot_results(args: argparse.Namespace, config_dict: dict = None) -> dict[str, np.ndarray]:
    """
    Generates MOT results for the specified arguments and configuration.

    Args:
        args (Namespace): Parsed command line arguments.
        config_dict (dict, optional): Additional configuration dictionary.

    Returns:
        dict[str, np.ndarray]: {seq_name: array} with frame ids used for MOT
    """
    args.device = select_device(args.device)
    tracker = create_tracker(
        args.tracking_method,
        TRACKER_CONFIGS / (args.tracking_method + '.yaml'),
        args.reid_model[0].with_suffix('.pt'),
        args.device,
        False,
        False,
        config_dict
    )

    # Add encoding='utf-8' for reading the header line safely
    with open(args.dets_file_path, 'r', encoding='utf-8') as file:
        first_line = file.readline()
        # Use regex to extract path after #, allowing for optional space
        match = re.match(r"^#\s*(.*)", first_line)
        if match:
            source_str = match.group(1).strip() # Get captured group and strip whitespace
            # Change to INFO level to ensure visibility
            # LOGGER.info(f"Validating parsed header path: '{source_str}'") # <-- Remove this, seems ineffective in threads
            temp_source_path = Path(source_str)
            if not temp_source_path.is_dir(): # Check if the extracted path is a directory
                # Modify the error message to include the problematic path
                error_msg = f"Parsed header path '{source_str}' from {args.dets_file_path} is not a valid directory."
                LOGGER.error(error_msg)
                raise ValueError(error_msg) # Raise error with the path included
            # -- End: Add validation --
            # Path is validated, now assign it to the main source variable
            source = temp_source_path
        else:
            # Handle case where header is missing or malformed
            LOGGER.error(f"Could not parse header line in {args.dets_file_path}. Expected '# path\\n', got: {first_line.strip()}")
            raise ValueError(f"Invalid header format in {args.dets_file_path}")

    # Specify encoding='utf-8' for loadtxt to handle potential UTF-8 headers/comments
    dets = np.loadtxt(args.dets_file_path, skiprows=1, encoding='utf-8')
    # Add encoding here too for robustness, although embs file might not have header
    # The embs_file_path should correspond to the specific reid model used for this run.
    # args.reid_model[0] was used to construct the embs_folder in run_generate_mot_results.
    # So args.embs_file_path is specific to that one reid model.
    embs = np.loadtxt(args.embs_file_path, encoding='utf-8', skiprows=1)

    # Add check for row count consistency
    if dets.shape[0] != embs.shape[0]:
        raise ValueError(
            f"Row count mismatch between detections ({dets.shape[0]}) and embeddings ({embs.shape[0]})\n"
            f"Dets file: {args.dets_file_path}\n"
            f"Embs file: {args.embs_file_path}\n"
            f"This indicates an issue during the detection/embedding generation process."
        )

    dets_n_embs = np.concatenate([dets, embs], axis=1)

    # 处理图片直接在序列目录下的情况
    if (source / 'img1').is_dir():
        # 传统结构：图片在img1子目录
        img_dir = source / 'img1' # This 'source' is from dets header, e.g. .../mysequence
    else:
        # 新结构：图片直接在序列目录下
        img_dir = source
        
    dataset = LoadImagesAndVideos(img_dir)

    # MOT result txt_path should use the actual sequence name, which is source.name
    txt_path = args.exp_folder_path / (source.name + '.txt')
    all_mot_results = []

    # Change FPS
    if args.fps:

        # Extract original FPS
        # 判断是否使用自定义gt文件夹来读取seqinfo.ini
        if hasattr(args, 'custom_gt_folder') and args.custom_gt_folder:
            seq_dir = Path(args.custom_gt_folder) / source.parent.name
            conf_path = seq_dir / 'seqinfo.ini'
        else:
            conf_path = source.parent / 'seqinfo.ini'
            
        # 如果source就是序列根目录（图片直接放在序列目录下的情况），则直接在source下寻找seqinfo.ini
        if not conf_path.exists() and source.is_dir():
            conf_path = source / 'seqinfo.ini'
            
        conf = configparser.ConfigParser()
        conf.read(conf_path)

        orig_fps = int(conf.get("Sequence", "frameRate"))
    
        if orig_fps < args.fps:
            LOGGER.warning(f"Original FPS ({orig_fps}) is lower than "
                           f"requested FPS ({args.fps}) for sequence "
                           f"{source.parent.name}. Using original FPS.")
            target_fps = orig_fps
        else:
            target_fps = args.fps

        
        step = orig_fps/target_fps
    else:
        step = 1
    
    # Create list with frame numbers according to needed step
    frame_nums = np.arange(1, len(dataset) + 1, step).astype(int).tolist()

    # seq_frame_nums key should be the actual sequence name (source.name)
    seq_frame_nums = {source.name: frame_nums.copy()}

    for frame_num, d in enumerate(tqdm(dataset, desc=source.name), 1):
        # Filter using list with needed numbers
        if len(frame_nums) > 0:
            if frame_num < frame_nums[0]:
                continue
            else:
                frame_nums.pop(0)

        im = d[1][0]
        frame_dets_n_embs = dets_n_embs[dets_n_embs[:, 0] == frame_num]

        dets = frame_dets_n_embs[:, 1:7]
        embs = frame_dets_n_embs[:, 7:]
        tracks = tracker.update(dets, im, embs)

        if tracks.size > 0:
            mot_results = convert_to_mot_format(tracks, frame_num)
            all_mot_results.append(mot_results)

    if all_mot_results:
        all_mot_results = np.vstack(all_mot_results)
    else:
        all_mot_results = np.empty((0, 0))

    write_mot_results(txt_path, all_mot_results)

    return seq_frame_nums


def parse_mot_results(results: str) -> dict:
    """
    Extracts the COMBINED HOTA, MOTA, IDF1 from the results generated by the run_mot_challenge.py script.

    Args:
        results (str): MOT results as a string.

    Returns:
        dict: A dictionary containing HOTA, MOTA, and IDF1 scores.
    """
    combined_results = results.split('COMBINED')[2:-1]
    combined_results = [float(re.findall(r"[-+]?(?:\d*\.*\d+)", f)[0])
                        for f in combined_results]

    results_dict = {}
    for key, value in zip(["HOTA", "MOTA", "IDF1"], combined_results):
        results_dict[key] = value

    return results_dict


def trackeval(args: argparse.Namespace, seq_paths: list, save_dir: Path, MOT_results_folder: Path, gt_folder: Path, metrics: list = ["HOTA", "CLEAR", "Identity"]) -> str:
    """
    Executes a Python script to evaluate MOT challenge tracking results using specified metrics.

    Args:
        seq_paths (list): List of sequence paths.
        save_dir (Path): Directory to save evaluation results.
        MOT_results_folder (Path): Folder containing MOT results.
        gt_folder (Path): Folder containing ground truth data.
        metrics (list, optional): List of metrics to use for evaluation. Defaults to ["HOTA", "CLEAR", "Identity"].

    Returns:
        str: Standard output from the evaluation script.
    """

    d = [seq_path.name for seq_path in seq_paths]

    # 使用自定义的gt文件夹路径（如果指定）
    if hasattr(args, 'custom_gt_folder') and args.custom_gt_folder:
        actual_gt_folder = Path(args.custom_gt_folder)
        LOGGER.info(f"使用自定义ground truth文件夹进行评估: {actual_gt_folder}")
    else:
        actual_gt_folder = gt_folder

    # Build the arguments list for run_mot_challenge.py
    trackeval_cmd_args = [
        sys.executable, EXAMPLES / 'val_utils' / 'scripts' / 'run_mot_challenge.py',
        "--GT_FOLDER", str(actual_gt_folder),
        "--BENCHMARK", "", # Set benchmark to empty since we use custom GT folder
        "--TRACKERS_FOLDER", args.exp_folder_path, # Folder containing the tracker output .txt files
        "--TRACKERS_TO_EVAL", "", # Evaluate the specific tracker folder created
        "--SPLIT_TO_EVAL", "train", # This might need adjustment depending on GT structure, but often needed
        "--METRICS", *metrics,
        "--USE_PARALLEL", "True",
        "--TRACKER_SUB_FOLDER", "", # Tracker results are directly in TRACKERS_FOLDER
        "--NUM_PARALLEL_CORES", str(4),
        "--SKIP_SPLIT_FOL", "True", # Important if GT data is directly in GT_FOLDER/seq_name/gt
        "--GT_LOC_FORMAT", "{gt_folder}/{seq}/normal_seq.txt", # Path structure for GT files
        # Disable default preprocessing (class filtering)
        "--DO_PREPROC", "False"
    ]

    # Add class filtering argument if specified
    if args.eval_classes is not None:
        # Convert class IDs to strings for command line
        class_ids_str = [str(c) for c in args.eval_classes]
        trackeval_cmd_args.extend(["--CLASSES_TO_EVAL", *class_ids_str])

    # Add sequence info last
    trackeval_cmd_args.extend(["--SEQ_INFO", *d])

    # Print the command being executed for debugging
    # Convert all args to strings before joining for logging
    cmd_str_for_log = ' '.join(map(str, trackeval_cmd_args))
    LOGGER.debug(f"Running TrackEval command: {cmd_str_for_log}")

    p = subprocess.Popen(
        args=trackeval_cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = p.communicate()

    if stderr:
        print("Standard Error:\n", stderr)
    return stdout


def run_generate_dets_embs(opt: argparse.Namespace) -> None:
    """
    Runs the generate_dets_embs function for all YOLO models and source directories.

    Args:
        opt (Namespace): Parsed command line arguments.
    """
    mot_folder_paths = sorted([item for item in Path(opt.source).iterdir() if item.is_dir()])
    for y in opt.yolo_model:
        for i, mot_folder_path in enumerate(mot_folder_paths):
            # Determine the actual source path for images (either mot_folder_path or mot_folder_path / 'img1')
            current_sequence_source_path = mot_folder_path
            if (mot_folder_path / 'img1').is_dir():
                current_sequence_source_path = mot_folder_path / 'img1'
            elif not (any(mot_folder_path.glob('*.jpg')) or any(mot_folder_path.glob('*.png'))):
                LOGGER.warning(f'未在{mot_folder_path}找到图片或img1子目录，跳过')
                continue

            # Use current_sequence_source_path.name (which is the sequence name like 'mysequence') for file naming
            sequence_name = mot_folder_path.name # mot_folder_path itself is the sequence directory, e.g., .../test2/mysequence
            
            dets_path = Path(opt.project) / 'dets_n_embs' / y.stem / 'dets' / (sequence_name + '.txt')
            embs_path = Path(opt.project) / 'dets_n_embs' / y.stem / 'embs' / (opt.reid_model[0].stem) / (sequence_name + '.txt')
            
            if dets_path.exists() and embs_path.exists():
                if prompt_overwrite('Detections and Embeddings', dets_path, opt.ci):
                    LOGGER.debug(f'Overwriting detections and embeddings for {mot_folder_path}...')
                else:
                    LOGGER.debug(f'Skipping generation for {mot_folder_path} as they already exist.')
                    continue
            LOGGER.debug(f'Generating detections and embeddings for data under {mot_folder_path} [{i + 1}/{len(mot_folder_paths)} seqs]')
            
            generate_dets_embs(opt, y, source=current_sequence_source_path, sequence_name_for_files=sequence_name)


def process_single_mot(opt: argparse.Namespace, d: Path, e: Path, evolve_config: dict):
    # Create a deep copy of opt so each task works independently
    new_opt = copy.deepcopy(opt)
    new_opt.dets_file_path = d
    new_opt.embs_file_path = e
    frames_dict = generate_mot_results(new_opt, evolve_config)
    return frames_dict

def run_generate_mot_results(opt: argparse.Namespace, evolve_config: dict = None) -> None:
    """
    Runs the generate_mot_results function for all YOLO models and detection/embedding files
    in parallel.
    """
    
    for y in opt.yolo_model:
        exp_folder_path = opt.project / 'mot' / (f"{y.stem}_{opt.reid_model[0].stem}_{opt.tracking_method}")
        exp_folder_path = increment_path(path=exp_folder_path, sep="_", exist_ok=False)
        opt.exp_folder_path = exp_folder_path

        # Ensure the experiment folder exists before any operations
        opt.exp_folder_path.mkdir(parents=True, exist_ok=True)

        mot_folder_names = [item.stem for item in Path(opt.source).iterdir()]
        
        dets_folder = opt.project / "dets_n_embs" / y.stem / 'dets'
        embs_folder = opt.project / "dets_n_embs" / y.stem / 'embs' / opt.reid_model[0].stem
        
        dets_file_paths = sorted([
            item for item in dets_folder.glob('*.txt')
            if not item.name.startswith('.') and item.stem in mot_folder_names
        ])
        embs_file_paths = sorted([
            item for item in embs_folder.glob('*.txt')
            if not item.name.startswith('.') and item.stem in mot_folder_names
        ])
        
        LOGGER.info(f"\nStarting tracking on:\n\t{opt.source}\nwith preloaded dets\n\t({dets_folder.relative_to(ROOT)})\nand embs\n\t({embs_folder.relative_to(ROOT)})\nusing\n\t{opt.tracking_method}")
        if hasattr(opt, 'custom_gt_folder') and opt.custom_gt_folder:
            LOGGER.info(f"将使用自定义ground truth文件夹: {opt.custom_gt_folder}")

        tasks = []
        # Create a thread pool to run each file pair in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for d, e in zip(dets_file_paths, embs_file_paths):
                mot_result_path = exp_folder_path / (d.stem + '.txt')
                if mot_result_path.exists():
                    if prompt_overwrite('MOT Result', mot_result_path, opt.ci):
                        LOGGER.info(f'Overwriting MOT result for {d.stem}...')
                    else:
                        LOGGER.info(f'Skipping MOT result generation for {d.stem} as it already exists.')
                        continue
                # Submit the task to process this file pair in parallel
                tasks.append(executor.submit(process_single_mot, opt, d, e, evolve_config))
            
            # Dict with {seq_name: [frame_nums]}
            seqs_frame_nums = {}
            # Wait for all tasks to complete and log any exceptions
            for future in concurrent.futures.as_completed(tasks):
                try:
                    seqs_frame_nums.update(future.result())
                except Exception as exc:
                    LOGGER.error(f'Error processing file pair: {exc}')
    
    # Postprocess data with gsi if requested
    if opt.gsi:
        gsi(mot_results_folder=opt.exp_folder_path)

    with open(opt.exp_folder_path / 'seqs_frame_nums.json', 'w') as f:
        json.dump(seqs_frame_nums, f)


def run_trackeval(opt: argparse.Namespace) -> dict:
    """
    Runs the trackeval function to evaluate tracking results.

    Args:
        opt (Namespace): Parsed command line arguments.
    """
    seq_paths, save_dir, MOT_results_folder, gt_folder = eval_setup(opt, opt.val_tools_path)
    trackeval_results = trackeval(opt, seq_paths, save_dir, MOT_results_folder, gt_folder)
    hota_mota_idf1 = parse_mot_results(trackeval_results)
    if opt.verbose:
        LOGGER.info(trackeval_results)
        with open(opt.tracking_method + "_output.json", "w") as outfile:
            outfile.write(json.dumps(hota_mota_idf1))
    LOGGER.info(json.dumps(hota_mota_idf1))
    return hota_mota_idf1


def run_all(opt: argparse.Namespace) -> None:
    """
    Runs all stages of the pipeline: generate_dets_embs, generate_mot_results, and trackeval.

    Args:
        opt (Namespace): Parsed command line arguments.
    """
    run_generate_dets_embs(opt)
    run_generate_mot_results(opt)
    run_trackeval(opt)


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Global arguments
    parser.add_argument('--yolo-model', nargs='+', type=Path, default=[WEIGHTS / 'yolov8n.pt'], help='yolo model path')
    parser.add_argument('--reid-model', nargs='+', type=Path, default=[WEIGHTS / 'osnet_x0_25_msmt17.pt'], help='reid model path')
    parser.add_argument('--source', type=str, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--custom-gt-folder', type=str, help='自定义ground truth文件夹的路径')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=None, help='inference size h,w')
    parser.add_argument('--fps', type=int, default=None, help='video frame-rate')
    parser.add_argument('--conf', type=float, default=0.01, help='min confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, default=0, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs', type=Path, help='save results to project/name')
    parser.add_argument('--name', default='', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', default=True, help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--ci', action='store_true', help='Automatically reuse existing due to no UI in CI')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc, boosttrack')
    parser.add_argument('--dets-file-path', type=Path, help='path to detections file')
    parser.add_argument('--embs-file-path', type=Path, help='path to embeddings file')
    parser.add_argument('--exp-folder-path', type=Path, help='path to experiment folder')
    parser.add_argument('--verbose', action='store_true', help='print results')
    parser.add_argument('--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--gsi', action='store_true', help='apply Gaussian smooth interpolation postprocessing')
    parser.add_argument('--n-trials', type=int, default=4, help='nr of trials for evolution')
    parser.add_argument('--objectives', type=str, nargs='+', default=["HOTA", "MOTA", "IDF1"], help='set of objective metrics: HOTA,MOTA,IDF1')
    parser.add_argument('--val-tools-path', type=Path, default=EXAMPLES / 'val_utils', help='path to store trackeval repo in')
    parser.add_argument('--split-dataset', action='store_true', help='Use the second half of the dataset')
    parser.add_argument('--eval-classes', nargs='+', type=int, default=None, help='Classes to evaluate (e.g., 0 or 0 2). Default is all.')

    subparsers = parser.add_subparsers(dest='command')

    # Subparser for generate_dets_embs
    generate_dets_embs_parser = subparsers.add_parser('generate_dets_embs', help='Generate detections and embeddings')
    generate_dets_embs_parser.add_argument('--source', type=str, required=True, help='file/dir/URL/glob, 0 for webcam')
    generate_dets_embs_parser.add_argument('--yolo-model', nargs='+', type=Path, default=WEIGHTS / 'yolov8n.pt', help='yolo model path')
    generate_dets_embs_parser.add_argument('--reid-model', nargs='+', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt', help='reid model path')
    generate_dets_embs_parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    generate_dets_embs_parser.add_argument('--classes', nargs='+', type=int, default=0, help='filter by class: --classes 0, or --classes 0 2 3')

    # Subparser for generate_mot_results
    generate_mot_results_parser = subparsers.add_parser('generate_mot_results', help='Generate MOT results')
    generate_mot_results_parser.add_argument('--yolo-model', nargs='+', type=Path, default=WEIGHTS / 'yolov8n.pt', help='yolo model path')
    generate_mot_results_parser.add_argument('--reid-model', nargs='+', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt', help='reid model path')
    generate_mot_results_parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc, boosttrack')
    generate_mot_results_parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')

    # Subparser for trackeval
    trackeval_parser = subparsers.add_parser('trackeval', help='Evaluate tracking results')
    trackeval_parser.add_argument('--source', type=str, required=True, help='file/dir/URL/glob, 0 for webcam')
    trackeval_parser.add_argument('--exp-folder-path', type=Path, required=True, help='path to experiment folder')

    opt = parser.parse_args()
    source_path = Path(opt.source)
    opt.benchmark, opt.split = source_path.parent.name, source_path.name

    return opt


if __name__ == "__main__":
    opt = parse_opt()
    
    # download MOT benchmark
    download_mot_eval_tools(opt.val_tools_path)

    if not Path(opt.source).exists():
        zip_path = download_mot_dataset(opt.val_tools_path, opt.benchmark)
        unzip_mot_dataset(zip_path, opt.val_tools_path, opt.benchmark)

    if opt.benchmark == 'MOT17':
        cleanup_mot17(opt.source)

    if opt.split_dataset:
        opt.source, opt.benchmark = split_dataset(opt.source)

    if opt.command == 'generate_dets_embs':
        run_generate_dets_embs(opt)
    elif opt.command == 'generate_mot_results':
        run_generate_mot_results(opt)
    elif opt.command == 'trackeval':
        run_trackeval(opt)
    else:
        run_all(opt)

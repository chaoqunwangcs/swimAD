# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import glob
import argparse
from labelme import utils # Keep for imageData decoding fallback
import logging
from datetime import datetime # Needed for info and date_captured
from collections import defaultdict # Needed for dynamic categories
from tqdm import tqdm
import pdb 


EXTS = ['.jpg', '.png']

def setup_logging(log_level, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{current_date}.log")

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file), 
            logging.StreamHandler()
        ]
    )


def parse_arguments():
    """
    使用 argparse 解析命令行参数
    """
    
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument("-r", "--root", type=str, default="dataset", help="dataset root path")
    parser.add_argument("-in", "--input_version", type=str, default="dataset_v20250506", help="input version of the dataset")
    parser.add_argument("-out", "--output_dir", type=str, default="data_transfer", help="output annotation root")
    parser.add_argument("-lf", "--label_format", type=str, choices=["coco", "yolo", "mot"], default="labelme", help="the annotation types")

    parser.add_argument("-l", "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="set log level, default is INFO")
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level)
    setup_logging(log_level, "logs")

    return args

def ensure_dir(directory_path):
    """Creates a directory if it doesn't exist."""
    try:
        os.makedirs(directory_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {directory_path}: {e}")

def parse_custom_label(label_str):
    """
    Parses the LabelMe custom label string "l1,l2,l3,l4" or "l1，l2，l3，l4".
    Handles both half-width (,) and full-width (，) commas.
    Returns a tuple (label_1, label_2, label_3, label_4) as integers.
    Returns None if parsing fails or format is incorrect.
    """
    if not label_str:
        logging.warning("Empty label string encountered.")
        return None

    # *** FIX: Replace full-width comma with half-width comma ***
    normalized_label_str = label_str.replace('，', ',')

    parts = normalized_label_str.split(',')
    if len(parts) != 4:
        # Log the original string for better debugging
        logging.warning(f"Label string '{label_str}' (normalized: '{normalized_label_str}') does not have 4 parts after normalization. Skipping.")
        return None
    try:
        # Convert parts to integers
        labels = tuple(int(p.strip()) for p in parts)
        return labels
    except ValueError:
        # Log the original string for better debugging
        logging.warning(f"Could not convert parts of label '{label_str}' (normalized: '{normalized_label_str}') to integers. Skipping.")
        return None

def preprocess_VL(folders):
    pass

def main(args):

    logging.info(f"Starting recursive search for dataset version: {args.input_version}")
    logging.info(f"transfed {args.label_format} format annotation will be saved under: {args.output_dir}")
    # parse video clips and views of a dataset
    all_view_folders = []
    dataset_path = os.path.join(args.root, args.input_version)
    clips = os.listdir(dataset_path)
    for clip in clips:
        clip_path = os.path.join(dataset_path, clip)
        views = os.listdir(clip_path)
        for view in views:
            view_path = os.path.join(clip_path, view)
            all_view_folders.append(view_path)

    all_ignore_box = dict() # img_path: box: 
    for folder in all_view_folders:
        ignore_box = preprocess_VL(folder)
        all_ignore_box.update(ignore_box)
    
    with open(os.path.join(args.root, 'ignore_box.json'), 'w') as f:
        json.dump(all_ignore_box)

if __name__ == '__main__':

    args = parse_arguments()
    main(args)



    # if not os.path.isdir(LABELME_JSON_DIR):
    #     logging.error(f"LabelMe JSON directory not found: {LABELME_JSON_DIR}")
    #     exit(1) # Use non-zero exit code for errors
    # if not SAVED_COCO_PATH:
    #      logging.error("Output COCO path (SAVED_COCO_PATH) is not specified.")
    #      exit(1)
    # if not CATEGORY_NAME_PREFIX:
    #     logging.warning("CATEGORY_NAME_PREFIX is empty. Category names will just be the label_4 numbers.")


    # # --- Create Output Directory ---
    # try:
    #     # Check if path exists and is a directory, create if not
    #     if not os.path.exists(SAVED_COCO_PATH):
    #          os.makedirs(SAVED_COCO_PATH, exist_ok=True)
    #          logging.info(f"Created output directory: {SAVED_COCO_PATH}")
    #     elif not os.path.isdir(SAVED_COCO_PATH):
    #          logging.error(f"Output path {SAVED_COCO_PATH} exists but is not a directory.")
    #          exit(1)
    #     else:
    #          logging.info(f"Output directory already exists: {SAVED_COCO_PATH}")

    # except Exception as e:
    #     logging.error(f"Could not create or access output directory {SAVED_COCO_PATH}: {e}")
    #     exit(1)


    # # --- Find LabelMe JSON files ---
    # json_pattern = os.path.join(LABELME_JSON_DIR, "*.json")
    # json_list_path = glob.glob(json_pattern) # Get all JSON file paths
    # if not json_list_path:
    #     logging.error(f"No JSON files found in directory: {LABELME_JSON_DIR} with pattern '*.json'")
    #     exit(1)
    # logging.info(f"Found {len(json_list_path)} JSON files in {LABELME_JSON_DIR}")

    # # --- Initialize Converter ---
    # converter = Labelme2Coco()

    # # --- Process All Data ---
    # logging.info("Starting LabelMe to COCO conversion process...")
    # coco_instance = converter.generate_coco_annotation(json_list_path) # Generate annotations




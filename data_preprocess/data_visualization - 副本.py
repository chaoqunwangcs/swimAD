import json
import os
import numpy as np
import cv2
import argparse
import logging
from datetime import datetime
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
    parser.add_argument("-lf", "--label_format", type=str, choices=["labelme", "coco", "yolo", "mot"], default="labelme", help="the annotation types")
    parser.add_argument("-out", "--output_dir", type=str, default="data_visualtion", help="input version of the dataset")

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

def read_image_safe(image_path):
    """
    Reads an image from a path that might contain non-ASCII characters.
    Returns the image as a NumPy array or None if reading fails.
    """
    try:
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        img_np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
        if img is None:
             return None
        return img
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading/decoding image {image_path}: {e}")
        return None

def visualize(img_path, anno_path, save_path):
    img = read_image_safe(img_path)
    with open(anno_path, 'r', encoding='utf-8') as f:
        ann_data = json.load(f)

    annotations_drawn = 0
    try:
        if 'shapes' not in ann_data or not isinstance(ann_data['shapes'], list):
                logging.warning(f"'shapes' key missing or not a list in {anno_path}. Skipping drawing.")
        else:
            for num_annotation, shape in enumerate(ann_data['shapes']):
                if shape.get('shape_type') != 'rectangle':
                    continue

                # *** MODIFICATION: Check for required keys: 'points', 'label', 'line_color' ***
                required_keys = ['points', 'label', 'line_color']
                if not all(k in shape for k in required_keys):
                    missing_keys = [k for k in required_keys if k not in shape]
                    logging.warning(f"Annotation #{num_annotation+1} in {anno_path} is missing keys: {missing_keys}. Skipping annotation.")
                    continue
                if not isinstance(shape['points'], list) or len(shape['points']) != 2:
                    logging.warning(f"Annotation #{num_annotation+1} in {anno_path} has 'points' with wrong format. Skipping annotation.")
                    continue
                # Check line_color format
                if not isinstance(shape['line_color'], list) or len(shape['line_color']) < 3:
                        logging.warning(f"Annotation #{num_annotation+1} in {anno_path} has 'line_color' with wrong format (expected list of >=3 numbers). Skipping annotation.")
                        continue

                try:
                    # Extract points
                    points = shape['points']
                    if not (len(points[0]) == 2 and len(points[1]) == 2):
                        logging.warning(f"Invalid coordinate structure in 'points' for annotation #{num_annotation+1} in {anno_path}. Skipping.")
                        continue
                    xmin = int(points[0][0])
                    ymin = int(points[0][1])
                    xmax = int(points[1][0])
                    ymax = int(points[1][1])

                    if xmin >= xmax or ymin >= ymax:
                        logging.warning(f" Invalid coordinates (min >= max) in annotation #{num_annotation+1} in {anno_path}: ({xmin},{ymin}) ({xmax},{ymax}). Skipping.")
                        continue

                    # Extract label
                    cat_label = str(shape['label'])
                    cat_label = cat_label.replace('，', ',')
                    display_text = cat_label

                    # *** MODIFICATION: Extract line_color and convert RGBA to BGR ***
                    line_color_rgba = shape['line_color']
                    # Ensure color values are integers
                    try:
                        r = int(line_color_rgba[0])
                        g = int(line_color_rgba[1])
                        b = int(line_color_rgba[2])
                    except (ValueError, IndexError):
                        logging.warning(f"Invalid number format in 'line_color' for annotation #{num_annotation+1} in {anno_path}. Using default color. Skipping annotation.")
                        continue # Skip this annotation if color is invalid

                    # Clamp values to 0-255
                    r = max(0, min(255, r))
                    g = max(0, min(255, g))
                    b = max(0, min(255, b))

                    bgr_color = (b, g, r) # OpenCV uses BGR order

                except (ValueError, TypeError, IndexError, KeyError) as e:
                    logging.warning(f"Invalid/missing data in annotation #{num_annotation+1} in {anno_path}: {e}. Skipping annotation.")
                    continue

                # --- Define drawing parameters using JSON color ---
                color = bgr_color      # Use converted BGR for border
                #text_color = bgr_color # Use converted BGR for text (adjust if contrast is bad)
                text_color =( 0, 0, 255) # BGR for Red
                
                thickness = 2 # You might want to adjust thickness too
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                text_thickness = 2

                # Draw the rectangle
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

                # Draw the label text
                text_y_pos = ymin - 7
                if text_y_pos < 10:
                    text_y_pos = ymin + 15
                # Note: If text_color makes text hard to read against some backgrounds,
                # you might want to define a separate text_color logic,
                # e.g., always white or black, or based on the brightness of bgr_color.
                cv2.putText(img, display_text, (xmin, text_y_pos), font, font_scale, text_color, text_thickness, lineType=cv2.LINE_AA)
                annotations_drawn += 1
    except:
        pdb.set_trace()
    if annotations_drawn > 0:
        logging.info(f"Finished drawing {annotations_drawn} rectangle annotations for {img_path}")
    elif 'shapes' in ann_data and isinstance(ann_data['shapes'], list):
        logging.warning(f"No valid 'rectangle' annotations with required keys found in {anno_path}.")
    
    is_success, im_buf_arr = cv2.imencode(os.path.splitext(save_path)[1], img)
    if is_success:
        with open(save_path, 'wb') as f:
            f.write(im_buf_arr.tobytes())
    
    return is_success
    
def process_folder(source_path, save_path):
    # pdb.set_trace()

    imgs = sorted(list(filter(lambda x: any(x.endswith(EXT) for EXT in EXTS), os.listdir(source_path))))

    for img in tqdm(imgs, desc=f"{source_path}: "):
        img_path = os.path.join(source_path, img)
        json_path = os.path.join(source_path, os.path.splitext(img)[0]+'.json')
        if not (os.path.exists(img_path) and os.path.exists(json_path)):
            logging.warning(f"file: {img_path} or {json_path} not exist")
            continue
        save_img_path = os.path.join(save_path, img)

        is_success = visualize(img_path, json_path, save_img_path)
        if not is_success:
            pdb.set_trace()

    return len(imgs)

def main(args):
    total_num = 0

    logging.info(f"Starting recursive search for dataset version: {args.input_version}")
    logging.info(f"Annotated images will be saved under: {args.output_dir}")

    
    # parse video clips and views of a dataset
    dataset_path = os.path.join(args.root, args.input_version)
    clips = os.listdir(dataset_path)
    for clip in clips:
        clip_path = os.path.join(dataset_path, clip)
        views = os.listdir(clip_path)
        for view in views:
            view_path = os.path.join(clip_path, view)
            save_view_path = os.path.join(args.output_dir, clip, view); ensure_dir(save_view_path)
            
            # process data in a folder
            img_num = process_folder(view_path, save_view_path)
            total_num += img_num
    
    log.info(f"finish {total_num} images")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)


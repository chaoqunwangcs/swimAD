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

class Labelme2Coco:
    """
    Handles the conversion from LabelMe format (with custom labels) to COCO format.
    Uses the 4th value from the custom label ("l1,l2,l3,l4") as the category ID.
    Parses "label" field like "1,1,0,0" or "1，1，0，0" into label_1, label_2, label_3, label_4.
    Includes line_color in the annotation.
    Removes segmentation data (sets to []).
    Saves only the annotations.json file directly to the specified path.
    Does NOT copy image files or find/read external image files for metadata.
    Relies solely on information within the LabelMe JSON files.
    Dynamically creates categories based on unique label_4 values found
    OR forces a single category based on the FORCE_SINGLE_CATEGORY flag.
    """
    def __init__(self):
        # Counters for unique IDs
        self.img_id_counter = 1 # Start image IDs from 1
        self.ann_id_counter = 1 # Start annotation IDs from 1
        self.images = []
        self.annotations = []
        self.found_category_ids = set()

        # hyper-parameter setting
        self.OUTPUT_IMAGE_EXT = ".jpg"
        self.CATEGORY_NAME_PREFIX = "object_"

        self.FORCE_SINGLE_CATEGORY = False  # if force using single category
        self.SINGLE_CATEGORY_NAME = "object"


    def _process_single_json(self, folder_path, json_file):
        """Processes a single LabelMe JSON file."""
        # pdb.set_trace()
        current_img_id = self.img_id_counter
        try:
            labelme_data = self._read_jsonfile(os.path.join(folder_path, json_file))
            if labelme_data is None: return None, []

            # Create image entry using only info from JSON file
            image_info = self._create_image_entry_from_json(labelme_data, folder_path, json_file, current_img_id)
            if image_info is None: return None, []

            img_annotations = []
            for shape in labelme_data.get('shapes', []):
                annotation = self._create_annotation_entry(shape, current_img_id)
                if annotation:
                    img_annotations.append(annotation)
                    # Keep track of the category ID used ONLY if not forcing single category
                    if not self.FORCE_SINGLE_CATEGORY:
                        self.found_category_ids.add(annotation['category_id'])
                    self.ann_id_counter += 1 # Increment annotation ID

            self.img_id_counter += 1 # Increment image ID
            return image_info, img_annotations
        except Exception as e:
            logging.error(f"Error processing JSON file {json_file}: {e}", exc_info=True)
            return None, []

    def generate_coco_annotation(self, folder_path):
        """
        Generates the COCO annotation structure for the given list of LabelMe JSON paths.
        Returns the COCO annotation dictionary with dynamically generated categories based on label_4.
        """
        json_file_list = sorted(list(filter(lambda x: x.endswith('.json'), os.listdir(folder_path))))
        for json_file in tqdm(json_file_list, desc=f"Processing {folder_path} files: "):
            image_info, img_annotations = self._process_single_json(folder_path, json_file)
            if image_info:
                self.images.append(image_info)
                self.annotations.extend(img_annotations)

    def merge_coco_annotion(self):
        # --- Define categories based on the chosen mode ---
        categories = []
        description_suffix = ""

        if self.FORCE_SINGLE_CATEGORY:
            # Mode 1: Force single category (like v1.5)
            categories = [{
                'id': 1,
                'name': self.SINGLE_CATEGORY_NAME,
                'supercategory': self.SINGLE_CATEGORY_NAME
            }]
            logging.info(f"FORCE_SINGLE_CATEGORY is True. Using single category definition: {categories}")
            description_suffix = f" (Forced Category ID 1: {self.SINGLE_CATEGORY_NAME}, Custom Labels)"
        else:
            # Mode 2: Dynamic categories from label_4 (original v1.6 behavior)
            sorted_category_ids = sorted(list(self.found_category_ids))
            for cat_id in sorted_category_ids:
                category_name = f"{self.CATEGORY_NAME_PREFIX}{int(cat_id)}"
                categories.append({
                    'id': int(cat_id),
                    'name': category_name,
                    'supercategory': category_name
                })
            logging.info(f"FORCE_SINGLE_CATEGORY is False. Generated {len(categories)} categories based on unique label_4 values: {sorted_category_ids}")
            logging.info(f"Category definitions: {categories}")
            description_suffix = " (Category ID from label_4, Custom Labels)"

        now = datetime.now()
        coco_format = {
            'info': {
                'description': f'Converted LabelMe annotations to COCO format{description_suffix}',
                'version': '1.0', 'year': now.year, 'contributor': 'Conversion Script',
                'date_created': now.strftime('%Y/%m/%d'),
            },
            'licenses': [{'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 1, 'name': 'Attribution License'}],
            'images': self.images,
            'annotations': self.annotations,
            'categories': categories, # Use the dynamically generated category list
        }
        return coco_format

    def _create_image_entry_from_json(self, labelme_data, folder_path, json_file, current_img_id):
        """
        Creates the image entry for the COCO format using ONLY info from the LabelMe JSON.
        Does not attempt to find or read external image files.
        """
        image_path_in_json = labelme_data.get('imagePath')
        h, w = 0, 0 # Initialize height and width

        # 1. Determine filename for COCO JSON 'file_name' field
        if image_path_in_json:
            # pdb.set_trace()
            # Use the filename provided in the JSON's imagePath field
            output_filename = os.path.join(folder_path, image_path_in_json)
        else:
            # Fallback: use the JSON filename itself as the base for 'file_name'
            output_filename = os.path.join(folder_path, json_file.replace(".png",self.OUTPUT_IMAGE_EXT))
            logging.warning(f"'imagePath' not found in {json_file}. Using JSON filename for 'file_name' field: {output_filename}")

        # 2. Get image dimensions
        # Priority: imageHeight/imageWidth fields > imageData > Error
        if 'imageHeight' in labelme_data and 'imageWidth' in labelme_data:
             h = labelme_data['imageHeight']
             w = labelme_data['imageWidth']
             logging.debug(f"Using image dimensions from JSON fields for {json_file}")
        elif labelme_data.get('imageData'):
             try:
                 img_arr = utils.img_b64_to_arr(labelme_data['imageData'])
                 h, w = img_arr.shape[:2]
                 logging.debug(f"Using image dimensions from imageData for {json_file}")
             except Exception as e:
                 logging.error(f"Could not decode imageData in {json_file} to get dimensions: {e}")
                 # Cannot determine dimensions, fail this image entry
                 return None
        else:
             # Dimensions are required, cannot proceed without them
             logging.error(f"Cannot determine image dimensions for {json_file}: Missing 'imageHeight'/'imageWidth' fields and 'imageData'.")
             return None

        # Validate dimensions
        if not isinstance(h, (int, float)) or not isinstance(w, (int, float)) or h <= 0 or w <= 0:
            logging.error(f"Invalid image dimensions (h={h}, w={w}) found or determined for {json_file}.")
            return None


        # 3. Create the image entry dictionary
        image_entry = {
            'id': current_img_id, 'width': int(w), 'height': int(h), 'file_name': output_filename, # Ensure width/height are ints
            'license': 1, 'flickr_url': '', 'coco_url': '',
            'date_captured': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        return image_entry

    def _create_annotation_entry(self, shape, current_image_id):
        """
        Creates a single annotation entry:
        - Uses the 4th value from the custom label as category ID OR forces ID to 1.
        - Parses custom label "l1,l2,l3,l4" (handles both comma types).
        - Adds label_1, label_2, label_3, label_4 fields.
        - Adds line_color field.
        - Sets segmentation to [].
        """
        # pdb.set_trace()
        label_str = shape.get('label')
        points = shape.get('points')
        shape_type = shape.get('shape_type', 'polygon')
        line_color = shape.get('line_color') # Get line color

        if not label_str or not points:
            logging.warning(f"Skipping shape due to missing label or points in image_id {current_image_id}.")
            return None

        # Parse the custom label string
        custom_labels = parse_custom_label(label_str)
        if custom_labels is None:
             # Warning is already logged in parse_custom_label
             return None # Skip if label parsing failed

        # *** Determine category_id based on the mode ***
        if self.FORCE_SINGLE_CATEGORY:
            category_id = 1
        else:
            # Use the 4th label as category_id (original v1.6 logic)
            try:
                category_id = int(custom_labels[3]) # The 4th element (index 3)
            except (IndexError, ValueError):
                 logging.warning(f"Could not extract a valid integer category ID (label_4) from label '{label_str}' for annotation id {self.ann_id_counter}. Skipping.")
                 return None


        # Initialize annotation dictionary
        annotation = {
            'id': self.ann_id_counter,
            'image_id': current_image_id,
            'category_id': category_id, # Use extracted category ID
            'iscrowd': 0,
            'segmentation': [], # Set segmentation to empty list as requested
            'bbox': [],
            'area': 0.0,
            # Add custom label fields
            'label_1': custom_labels[0],    # need modified
            'label_2': custom_labels[1],    # need modified
            'label_3': custom_labels[2],    # need modified
            'label_4': custom_labels[3],    # need modified
            'line_color': line_color if line_color else None,
        }

        # Calculate BBox and Area based on shape type
        if shape_type == 'polygon' or shape_type == 'linestrip' or shape_type == 'line':
            if len(points) < 3:
                logging.warning(f"Skipping polygon/line with < 3 points for annotation id {self.ann_id_counter}")
                return None
            # Segmentation is set to [] above, no need to calculate it here
            annotation['area'] = self._calculate_polygon_area(points)
            annotation['bbox'] = self._get_bounding_box(points)
        elif shape_type == 'rectangle':
            if len(points) != 2:
                 logging.warning(f"Skipping rectangle with != 2 points for annotation id {self.ann_id_counter}")
                 return None
            # Ensure points are numerical lists/tuples
            try:
                p1 = [float(coord) for coord in points[0]]
                p2 = [float(coord) for coord in points[1]]
            except (ValueError, TypeError):
                 logging.warning(f"Invalid coordinate format in rectangle points for annotation id {self.ann_id_counter}. Points: {points}. Skipping.")
                 return None

            x1, y1 = p1; x2, y2 = p2
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x1 - x2)
            h = abs(y1 - y2)
            annotation['bbox'] = [x, y, w, h]
            # Segmentation is set to [] above, no need to calculate it here
            annotation['area'] = w * h
        else:
            logging.warning(f"Unsupported shape type: {shape_type} for annotation id {self.ann_id_counter}. Skipping.")
            return None

        # Final validation for bbox and area
        # Check if bbox has 4 valid numbers and width/height are positive
        if not (isinstance(annotation.get('bbox'), list) and
                len(annotation['bbox']) == 4 and
                all(isinstance(v, (int, float)) for v in annotation['bbox']) and
                annotation['bbox'][2] > 0 and
                annotation['bbox'][3] > 0):
            logging.warning(f"Invalid bounding box generated for annotation id {self.ann_id_counter}: {annotation.get('bbox')}. Skipping.")
            return None

        # Check if area is a positive number
        if not (isinstance(annotation.get('area'), (int, float)) and annotation['area'] > 0):
             logging.warning(f"Non-positive or invalid area calculated for annotation id {self.ann_id_counter}: {annotation.get('area')}. Skipping.")
             return None


        return annotation

    def _read_jsonfile(self, path):
        """Reads a JSON file."""
        try:
            with open(path, "r", encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed read JSON {path}: {e}")
            return None

    def _get_bounding_box(self, points):
        """Calculates bounding box from polygon points."""
        if not points or len(points) < 1: return [0.0, 0.0, 0.0, 0.0]
        try:
            # Ensure points are numeric
            np_points = np.array(points, dtype=float)
            if np_points.shape[1] != 2: # Check if points have x, y coordinates
                 logging.warning(f"Invalid point dimensions for bbox calculation: {points}. Returning zero bbox.")
                 return [0.0, 0.0, 0.0, 0.0]
        except (ValueError, TypeError):
             logging.warning(f"Invalid data type in points for bbox calculation: {points}. Returning zero bbox.")
             return [0.0, 0.0, 0.0, 0.0]

        min_coords = np.min(np_points, axis=0)
        max_coords = np.max(np_points, axis=0)
        # COCO format: [xmin, ymin, width, height]
        width = max_coords[0] - min_coords[0]
        height = max_coords[1] - min_coords[1]
        return [
            float(min_coords[0]),
            float(min_coords[1]),
            float(width) if width >= 0 else 0.0, # Ensure non-negative width/height
            float(height) if height >= 0 else 0.0
        ]


    def _calculate_polygon_area(self, points):
        """Calculates polygon area using the Shoelace formula."""
        if not points or len(points) < 3: return 0.0
        try:
             # Ensure points are numeric
            np_points = np.array(points, dtype=float)
            if np_points.shape[1] != 2: # Check if points have x, y coordinates
                 logging.warning(f"Invalid point dimensions for area calculation: {points}. Returning 0.0 area.")
                 return 0.0
        except (ValueError, TypeError):
             logging.warning(f"Invalid data type in points for area calculation: {points}. Returning 0.0 area.")
             return 0.0

        x = np_points[:, 0]
        y = np_points[:, 1]
        # Apply Shoelace formula: 0.5 * |(x1*y2 + x2*y3 + ... + xn*y1) - (y1*x2 + y2*x3 + ... + yn*x1)|
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return float(area)


    def save_coco_json(self, instance, save_path):
        """Saves the COCO JSON data to a file."""
        try:
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                # Use indent for readability
                json.dump(instance, f, ensure_ascii=False, indent=2)
            logging.info(f"Successfully saved COCO JSON to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save COCO JSON to {save_path}: {e}")


def transfer_to_coco(folders, save_path):

    converter = Labelme2Coco()
    for folder in folders:
        # pdb.set_trace()
        converter.generate_coco_annotation(folder)
    
    coco_annotation = converter.merge_coco_annotion()

    # save
    converter.save_coco_json(coco_annotation, save_path)
    logging.info(f"Processed {len(coco_annotation['images'])} images and {len(coco_annotation.get('annotations', []))} annotations.") # Use .get for annotations too
    logging.info(f"Total unique categories created: {len(coco_annotation.get('categories', []))}")
    logging.info("LabelMe to COCO conversion finished! (Category from label_4, custom labels, no segmentation)")


def transfer_to_mot_video(folder, save_path):
    # pdb.set_trace()
    json_file_list = sorted(list(filter(lambda x: x.endswith('.json'), os.listdir(folder))))
    frame_id = 0
    normal_annotations = []
    abnormal_annotations = []   # with cls conf < 1     # box conf need put in mot_line
    for json_file in tqdm(json_file_list, desc=f"Processing {folder} files: "):
        frame_id += 1
        json_file_path = os.path.join(folder, json_file)

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
                img_height = json_data.get('imageHeight')
                img_width = json_data.get('imageWidth')
                shapes = json_data.get('shapes', [])
                assert img_height is not None and img_width is not None
                assert isinstance(img_height, (int, float)) and isinstance(img_width, (int, float))
                assert img_height > 0 or img_width > 0

                for shape in shapes:
                    label_str = shape.get('label')
                    points = shape.get('points')
                    shape_type = shape.get('shape_type')

                    if shape_type != 'rectangle' or label_str is None or points is None or len(points) != 2:
                        logging.info(f"  Skipping invalid/non-rectangle annotation in {filename}: {shape}")
                        continue
                    
                    parsed_labels = parse_custom_label(label_str)
                    if parsed_labels is None:
                        logging.error(f"Parsing label error for {label_str}")
                        pdb.set_trace()
                    
                    track_id = parsed_labels[0]
                    category_id = parsed_labels[3]

                    try:
                        x1, y1 = map(float, points[0])
                        x2, y2 = map(float, points[1])
                    except (ValueError, TypeError):
                        logging.error(f"  Skipping annotation with invalid coords in {filename}: {points}")
                        pdb.set_trace()
                    
                    bb_left = round(min(x1, x2))
                    bb_top = round(min(y1, y2))
                    bb_width = round(abs(x1 - x2))
                    bb_height = round(abs(y1 - y2))
                    if bb_width <= 0 or bb_height <= 0:
                        logging.error(f"  Skipping annotation with zero size in {filename}: w={bb_width}, h={bb_height}")
                        pdb.set_trace()
                    
                    conf = 1.0  # need modified, the true conf is annotated
                    world_x, world_y, world_z = -1, -1, -1

                    mot_line = f"{frame_id},{track_id},{bb_left},{bb_top},{bb_width},{bb_height},{conf},{category_id},{world_x},{world_y},{world_z}"
                    
                    if parsed_labels[1] == 0 or parsed_labels[2] == 0:
                        abnormal_annotations.append(mot_line)
                    else:
                        normal_annotations.append(mot_line)

        except:
            logging.warning(f"process failed in {json_file_path}.")
        
    
    normal_annotations.sort(key=lambda x: int(mot_line.split(',')[0]))
    abnormal_annotations.sort(key=lambda x: int(mot_line.split(',')[0]))

    normal_annotations_file_path = os.path.join(save_path, 'normal_seq.tx')
    abnormal_annotations_file_path = os.path.join(save_path, 'abnormal_seq.tx')

    if normal_annotations:
        with open(normal_annotations_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(normal_annotations) + "\n")
    else: logging.info(f"No normal annotations found for {json_file_path}.")

    if abnormal_annotations:
        with open(abnormal_annotations_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(abnormal_annotations) + "\n")
    else: logging.info(f"No abnormal annotations found for {json_file_path}.")

    logging.info(f"Finished processing directory {json_file_path}.")


def transfer_to_mot(folders, save_path):
    for folder in folders:
        data_root, data_version, clip_id, view_id = folder.split('/')
        save_folder_path = os.path.join(save_path, clip_id, view_id); ensure_dir(save_folder_path)
        transfer_to_mot_video(folder, save_folder_path)

    logging.info(f"Finished processeing {len(folders)} videos.")


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
    if args.label_format == 'coco':
        save_path = os.path.join(args.output_dir, f"annotation_coco_{args.input_version}.json")
        transfer_to_coco(all_view_folders, save_path)
    
    elif args.label_format == 'yolo':   # for YOLO model
        raise NotImplementedError(f"This function not support {args.input_version} format! comming soon")
    
    elif args.label_format == 'mot': # for MOT task
        save_path = os.path.join(args.output_dir, "mot",f"{args.input_version}")
        transfer_to_mot(all_view_folders, save_path)

    else:
        raise NotImplementedError(f"This function not support {args.input_version} format!")

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




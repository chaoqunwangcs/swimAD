# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import glob
# Removed: import cv2
from labelme import utils # Keep for imageData decoding fallback
import logging
import datetime # Needed for info and date_captured
from collections import defaultdict # Needed for dynamic categories


'''
LABELME_JSON_DIR\  <-- 这是你的 LABELME_JSON_DIR
            │
            │   frame_001.json   <-- **你的 LabelMe .json 文件必须直接放在这里

label第一位为跟踪id
在目标文件夹中，将label分类为正常和异常。
label中第二位或者第三位置信度为0的，视作异常label。
默认label第四位作为category id。可选择强制所有label的第四位为1，或者保持label第四位为category id。

'''

# --- Configuration Section (Modify these parameters) ---

# 1. Specify the path to your LabelMe annotation files (.json)
#    *** NOTE: Script relies ONLY on info inside JSON (imagePath, dimensions, imageData) ***
#    *** NOTE: Expects label format like "l1,l2,l3,l4" or "l1，l2，l3，l4" (handles both comma types) ***
#    *** NOTE: The 4th value (l4) will be used as the category_id ***
LABELME_JSON_DIR = r"G:\project\20250423游泳池数据最新标签修改返回\下午\3" 

# 2. Specify the path where the COCO annotations.json file will be saved
#    *** NOTE: Images will NOT be copied. ***
SAVED_COCO_PATH = r"G:\project\20250423游泳池数据最新标签修改返回__processed\yolo_label\label_test" # <<< CHANGE THIS: Use raw string (r"...") - Recommend a dedicated output folder

# 3. Define the output image format (e.g., '.jpg', '.png')
#    This affects the 'file_name' field in the JSON.
OUTPUT_IMAGE_EXT = ".jpg"

# 4. Define a prefix or naming convention for categories based on label_4
#    The final category name will be f"{CATEGORY_NAME_PREFIX}{label_4_value}"
#    Example: If label_4 is 5, category name will be "object_5"
CATEGORY_NAME_PREFIX = "object_"

# --- NEW: Control Category Behavior ---
# Set to True to force all annotations to category_id 1 (like v1.5)
# Set to False to use the 4th label value as category_id (like v1.6 original)
FORCE_SINGLE_CATEGORY = False # <<< CHANGE THIS (True or False)

# Define the name for the single category IF FORCE_SINGLE_CATEGORY is True
SINGLE_CATEGORY_NAME = "object" # <<< CHANGE THIS if FORCE_SINGLE_CATEGORY is True

# --- End of Configuration Section ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        # Store unique category IDs (label_4 values) found during processing
        # Only used if FORCE_SINGLE_CATEGORY is False
        self.found_category_ids = set()

    def _parse_custom_label(self, label_str):
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

    def _process_single_json(self, json_path):
        """Processes a single LabelMe JSON file."""
        current_img_id = self.img_id_counter
        try:
            labelme_data = self._read_jsonfile(json_path)
            if labelme_data is None: return None, []

            # Create image entry using only info from JSON file
            image_info = self._create_image_entry_from_json(labelme_data, json_path, current_img_id)
            if image_info is None: return None, []

            img_annotations = []
            for shape in labelme_data.get('shapes', []):
                annotation = self._create_annotation_entry(shape, current_img_id)
                if annotation:
                    img_annotations.append(annotation)
                    # Keep track of the category ID used ONLY if not forcing single category
                    if not FORCE_SINGLE_CATEGORY:
                        self.found_category_ids.add(annotation['category_id'])
                    self.ann_id_counter += 1 # Increment annotation ID

            self.img_id_counter += 1 # Increment image ID
            return image_info, img_annotations
        except Exception as e:
            logging.error(f"Error processing JSON file {json_path}: {e}", exc_info=True)
            return None, []

    def generate_coco_annotation(self, json_path_list):
        """
        Generates the COCO annotation structure for the given list of LabelMe JSON paths.
        Returns the COCO annotation dictionary with dynamically generated categories based on label_4.
        """
        self.images = []
        self.annotations = []
        self.found_category_ids = set() # Reset found categories for this run

        for json_path in json_path_list:
            image_info, img_annotations = self._process_single_json(json_path)
            if image_info:
                self.images.append(image_info)
                self.annotations.extend(img_annotations)

        # --- Define categories based on the chosen mode ---
        categories = []
        description_suffix = ""

        if FORCE_SINGLE_CATEGORY:
            # Mode 1: Force single category (like v1.5)
            categories = [{
                'id': 1,
                'name': SINGLE_CATEGORY_NAME,
                'supercategory': SINGLE_CATEGORY_NAME
            }]
            logging.info(f"FORCE_SINGLE_CATEGORY is True. Using single category definition: {categories}")
            description_suffix = f" (Forced Category ID 1: {SINGLE_CATEGORY_NAME}, Custom Labels)"
        else:
            # Mode 2: Dynamic categories from label_4 (original v1.6 behavior)
            sorted_category_ids = sorted(list(self.found_category_ids))
            for cat_id in sorted_category_ids:
                category_name = f"{CATEGORY_NAME_PREFIX}{int(cat_id)}"
                categories.append({
                    'id': int(cat_id),
                    'name': category_name,
                    'supercategory': category_name
                })
            logging.info(f"FORCE_SINGLE_CATEGORY is False. Generated {len(categories)} categories based on unique label_4 values: {sorted_category_ids}")
            logging.info(f"Category definitions: {categories}")
            description_suffix = " (Category ID from label_4, Custom Labels)"

        now = datetime.datetime.now()
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

    def _create_image_entry_from_json(self, labelme_data, json_path, current_img_id):
        """
        Creates the image entry for the COCO format using ONLY info from the LabelMe JSON.
        Does not attempt to find or read external image files.
        """
        image_path_in_json = labelme_data.get('imagePath')
        h, w = 0, 0 # Initialize height and width

        # 1. Determine filename for COCO JSON 'file_name' field
        if image_path_in_json:
            # Use the filename provided in the JSON's imagePath field
            output_filename = os.path.splitext(os.path.basename(image_path_in_json))[0] + OUTPUT_IMAGE_EXT
        else:
            # Fallback: use the JSON filename itself as the base for 'file_name'
            output_filename = os.path.splitext(os.path.basename(json_path))[0] + OUTPUT_IMAGE_EXT
            logging.warning(f"'imagePath' not found in {json_path}. Using JSON filename for 'file_name' field: {output_filename}")

        # 2. Get image dimensions
        # Priority: imageHeight/imageWidth fields > imageData > Error
        if 'imageHeight' in labelme_data and 'imageWidth' in labelme_data:
             h = labelme_data['imageHeight']
             w = labelme_data['imageWidth']
             logging.debug(f"Using image dimensions from JSON fields for {json_path}")
        elif labelme_data.get('imageData'):
             try:
                 img_arr = utils.img_b64_to_arr(labelme_data['imageData'])
                 h, w = img_arr.shape[:2]
                 logging.debug(f"Using image dimensions from imageData for {json_path}")
             except Exception as e:
                 logging.error(f"Could not decode imageData in {json_path} to get dimensions: {e}")
                 # Cannot determine dimensions, fail this image entry
                 return None
        else:
             # Dimensions are required, cannot proceed without them
             logging.error(f"Cannot determine image dimensions for {json_path}: Missing 'imageHeight'/'imageWidth' fields and 'imageData'.")
             return None

        # Validate dimensions
        if not isinstance(h, (int, float)) or not isinstance(w, (int, float)) or h <= 0 or w <= 0:
            logging.error(f"Invalid image dimensions (h={h}, w={w}) found or determined for {json_path}.")
            return None


        # 3. Create the image entry dictionary
        image_entry = {
            'id': current_img_id, 'width': int(w), 'height': int(h), 'file_name': output_filename, # Ensure width/height are ints
            'license': 1, 'flickr_url': '', 'coco_url': '',
            'date_captured': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
        label_str = shape.get('label')
        points = shape.get('points')
        shape_type = shape.get('shape_type', 'polygon')
        line_color = shape.get('line_color') # Get line color

        if not label_str or not points:
            logging.warning(f"Skipping shape due to missing label or points in image_id {current_image_id}.")
            return None

        # Parse the custom label string
        custom_labels = self._parse_custom_label(label_str)
        if custom_labels is None:
             # Warning is already logged in _parse_custom_label
             return None # Skip if label parsing failed

        # *** Determine category_id based on the mode ***
        if FORCE_SINGLE_CATEGORY:
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
            'label_1': custom_labels[0],
            'label_2': custom_labels[1],
            'label_3': custom_labels[2],
            'label_4': custom_labels[3], # Keep label_4 field as well if needed
            # Add line color if available
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

# --- Main Execution Logic ---
if __name__ == '__main__':

    # --- Validate Configuration ---
    if not os.path.isdir(LABELME_JSON_DIR):
        logging.error(f"LabelMe JSON directory not found: {LABELME_JSON_DIR}")
        exit(1) # Use non-zero exit code for errors
    if not SAVED_COCO_PATH:
         logging.error("Output COCO path (SAVED_COCO_PATH) is not specified.")
         exit(1)
    if not CATEGORY_NAME_PREFIX:
        logging.warning("CATEGORY_NAME_PREFIX is empty. Category names will just be the label_4 numbers.")


    # --- Create Output Directory ---
    try:
        # Check if path exists and is a directory, create if not
        if not os.path.exists(SAVED_COCO_PATH):
             os.makedirs(SAVED_COCO_PATH, exist_ok=True)
             logging.info(f"Created output directory: {SAVED_COCO_PATH}")
        elif not os.path.isdir(SAVED_COCO_PATH):
             logging.error(f"Output path {SAVED_COCO_PATH} exists but is not a directory.")
             exit(1)
        else:
             logging.info(f"Output directory already exists: {SAVED_COCO_PATH}")

    except Exception as e:
        logging.error(f"Could not create or access output directory {SAVED_COCO_PATH}: {e}")
        exit(1)


    # --- Find LabelMe JSON files ---
    json_pattern = os.path.join(LABELME_JSON_DIR, "*.json")
    json_list_path = glob.glob(json_pattern) # Get all JSON file paths
    if not json_list_path:
        logging.error(f"No JSON files found in directory: {LABELME_JSON_DIR} with pattern '*.json'")
        exit(1)
    logging.info(f"Found {len(json_list_path)} JSON files in {LABELME_JSON_DIR}")

    # --- Initialize Converter ---
    converter = Labelme2Coco()

    # --- Process All Data ---
    logging.info("Starting LabelMe to COCO conversion process...")
    coco_instance = converter.generate_coco_annotation(json_list_path) # Generate annotations

    # --- Save COCO JSON ---
    if coco_instance and coco_instance.get('images'): # Check if any images were processed using .get for safety
        coco_json_path = os.path.join(SAVED_COCO_PATH, 'annotations.json') # Define output path
        converter.save_coco_json(coco_instance, coco_json_path) # Save JSON
        logging.info(f"Processed {len(coco_instance['images'])} images and {len(coco_instance.get('annotations', []))} annotations.") # Use .get for annotations too
        logging.info(f"Total unique categories created: {len(coco_instance.get('categories', []))}")
    elif coco_instance:
        logging.warning("Conversion finished, but no valid images were processed or found in the output.")
    else:
        logging.error("Conversion failed to produce a COCO instance.")


    logging.info("LabelMe to COCO conversion finished! (Category from label_4, custom labels, no segmentation)")


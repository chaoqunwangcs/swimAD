import json
import os
import logging
import re # Import regex module for potential frame number extraction

'''

output：
G:\project\20250423游泳池数据最新标签修改返回__processed\mot_label_structured\
│
└───中午\
    │
    └───1\
        │   normal_seq.txt     <- Consolidated "normal" annotations from all JSONs in 中午\1
        │   abnormal_seq.txt   <- Consolidated "abnormal" annotations from all JSONs in 中午\1
        
# Script to convert LabelMe JSON annotations to MOT challenge format files,
# maintaining the input directory structure.
# Searches for JSON files exactly two levels deep within the base directory
# (e.g., base_dir/level1/level2/*.json).
# For each 'level2' directory found, creates a corresponding 'level1/level2'
# structure in the output directory.
# Consolidates annotations from ALL JSON files within a specific 'level2'
# input directory into two single TXT files in the corresponding output directory:
# 1. normal_seq.txt: Contains labels where l2 AND l3 are NOT zero, sorted by frame.
# 2. abnormal_seq.txt: Contains labels where l2 OR l3 IS zero, sorted by frame.
# Each pair of output files represents a sequence from a specific source folder.
# Output format per line: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<category_id>,<x>,<y>,<z>
'''

# --- Configure Logging ---
# Set logging level and format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configure Paths ---
# Input JSON base directory (script will look for .json files two levels down)
json_base_dir = r"G:\project\20250423游泳池数据最新标签修改返回" # 包含TARGET_DEPTH层子目录的根路径
# Output base directory where the mirrored structure will be created
output_base_dir = r"G:\project\20250423游泳池数据最新标签修改返回__processed\mot_label_structured" # Base directory for structured output
normal_seq_filename = "normal_seq.txt"    # Filename for the consolidated normal sequence within each subfolder
abnormal_seq_filename = "abnormal_seq.txt"  # Filename for the consolidated abnormal sequence within each subfolder
TARGET_DEPTH = 2 # Process directories exactly this many levels below json_base_dir

# --- Label Parsing Function ---
def _parse_custom_label(label_str):
    """
    Parses the LabelMe custom label string "l1,l2,l3,l4" or "l1，l2，l3，l4".
    Handles both half-width (,) and full-width (，) commas.
    Returns a tuple of integers (label_1, label_2, label_3, label_4) if valid.
    Returns None if parsing fails or format is incorrect (e.g., not 4 parts).
    """
    if not label_str:
        logging.warning("Encountered empty label string.")
        return None
    normalized_label_str = label_str.replace('，', ',')
    parts = normalized_label_str.split(',')
    if len(parts) != 4:
        logging.warning(f"Label string '{label_str}' (normalized: '{normalized_label_str}') does not have 4 parts. Skipping.")
        return None
    try:
        labels = tuple(int(p.strip()) for p in parts)
        return labels
    except ValueError:
        logging.warning(f"Could not convert parts of label '{label_str}' (normalized: '{normalized_label_str}') to integers. Skipping.")
        return None

# --- Frame Number Extraction Helper ---
def extract_frame_number(filename_without_ext):
    """
    Attempts to extract a frame number (integer) from the filename.
    Tries to find the last sequence of digits. Defaults to 1 if none found.
    """
    frame_number = 1 # Default frame number
    try:
        # Find all sequences of digits in the filename
        all_digits = re.findall(r'\d+', filename_without_ext)
        if all_digits:
            # Use the last sequence found
            frame_number = int(all_digits[-1])
            logging.debug(f"Extracted frame number {frame_number} from {filename_without_ext}")
        else:
             logging.debug(f"No frame number found in {filename_without_ext}, using default {frame_number}.")
    except (ValueError, IndexError):
        logging.warning(f"Could not parse frame number from {filename_without_ext}, using default {frame_number}.")
    return frame_number

# --- Sorting Helper ---
def get_frame_number_from_mot_line(mot_line):
    """
    Helper function to extract the frame number (first element) as integer
    from a MOT line string for sorting.
    """
    try:
        return int(mot_line.split(',')[0])
    except (IndexError, ValueError):
        logging.warning(f"Could not parse frame number for sorting from line: {mot_line}. Treating as frame 0.")
        return 0 # Default to 0 if parsing fails

# --- Main Conversion Function ---
def convert_json_to_structured_mot(json_base_dir, output_base_dir, normal_fname, abnormal_fname, target_depth):
    """
    Converts JSON annotation files found at a specific depth in json_base_dir
    into consolidated MOT format TXT files, mirroring the directory structure
    in output_base_dir.

    Args:
        json_base_dir (str): Path to the root input directory.
        output_base_dir (str): Path to the root output directory.
        normal_fname (str): Filename for the normal consolidated MOT file.
        abnormal_fname (str): Filename for the abnormal consolidated MOT file.
        target_depth (int): The exact subdirectory depth to process (relative to json_base_dir).
    """
    processed_folders = 0
    total_folders_at_depth = 0
    grand_total_normal_annotations = 0
    grand_total_abnormal_annotations = 0

    logging.info(f"Starting search for JSON files at depth {target_depth} in {json_base_dir}...")

    # Normalize base paths to remove trailing separators for consistent depth calculation
    norm_json_base_dir = os.path.normpath(json_base_dir)
    base_dir_parts_count = len(norm_json_base_dir.split(os.sep))

    for root, dirs, files in os.walk(json_base_dir, topdown=True):
        norm_root = os.path.normpath(root)
        current_depth = len(norm_root.split(os.sep)) - base_dir_parts_count

        # --- Depth Check ---
        if current_depth == target_depth:
            total_folders_at_depth += 1
            logging.info(f"Processing target directory at depth {target_depth}: {root}")

            # Determine the relative path from the base directory
            relative_path = os.path.relpath(root, json_base_dir)
            # Create the corresponding output directory structure
            current_output_dir = os.path.join(output_base_dir, relative_path)
            try:
                os.makedirs(current_output_dir, exist_ok=True)
                logging.debug(f"Ensured output subdirectory exists: {current_output_dir}")
            except OSError as e:
                logging.error(f"Error creating output subdirectory {current_output_dir}: {e}. Skipping this folder.")
                # Prevent further processing within this branch if directory creation fails
                dirs[:] = [] # Stop os.walk from descending further into this branch
                continue

            # Lists to store annotations for THIS specific directory
            current_normal_annotations = []
            current_abnormal_annotations = []
            json_files_in_folder = 0
            processed_files_in_folder = 0
            error_files_in_folder = 0

            # Process JSON files ONLY in this specific directory (not subdirs)
            for filename in files:
                if filename.lower().endswith('.json'):
                    json_files_in_folder += 1
                    json_filepath = os.path.join(root, filename)
                    base_filename = os.path.splitext(filename)[0]
                    logging.debug(f"  Processing JSON file: {json_filepath}")

                    # Extract frame number from the JSON filename
                    frame_number = extract_frame_number(base_filename)

                    json_processed = False
                    try:
                        with open(json_filepath, 'r', encoding='utf-8') as f_json:
                            data = json.load(f_json)

                        img_height = data.get('imageHeight')
                        img_width = data.get('imageWidth')
                        shapes = data.get('shapes', [])

                        if img_height is None or img_width is None or not isinstance(img_height, (int, float)) or not isinstance(img_width, (int, float)) or img_height <= 0 or img_width <= 0:
                            logging.warning(f"  Skipping {filename}: Invalid or missing 'imageHeight' or 'imageWidth'.")
                            error_files_in_folder += 1
                            continue

                        annotations_in_file = 0
                        for shape in shapes:
                            label_str = shape.get('label')
                            points = shape.get('points')
                            shape_type = shape.get('shape_type')

                            if shape_type != 'rectangle' or label_str is None or points is None or len(points) != 2:
                                logging.debug(f"  Skipping invalid/non-rectangle annotation in {filename}: {shape}")
                                continue

                            parsed_labels = _parse_custom_label(label_str)
                            if parsed_labels is None:
                                continue

                            track_id = parsed_labels[0]
                            category_id = parsed_labels[3]

                            try:
                                x1, y1 = map(float, points[0])
                                x2, y2 = map(float, points[1])
                            except (ValueError, TypeError):
                                logging.warning(f"  Skipping annotation with invalid coords in {filename}: {points}")
                                continue

                            bb_left = round(min(x1, x2))
                            bb_top = round(min(y1, y2))
                            bb_width = round(abs(x1 - x2))
                            bb_height = round(abs(y1 - y2))

                            if bb_width <= 0 or bb_height <= 0:
                                logging.warning(f"  Skipping annotation with zero size in {filename}: w={bb_width}, h={bb_height}")
                                continue

                            conf = 1.0
                            world_x, world_y, world_z = -1, -1, -1

                            mot_line = f"{frame_number},{track_id},{bb_left},{bb_top},{bb_width},{bb_height},{conf},{category_id},{world_x},{world_y},{world_z}"
                            annotations_in_file += 1

                            if parsed_labels[1] == 0 or parsed_labels[2] == 0:
                                current_abnormal_annotations.append(mot_line)
                            else:
                                current_normal_annotations.append(mot_line)

                        if annotations_in_file > 0:
                            json_processed = True

                    except json.JSONDecodeError:
                        logging.error(f"  Error decoding JSON for {filename}. Skipping.")
                        error_files_in_folder += 1
                    except FileNotFoundError:
                        logging.error(f"  File not found when trying to open {json_filepath}. Skipping.")
                        error_files_in_folder += 1
                    except Exception as e:
                        logging.error(f"  Unexpected error processing {filename}: {e}", exc_info=True)
                        error_files_in_folder += 1
                    finally:
                        if json_processed:
                            processed_files_in_folder += 1

            # --- Sort and Write files for THIS directory ---
            logging.info(f"  Sorting annotations for directory: {root}")
            current_normal_annotations.sort(key=get_frame_number_from_mot_line)
            current_abnormal_annotations.sort(key=get_frame_number_from_mot_line)

            normal_output_filepath = os.path.join(current_output_dir, normal_fname)
            abnormal_output_filepath = os.path.join(current_output_dir, abnormal_fname)

            folder_normal_count = len(current_normal_annotations)
            folder_abnormal_count = len(current_abnormal_annotations)
            grand_total_normal_annotations += folder_normal_count
            grand_total_abnormal_annotations += folder_abnormal_count

            if current_normal_annotations:
                logging.info(f"  Writing {folder_normal_count} normal annotations to {normal_output_filepath}")
                try:
                    with open(normal_output_filepath, 'w', encoding='utf-8') as f_txt:
                        f_txt.write("\n".join(current_normal_annotations) + "\n")
                except IOError as e:
                     logging.error(f"  Error writing normal annotation file {normal_output_filepath}: {e}")
            else:
                 logging.info(f"  No normal annotations found for {root}.")

            if current_abnormal_annotations:
                logging.info(f"  Writing {folder_abnormal_count} abnormal annotations to {abnormal_output_filepath}")
                try:
                    with open(abnormal_output_filepath, 'w', encoding='utf-8') as f_txt:
                        f_txt.write("\n".join(current_abnormal_annotations) + "\n")
                except IOError as e:
                    logging.error(f"  Error writing abnormal annotation file {abnormal_output_filepath}: {e}")
            else:
                logging.info(f"  No abnormal annotations found for {root}.")

            logging.info(f"  Finished processing directory {root}. Found {json_files_in_folder} JSONs, processed {processed_files_in_folder}, errors {error_files_in_folder}.")
            processed_folders += 1

            # We've processed this folder at the target depth, don't go deeper
            dirs[:] = [] # Stop os.walk from descending further into this branch

        elif current_depth > target_depth:
            # If we somehow went deeper than target, stop descending further
            dirs[:] = []
            continue
        else:
            # Continue walking if not yet at target depth
            pass


    # --- Output Final Summary ---
    logging.info(f"--- Conversion Summary ---")
    logging.info(f"Searched base directory: {json_base_dir}")
    logging.info(f"Target processing depth: {target_depth}")
    logging.info(f"Found {total_folders_at_depth} directories at the target depth.")
    logging.info(f"Successfully processed {processed_folders} directories.")
    logging.info(f"Grand total normal annotations written: {grand_total_normal_annotations}")
    logging.info(f"Grand total abnormal annotations written: {grand_total_abnormal_annotations}")
    logging.info(f"Output saved in structured directories under: {output_base_dir}")


# --- Script Execution Entry Point ---
if __name__ == "__main__":
    # Run the conversion using the paths and depth defined at the start
    convert_json_to_structured_mot(json_base_dir, output_base_dir, normal_seq_filename, abnormal_seq_filename, TARGET_DEPTH)

import json
import os
import numpy as np
import cv2

# --- Configuration ---
# 1. Base input directory (will search recursively within this)
input_base_dir = r'G:\20250423游泳池数据最新标签修改返回' #<-- ADJUST AS NEEDED

# 2. Base output directory (will replicate input subfolder structure here)
output_base_dir = r'G:\20250423游泳池数据最新标签修改返回__gt_display' #<-- ADJUST OUTPUT DIR NAME AS NEEDED

# 3. Optional: Limit processing (set to None or a large number to process all)
# MAX_IMAGES_TO_PROCESS = 200
MAX_IMAGES_TO_PROCESS = None

# --- Helper Function to Create Output Directory ---
def ensure_dir(directory_path):
    """Creates a directory if it doesn't exist."""
    try:
        os.makedirs(directory_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {directory_path}: {e}")

# --- Helper Function to Read Image with Unicode Path ---
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
             # print(f"Warning: cv2.imdecode failed for {image_path}") # Optional verbose log
             return None
        return img
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading/decoding image {image_path}: {e}")
        return None

# --- Main Processing Logic ---
files_processed_count = 0
json_files_found = 0

print(f"Starting recursive search in: {input_base_dir}")
print(f"Annotated images will be saved under: {output_base_dir}")
if MAX_IMAGES_TO_PROCESS is not None:
    print(f"Processing limit set to: {MAX_IMAGES_TO_PROCESS} images")

# Use os.walk to traverse the directory tree
for root, dirs, files in os.walk(input_base_dir):
    files.sort()
    dirs.sort()

    try:
        print(f"\n--- Scanning directory: {root} ---")
    except UnicodeEncodeError:
        print(f"\n--- Scanning directory: (path contains characters not displayable in console) ---")

    found_in_dir = 0

    for filename in files:
        if MAX_IMAGES_TO_PROCESS is not None and files_processed_count >= MAX_IMAGES_TO_PROCESS:
            print(f"\nReached the processing limit of {MAX_IMAGES_TO_PROCESS}. Stopping search.")
            break

        if filename.lower().endswith('.json'):
            json_files_found += 1
            found_in_dir += 1
            json_name = filename
            base_name = os.path.splitext(json_name)[0]

            try:
                 print(f"\nProcessing JSON ({found_in_dir}): {json_name} in {root}")
            except UnicodeEncodeError:
                 print(f"\nProcessing JSON ({found_in_dir}): {json_name} in (path contains characters not displayable in console)")

            json_path_i = os.path.join(root, json_name)
            img_name_png = base_name + '.png'
            img_name_jpg = base_name + '.jpg'
            img_path_png = os.path.join(root, img_name_png)
            img_path_jpg = os.path.join(root, img_name_jpg)

            # --- Read the corresponding image using the safe function ---
            img = read_image_safe(img_path_png)
            img_name = img_name_png
            img_path_one = img_path_png

            if img is None:
                print(f"Info: PNG not found or failed to read/decode ({img_path_png}). Trying JPG...")
                img = read_image_safe(img_path_jpg)
                if img is None:
                    print(f"Error: Could not read/decode image file (tried PNG and JPG): {img_name_png} / {img_name_jpg} in {root}. Skipping.")
                    continue
                else:
                    img_name = img_name_jpg
                    img_path_one = img_path_jpg
                    print(f"Successfully loaded image: {img_name}")
            else:
                 print(f"Successfully loaded image: {img_name}")

            # --- Load JSON annotation ---
            try:
                with open(json_path_i, 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON format in file: {json_path_i}. Skipping.")
                continue
            except FileNotFoundError:
                print(f"Error: JSON file not found: {json_path_i}. Skipping.")
                continue
            except UnicodeDecodeError as e:
                 print(f"Error reading JSON file {json_path_i} with UTF-8 encoding: {e}. Skipping.")
                 continue
            except Exception as e:
                print(f"Error reading JSON file {json_path_i}: {e}. Skipping.")
                continue

            # --- Draw annotations on the image ---
            annotations_drawn = 0
            try:
                if 'shapes' not in ann_data or not isinstance(ann_data['shapes'], list):
                     print(f"Warning: 'shapes' key missing or not a list in {json_name}. Skipping drawing.")
                else:
                    for num_annotation, shape in enumerate(ann_data['shapes']):
                        if shape.get('shape_type') != 'rectangle':
                            continue

                        # *** MODIFICATION: Check for required keys: 'points', 'label', 'line_color' ***
                        required_keys = ['points', 'label', 'line_color']
                        if not all(k in shape for k in required_keys):
                            missing_keys = [k for k in required_keys if k not in shape]
                            print(f"Warning: Annotation #{num_annotation+1} in {json_name} is missing keys: {missing_keys}. Skipping annotation.")
                            continue
                        if not isinstance(shape['points'], list) or len(shape['points']) != 2:
                            print(f"Warning: Annotation #{num_annotation+1} in {json_name} has 'points' with wrong format. Skipping annotation.")
                            continue
                        # Check line_color format
                        if not isinstance(shape['line_color'], list) or len(shape['line_color']) < 3:
                             print(f"Warning: Annotation #{num_annotation+1} in {json_name} has 'line_color' with wrong format (expected list of >=3 numbers). Skipping annotation.")
                             continue

                        try:
                            # Extract points
                            points = shape['points']
                            if not (len(points[0]) == 2 and len(points[1]) == 2):
                                print(f"Warning: Invalid coordinate structure in 'points' for annotation #{num_annotation+1} in {json_name}. Skipping.")
                                continue
                            xmin = int(points[0][0])
                            ymin = int(points[0][1])
                            xmax = int(points[1][0])
                            ymax = int(points[1][1])

                            if xmin >= xmax or ymin >= ymax:
                                print(f"Warning: Invalid coordinates (min >= max) in annotation #{num_annotation+1} in {json_name}: ({xmin},{ymin}) ({xmax},{ymax}). Skipping.")
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
                                print(f"Warning: Invalid number format in 'line_color' for annotation #{num_annotation+1} in {json_name}. Using default color. Skipping annotation.")
                                continue # Skip this annotation if color is invalid

                            # Clamp values to 0-255
                            r = max(0, min(255, r))
                            g = max(0, min(255, g))
                            b = max(0, min(255, b))

                            bgr_color = (b, g, r) # OpenCV uses BGR order

                        except (ValueError, TypeError, IndexError, KeyError) as e:
                            print(f"Warning: Invalid/missing data in annotation #{num_annotation+1} in {json_name}: {e}. Skipping annotation.")
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

                if annotations_drawn > 0:
                    print(f"Finished drawing {annotations_drawn} rectangle annotations for {img_name}")
                elif 'shapes' in ann_data and isinstance(ann_data['shapes'], list):
                     print(f"No valid 'rectangle' annotations with required keys found in {json_name}.")

                # --- Determine and Create Output Path ---
                relative_path = os.path.relpath(root, input_base_dir)
                if relative_path == '.':
                    output_subdir = output_base_dir
                else:
                    output_subdir = os.path.join(output_base_dir, relative_path)

                ensure_dir(output_subdir)
                output_filename = os.path.join(output_subdir, img_name)

                # --- Save the drawn picture ---
                try:
                    is_success, im_buf_arr = cv2.imencode(os.path.splitext(output_filename)[1], img)
                    if is_success:
                        with open(output_filename, 'wb') as f:
                            f.write(im_buf_arr.tobytes())
                        print(f"Successfully saved annotated image to: {output_filename}")
                        files_processed_count += 1
                    else:
                         print(f"Error: Failed to encode image for saving to: {output_filename}")

                except Exception as e:
                    print(f"Error encoding or writing image file {output_filename}: {e}")

            except Exception as e:
                print(f"An unexpected error occurred while processing annotations or saving image for {json_name}: {e}")

    if MAX_IMAGES_TO_PROCESS is not None and files_processed_count >= MAX_IMAGES_TO_PROCESS:
        break

print(f"\n--- Finished processing ---")
print(f"Total JSON files found: {json_files_found}")
print(f"Total images processed and saved: {files_processed_count}.")


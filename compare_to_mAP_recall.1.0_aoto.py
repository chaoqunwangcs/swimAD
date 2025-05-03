import sys
import os
import json # Required for manual loading
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import traceback # For detailed error reporting

# --- Configuration ---
# Use forward slashes or raw strings for paths
ground_truth_file = r'G:\project\ML\drown\20250412return\temp_json_to_coco\100_annotations.json'
detection_file = 'G:/project/ML/drown/20250412return/1_dinox/100_annotations_coco.json'

# Ensure this matches the data in your detection file ('segm' or 'bbox')
iou_type = 'bbox' # Set evaluation type to bounding box
# --- End Configuration ---

def evaluate_coco(gt_file, dt_file, eval_type='bbox'):
    """
    Loads ground truth and detection files using UTF-8 encoding,
    runs COCO evaluation for the specified eval_type, and prints summary.
    Handles detection files that are dicts and adds dummy scores if missing.
    """
    # --- Input Validation ---
    if not os.path.exists(gt_file):
        print(f"Error: Ground truth file not found at '{gt_file}'", file=sys.stderr)
        return
    if not os.path.exists(dt_file):
        print(f"Error: Detection file not found at '{dt_file}'", file=sys.stderr)
        return
    if eval_type not in ['bbox', 'segm']:
        print(f"Error: Invalid eval_type '{eval_type}'. Choose 'bbox' or 'segm'.", file=sys.stderr)
        return

    # Initialize variables
    cocoGt = None
    gt_data = None
    cocoDt = None
    detections_list = None
    loaded_data = None
    cocoEval = None

    try:
        # --- Load Ground Truth with explicit UTF-8 encoding ---
        print(f"Loading ground truth annotations from: {gt_file} using UTF-8")
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        print("Ground truth JSON loaded successfully.")

        # Validate basic GT structure
        if not isinstance(gt_data, dict) or 'annotations' not in gt_data or 'images' not in gt_data:
             raise ValueError(f"Ground truth file '{gt_file}' does not appear to be a valid COCO annotation format.")

        # Create and index the ground truth COCO object
        cocoGt = COCO()
        cocoGt.dataset = gt_data
        cocoGt.createIndex()
        print("Ground truth COCO object created and indexed.")
        print(f"Number of images in Ground Truth: {len(cocoGt.getImgIds())}")


        # --- Load Detections with explicit UTF-8 encoding ---
        print(f"\nLoading detection results from: {dt_file} using UTF-8")
        with open(dt_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        print("Detection results JSON loaded successfully.")

        # --- Adapt and Validate Detection Format ---
        detections_list = None
        added_dummy_scores = False

        if isinstance(loaded_data, list):
            detections_list = loaded_data
            print("Detection file contains a JSON list (expected format).")
        elif isinstance(loaded_data, dict) and 'annotations' in loaded_data:
            detections_list = loaded_data.get('annotations', [])
            print(f"Warning: Detection file '{dt_file}' was a JSON dictionary. Extracted list from 'annotations' key.", file=sys.stderr)
            if not isinstance(detections_list, list):
                 raise ValueError(f"The 'annotations' key in '{dt_file}' did not contain a list.")
            print(f"Extracted {len(detections_list)} annotations to treat as detections.")
        else:
            raise ValueError(f"Detection file '{dt_file}' should contain a JSON list or a JSON dict with an 'annotations' key.")

        # --- Check for and add missing 'score' key ---
        if not isinstance(detections_list, list):
             raise ValueError("Failed to obtain a list of detections.")

        if detections_list:
            num_detections_processed = 0
            for detection in detections_list:
                if not isinstance(detection, dict):
                     raise ValueError(f"Item {num_detections_processed} in detection list should be a dict.")
                if 'score' not in detection:
                    detection['score'] = 1.0 # Add dummy score
                    added_dummy_scores = True
                num_detections_processed += 1

            if added_dummy_scores:
                print("Warning: Added dummy score=1.0 to detections missing the 'score' key. Evaluation results (AP) may not be meaningful.", file=sys.stderr)

            # --- Basic validation on first item ---
            first_item = detections_list[0]
            required_keys = ['image_id', 'category_id', 'score']
            if eval_type == 'bbox':
                required_keys.append('bbox')
                if 'bbox' not in first_item: raise ValueError(f"First detection item missing 'bbox' key for 'bbox' evaluation.")
                if not isinstance(first_item['bbox'], list) or len(first_item['bbox']) != 4: raise ValueError(f"Key 'bbox' in first item is not a list of 4 numbers.")
            # Note: No check for 'segmentation' needed if eval_type is 'bbox'

            missing_keys = [key for key in required_keys if key not in first_item]
            if missing_keys: raise ValueError(f"First detection item missing required key(s): {missing_keys}")
        else:
            print("Warning: The detection list is empty.", file=sys.stderr)


        print("Detection results format adapted and appears valid (basic checks passed).")

        # --- Load Detections into COCO result object ---
        if cocoGt is None:
             raise ValueError("cocoGt object was not initialized correctly.")
        cocoDt = cocoGt.loadRes(detections_list)
        print("Detection results loaded into COCO result object.")

        # --- Initialize Evaluation ---
        print(f"\nInitializing COCO evaluation for type: '{eval_type}'")
        cocoEval = COCOeval(cocoGt, cocoDt, iouType=eval_type)

        # --- Optional: Modify Evaluation Parameters ---
        # See explanation in chat for examples like:
        # cocoEval.params.catIds = [...]
        # cocoEval.params.iouThrs = [...]
        # cocoEval.params.imgIds = [...]

        # --- Run Evaluation ---
        print("\nRunning evaluation...")
        cocoEval.evaluate()
        print("Accumulating results...")
        cocoEval.accumulate()

        # --- Summarize Results ---
        print("\nEvaluation Summary:")
        cocoEval.summarize()

    # --- Exception Handling ---
    except FileNotFoundError as e:
        print(f"\nError: File not found.", file=sys.stderr)
        print(e, file=sys.stderr)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"\nError: Failed to read or decode JSON from a file.", file=sys.stderr)
        print(f"Specific error: {e}", file=sys.stderr)
        if 'gt_data' not in locals() or gt_data is None: print(f"Potential issue in: {gt_file}", file=sys.stderr)
        elif 'loaded_data' not in locals() or loaded_data is None : print(f"Potential issue in: {dt_file}", file=sys.stderr)
    except ValueError as e:
        print(f"\nError: Problem with data format or content.", file=sys.stderr)
        print(f"Specific error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"\nAn unexpected error occurred during COCO evaluation:", file=sys.stderr)
        print(f"Specific error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # --- Cleanup ---
    finally:
        # Safely attempt cleanup
        try:
            variables_to_delete = ['cocoGt', 'gt_data', 'cocoDt', 'detections_list', 'loaded_data', 'cocoEval']
            existing_vars = [var for var in variables_to_delete if var in locals()]
            for var_name in existing_vars:
                 del locals()[var_name]
        except Exception: pass
        print("\nEvaluation script finished.")

# --- Main Execution ---
if __name__ == "__main__":
    # Pass the configured iou_type to the function's eval_type parameter
    evaluate_coco(ground_truth_file, detection_file, eval_type=iou_type)

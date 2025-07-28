from ultralytics.data.utils import visualize_image_annotations

label_map = {  # Define the label map with all annotated class labels.
  0: "ashore",
  1: "above",
  2: "under"
}

# Visualize
visualize_image_annotations(
    "/home/chaoqunwang/swimAD/dataset/dataset_v20250604/afternoon_v1/1/afternoon_1_0001.jpg",  # Input image path.
    "/home/chaoqunwang/swimAD/data_transfer/yolo/dataset_v20250604_seglabel/afternoon_v1/1/afternoon_1_0001.txt",  # Annotation file path for the image.
    label_map,
)
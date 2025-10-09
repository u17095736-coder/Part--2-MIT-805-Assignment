# Part--2-MIT-805-Assignment

This project implements a traffic sign detection system using the Mapillary Traffic Sign Dataset and the DFG Traffic Sign Dataset. The pipeline integrates Big Data preprocessing with Apache Spark (MapReduce) and deep learning with YOLOv5 to build a scalable, real-world traffic sign detection model. The work was developed as part of a university assignment focusing on MapReduce, visualization, and object detection.


### Overview
- **Goal**: Build a traffic sign classifier from object detection annotations by converting bounding boxes into class-specific image crops and training a YOLOv8 classification model.
- **Approach**: PySpark for scalable XML parsing and image cropping; Ultralytics YOLOv8n-cls for training and evaluation.
- **Result**: 76-class classifier achieving ~**86.6% top-1** and ~**99.6% top-5** accuracy on the test split after 5 epochs on CPU.

### Dataset
- **Source**: Kaggle `nomihsa965/traffic-signs-dataset-mapillary-and-dfg` (via `kagglehub`).
- **Size**: ~11.6 GB download.
- **Contents discovered**:
  - 19,346 XML annotation files
  - 49,905 images
  - 30,557 valid labeled object instances after filtering

### Pipeline
1. Download dataset using `kagglehub`.
2. Parse PASCAL-VOC-style XML annotations with PySpark into records: `(image_filename, class, bbox, image_wh)`.
3. Validate and filter records (bbox bounds and positive image dimensions).
4. Join image paths and compute deterministic train/val/test split via MD5 hash of the image key.
5. Crop each bounding box to a 224×224 RGB JPEG and save under split/class directories.
6. Train YOLOv8n-cls on the generated classification dataset.
7. Evaluate on validation and test splits; export sample predictions and run metadata.

Generated dataset structure (created by the notebook):
- `./data_cls/train/<class>/*.jpg`
- `./data_cls/val/<class>/*.jpg`
- `./data_cls/test/<class>/*.jpg`

Split sizes detected by YOLO during training:
- **Train**: 21,250 images across 76 classes
- **Val**: 4,589 images across 76 classes
- **Test**: 4,718 images across 76 classes

### Model and Training
- **Base model**: `yolov8n-cls.pt` (Ultralytics)
- **Image size**: 224
- **Epochs**: 5 (demo setting; increase for better accuracy)
- **Device**: CPU (GPU recommended for speed and improved performance)
- **Environment (observed)**:
  - Ultralytics: 8.3.207
  - PyTorch: 2.8.0
  - PySpark: 3.5.1
  - Python: 3.12.11

### Results
- **Validation**: top-1 ≈ 0.865, top-5 ≈ 0.997
- **Test**: top-1 ≈ 0.866, top-5 ≈ 0.996
- **Example predictions (on 224×224 crops)**:
  - `warning--curve-right 0.69`
  - `regulatory--keep-right 0.71`
  - `regulatory--maximum-speed-limit 0.99`

Artifacts produced:
- Ultralytics run directories under `runs/classify/...` (e.g., `train`, `val`, `val2`).
- Prepared dataset at `./data_cls`.
- Metadata at `./artifacts/run_info.json` (data root and last run path).

### Quickstart
```bash
# 1) Environment
pip install kagglehub pyspark ultralytics pillow pandas scikit-learn

# 2) Execute the notebook end-to-end (downloads ~11.6GB)
#    - Downloads dataset (kagglehub)
#    - Builds classification dataset with Spark
#    - Trains YOLOv8n-cls for 5 epochs
#    - Evaluates and writes artifacts

# 3) After training, metrics and run directories are printed
#    Ultralytics runs live under: ./runs/classify/
```

### Repository Layout (suggested)
- `Road_sign_detection.ipynb`: end-to-end pipeline for data prep, training, and evaluation.
- `data_cls/`: generated classification dataset (train/val/test by class).
- `runs/classify/`: Ultralytics training/eval runs.
- `artifacts/run_info.json`: metadata with pointers to data root and last run.

### Notes and Considerations
- The dataset is large; ensure adequate disk space and bandwidth.
- For better accuracy, train longer and/or use a larger model (e.g., `yolov8s-cls.pt`) with a GPU.
- Class imbalance likely exists; consider longer training, balanced sampling, or augmentation strategies.

### References
- Ultralytics YOLOv8: https://docs.ultralytics.com
- Kaggle dataset: https://www.kaggle.com/datasets/nomihsa965/traffic-signs-dataset-mapillary-and-dfg
- PySpark: https://spark.apache.org/docs/latest/api/python/


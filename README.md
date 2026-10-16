# Smart Helmet Detection using YOLOv8

Milestone 2 prototype for object detection of motorcycle riders **with helmets**, riders **without helmets**, and **licence plates** using a fine-tuned Ultralytics YOLOv8n model.

## Results (held-out test set)

Evaluation on **154 test images / 237 annotated objects** using the standard Ultralytics evaluation settings:

| Metric | Value |
|---|---:|
| Precision | 0.7801 |
| Recall | 0.7410 |
| mAP@0.50 | 0.7967 |
| mAP@0.50:0.95 | 0.4836 |

Class-specific mAP@0.50: With Helmet **0.779**, Without Helmet **0.711**, licence **0.900**. These metrics reflect object detection, **not classification accuracy**.

## Project files

- `notebooks/Helmet_Detection_Milestone_2.ipynb` — Kaggle notebook source, with outputs cleared to keep it easy to review; includes exploration, preprocessing, training, validation, predictions, and ZIP export.
- `weights/best.pt` — trained baseline weights.
- `results/` — training graphs, test confusion matrix at 0.25 confidence, sample predictions, and dataset figures.
- `config/data.example.yaml` — portable dataset path template.
- `requirements.txt` — Python dependencies.

## Dataset and preprocessing

Roboflow **Helmet Detection v6 balanced dataset**, distributed in YOLOv8 annotation format. The exported dataset's `data.yaml` lists a Roboflow project and licensing information; preserve and credit those details when redistributing. Original: **1,533 images**, split into 1,070 training, 309 validation, 154 testing images. Cleaning removed **one exact duplicate validation image**, yielding **1,532 images**, 2,501 valid annotations, and 9 empty/background label files. Bounding boxes were clipped to valid image coordinates and reformatted; the audit script rewrote all 2,502 original annotation lines (this is **not** a count of 2,502 incorrect labels). Image resizing/normalisation and augmentation were handled by the Ultralytics training pipeline.

**Dataset images are not included in this repository.** Obtain the original YOLOv8-export dataset from the relevant Roboflow project and attach it as a Kaggle input. The notebook was developed on Kaggle and references its original absolute path (`/kaggle/input/datasets/sadiiamjad/helmet-detection-v6-balanced-dataset-yolov8`); replace that path if your attachment differs. For local execution, update the paths in the notebook and the provided YAML template.

## Run on Kaggle

1. Create a Kaggle notebook, attach the YOLOv8 dataset, and enable a GPU (Tesla T4 used for this experiment) and internet access for package installation.
2. Install the dependencies: `pip install -r requirements.txt` (or install `ultralytics` in Kaggle).
3. Open `notebooks/Helmet_Detection_Milestone_2.ipynb` and execute cells in order. **Review the dataset path before running.**
4. The notebook makes a working copy under `/kaggle/working/helmet_dataset`, validates and cleans labels, creates `/kaggle/working/helmet_data.yaml`, and trains YOLOv8n.
5. Training configuration: **30 epochs**, **640 px**, **batch 16**, **AdamW**, **lr0=0.001**, GPU 0, with geometric and mosaic augmentation. Checkpoints are written to `/kaggle/working/helmet_runs/yolov8n_baseline/weights/`.
6. The notebook evaluates the best checkpoint on the held-out test set and generates predictions and graphs. Alternatively, use `weights/best.pt` for inference without retraining.

**Caution:** The notebook is a record of the original experiment, not a fully portable one-click script. Some cells modify the working dataset, and rerunning them with a pre-existing working directory can change outcomes. For a clean reproduction, start in a fresh Kaggle session, verify your dataset source, and run in order. Do not use test results to tune the baseline retrospectively.

## Quick inference example

```python
from ultralytics import YOLO
model = YOLO("weights/best.pt")
results = model.predict(source="path/to/image.jpg", imgsz=640, conf=0.25)
for result in results:
    result.show()
```

## Limitations

Helmet/no-helmet predictions are less reliable than licence-plate detection on this dataset, and some source annotations/images may be noisy or outside motorcycle-road scenes. Model outputs are for an academic prototype and should not be used as sole evidence of traffic violations.

## Milestone 2 coverage

Dataset exploration and class distribution; label validation and duplicate checks; cleaned train/validation/test partitions; YOLOv8n baseline training; precision/recall/mAP evaluation; confusion matrix and prediction demonstrations.

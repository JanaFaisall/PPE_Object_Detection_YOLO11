# Construction PPE Detection with YOLO11n

**Computer Vision Bootcamp – Saudi Digital Academy × atomcamp Arabia**

Project Date: July 2026

A computer vision project that detects personal protective equipment (PPE) and safety violations in construction-site images.  
The model was fine-tuned using **YOLO11n** to identify multiple objects, locate them with bounding boxes, and evaluate performance using per-class metrics and a confusion matrix.

---

## 🔍 Project Overview

The system answers two questions:

1. **What object is present?**
2. **Where is it located?**

Unlike image classification, which predicts one label for the full image, object detection can identify several objects and their locations in the same scene.

The model detects 11 classes:

`helmet` · `gloves` · `vest` · `boots` · `goggles` · `none` · `Person` · `no_helmet` · `no_goggle` · `no_gloves` · `no_boots`

---

## ✏️ My Work

- Prepared the dataset structure and YOLO configuration file
- Audited the number of images and annotation files
- Read and decoded YOLO bounding-box labels
- Visualized ground-truth boxes using OpenCV and Matplotlib
- Implemented Intersection over Union (IoU) manually
- Fine-tuned a pretrained YOLO11n model
- Evaluated Precision, Recall, mAP, and per-class performance
- Analyzed the normalized confusion matrix
- Ran predictions on unseen validation images
- Investigated weak classes and dataset limitations

---

## 📂 Dataset

| Split | Images | Label Files |
|---|---:|---:|
| Training | 1,132 | 1,142 |
| Validation | 143 | 143 |
| Test | 141 | 141 |

The training split contains more label files than images, which suggests that some annotation files may not have a matching image and should be reviewed during data cleaning.

---

## ⚙️ Model Training

| Setting | Value |
|---|---|
| Model | YOLO11n |
| Training Method | Transfer Learning |
| Epochs | 15 |
| Image Size | 640 × 640 |
| Batch Size | 8 |
| Early-Stopping Patience | 5 |
| Hardware | CPU |
| Prediction Threshold | 0.35 |

A pretrained model was used instead of training from scratch, allowing the model to reuse previously learned visual features and adapt them to the PPE dataset.

---

## 📊 Overall Results

| Metric | Score |
|---|---:|
| Precision | 0.803 |
| Recall | 0.517 |
| mAP@0.50 | 0.581 |
| mAP@0.50:0.95 | 0.289 |

The model achieved good overall precision, meaning that many reported detections were correct.  
However, the lower recall shows that the model still missed a considerable number of real objects.

---

## 🚧 Main Challenge: Class Imbalance

The dataset is **imbalanced**, meaning that some classes appear much more frequently than others.

The model had more opportunities to learn common classes such as `Person`, `helmet`, and `vest`, while missing-PPE classes had fewer examples.

| Class | Validation Instances | Recall |
|---|---:|---:|
| Person | 239 | 0.908 |
| helmet | 201 | 0.833 |
| vest | 171 | 0.819 |
| no_helmet | 45 | 0.333 |
| no_goggle | 41 | 0.000 |
| no_gloves | 56 | 0.089 |
| no_boots | 4 | 0.000 |

The confusion matrix shows that many minority-class objects were predicted as background:

- `no_goggle`: 100% missed as background
- `no_gloves`: 91% missed as background
- `no_boots`: 75% missed as background
- `no_helmet`: 60% missed as background

These results suggest that class imbalance was a major reason for the weak performance of the missing-PPE classes. Other possible factors include small object sizes, annotation quality, visually overlapping classes, and limited training time.

---

## 🛠️ Possible Improvements

- Collect more examples for the minority classes
- Review unmatched images and label files
- Check bounding boxes and class IDs for annotation errors
- Apply targeted augmentation to rare classes
- Oversample images containing missing-PPE cases
- Use copy-paste augmentation for small safety items
- Reconsider the ambiguous `none` class
- Train for more epochs using a GPU
- Compare YOLO11n with a larger YOLO model
- Re-evaluate using per-class Recall and mAP

Increasing model size alone is not enough when the main limitation comes from the data.

---

## 📈 Training Results

The losses generally decreased during training, while Precision, Recall, and mAP improved across the 15 epochs.

![Training Results](README_IMAGES/results.png)

---

## 🧩 Confusion Matrix

The normalized confusion matrix helped reveal which classes were detected correctly and which classes were frequently missed as background.

![Normalized Confusion Matrix](README_IMAGES/confusion-matrix.png)

---

## 📚 Four-Day Learning Summary

### Day 1 – Computer Vision Foundations
- Pixels, RGB channels, grayscale images, and image shapes
- Image preprocessing and normalization
- Histograms and basic OpenCV operations
- CNN foundations and dataset preparation

### Day 2 – Classification and Object Detection
- CNNs and feature maps
- Transfer Learning and fine-tuning
- Image classification vs. object detection
- Bounding boxes and YOLO

### Day 3 – Model Evaluation
- Confusion matrices
- Precision, Recall, F1-score, IoU, and mAP
- Overfitting and class imbalance
- Failure-case and robustness analysis

### Day 4 – Responsible AI and Deployment
- Fairness, interpretability, privacy, and safety
- Dataset bias and responsible evaluation
- Edge vs. cloud deployment
- Latency, FPS, and real-time inference concepts

---

## 🧰 Tools & Libraries

- Python
- Google Colab
- Ultralytics YOLO
- OpenCV
- PyTorch
- NumPy
- Matplotlib
- PyYAML
- Google Drive

---

## ▶️ How to Run

### Step 1: Install Ultralytics

```bash
pip install -U ultralytics
```

### Step 2: Open the notebook

Upload the notebook to Google Colab:

```text
PPE_Object_Detection_YOLO11.ipynb
```

### Step 3: Prepare the dataset

Use the standard YOLO folder structure:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### Step 4: Update the project path

Change the Google Drive project path inside the notebook to match your dataset location.

### Step 5: Run all cells

The notebook will:

1. Inspect the dataset
2. Visualize annotations
3. Train YOLO11n
4. Validate the best checkpoint
5. Generate metrics and confusion matrices
6. Run predictions

---

## 📁 Repository Structure

```text
.
├── 02_Object_Detection_YOLO_STUDENT_Jana.ipynb
├── README.md
└── README_IMAGES/
    ├── results.png
    ├── confusion.png
    └── prediction_examples.png
```

---

## ⚠️ Notes

- This is an educational Computer Vision Bootcamp project
- The model was trained for 15 epochs using CPU
- The current model is not ready for real-world safety deployment
- A larger and more balanced dataset is required before deployment
- The main goal was not only to train a model, but to understand where it fails and why

---

## 💡 Key Takeaway

A good overall score does not mean that every class performs well.

This project showed how a model can successfully detect common objects while almost completely missing underrepresented safety violations.  
For this reason, dataset auditing, per-class evaluation, and confusion-matrix analysis are just as important as training the model itself.

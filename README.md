# Brain Tumor Detection Using Deep Learning

## Project Overview

This project focuses on detecting brain tumors from MRI scan images using Deep Learning and Transfer Learning techniques. The model is trained to classify MRI images into two categories:

- Tumor Present (Yes)
- No Tumor (No)

The project uses TensorFlow and the pre-trained MobileNetV2 architecture to achieve high accuracy in medical image classification.

The main objective of this project is to build an intelligent AI-based system that can assist medical professionals in early brain tumor detection using MRI scans.

---

# Features

- MRI brain scan image classification
- Transfer Learning using MobileNetV2
- Fine-tuning deep learning models
- Data preprocessing and augmentation
- Performance evaluation using multiple metrics
- Confusion Matrix and ROC Curve visualization
- Grad-CAM implementation for explainable AI
- Threshold-based prediction analysis

---

# Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# Dataset

The dataset contains MRI images categorized into:

- Yes → Brain Tumor
- No → No Brain Tumor

Dataset downloaded using:

```python
kagglehub.dataset_download(
    "navoneel/brain-mri-images-for-brain-tumor-detection"
)

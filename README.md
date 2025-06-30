# 🧠 Enterprise Computer Vision Projects

This repository showcases a collection of applied computer vision projects developed using **TensorFlow**, **Keras**, and **OpenCV**, focused on real-world tasks such as image classification, face detection, model interpretability, and transfer learning. All projects were executed in **Jupyter Notebooks hosted on AWS EC2 virtual machines** as part of the **AI and Data Analytics Strategy** course.

---

## 📁 Projects Overview

### 1. 📷 Image Classification with CNN using Keras
- Built a custom **Convolutional Neural Network (CNN)** architecture using `TensorFlow/Keras`
- Preprocessed image data using `img_to_array`, `load_img`, and pixel normalization
- Applied **data augmentation** for generalization
- Trained, validated, and visualized results using `matplotlib`

### 2. 😊 Face Detection and Recognition using OpenCV
- Used `OpenCV`’s Haar Cascade classifiers for **real-time face detection**
- Captured frames from webcam using `cv2.VideoCapture`
- Processed frames with grayscale conversion, edge detection, and bounding boxes

### 3. 🧠 Transfer Learning for Image Classification (VGG/ResNet)
- Leveraged pretrained models from `tensorflow.keras.applications` (e.g., ResNet, VGG)
- Fine-tuned model layers for domain-specific classification tasks
- Integrated `callbacks` like EarlyStopping and learning rate schedulers
- Used `decode_predictions` for model interpretability

### 4. 🎯 Real-Time Object Detection with LIME & Explainability
- Trained CNNs for classification and interpreted predictions using `LIME`
- Visualized regions contributing to predictions via `skimage.segmentation.mark_boundaries`
- Explained model outputs on test images using `lime_image`

---

## 🧰 Libraries & Tools Used

| Category              | Libraries                                                                 |
|-----------------------|---------------------------------------------------------------------------|
| 🧠 Deep Learning       | `tensorflow`, `keras`, `tensorflow_datasets`                              |
| 🖼️ Image Processing    | `opencv-python (cv2)`, `skimage`, `matplotlib`, `PIL`                     |
| 🧪 Modeling Utilities  | `models`, `layers`, `losses`, `metrics`, `callbacks`, `schedules`         |
| 📊 Data Handling       | `pandas`, `numpy`, `LabelBinarizer`, `train_test_split`                  |
| 🧠 Transfer Learning   | `tensorflow.keras.applications` (ResNet, VGG), `preprocess_input`         |
| 🔍 Model Explainability| `lime`, `lime_image`, `skimage.segmentation.mark_boundaries`             |

---

## 📌 Key Techniques Applied

- CNN model design & training
- Transfer learning (fine-tuning with ResNet/VGG)
- Real-time face detection with OpenCV
- Data augmentation pipelines
- LIME-based explainability overlays
- Evaluation using accuracy, MSE, LIME visualizations

---

## 🖥️ Environment

> ✅ All models were built using **Python** in  
> ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter) and executed on **AWS EC2 Virtual Machines**

---
## 🖥️ Environment Setup

All notebooks in this repository were developed and executed in a **cloud-based compute environment** powered by:

- 🖥️ **AWS EC2 Virtual Machines** (Ubuntu)
- 📒 **Jupyter Notebook Interface**
- ⚙️ **Python 3.x** with GPU acceleration (when needed)
- 📦 Deep learning libraries installed directly on the VM for high-performance image processing tasks

> This cloud-based setup enabled efficient training of deep learning models and real-time computer vision processing for all projects in this repository.

## 📂 Repository Structure


---

## ✅ Outcomes

- Built a suite of computer vision solutions using deep learning
- Leveraged AWS VM infrastructure for compute-intensive training
- Implemented and explained models for business-relevant tasks
- Enhanced model transparency using interpretable AI frameworks

---

## 📬 Contact

Created by **Balbir Singh**  
📧 bsingh73@asu.edu | [LinkedIn](https://www.linkedin.com/in/balbir-singh27/)


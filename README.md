# Improving Plant Disease Classification Using Realistic Data Augmentation (2024 Paper Reproduction)

This repository contains the baseline replication framework for **Assignment 02** in Deep Learning (Fall 2026) at Bahria University.

## 👥 Group Members
* **Komail Nazir** (Enrollment: 01-134232-219)
* **Ameer Abdullah** (Enrollment: 01-134232-262)

## 📌 Project Overview
This project successfully reproduces the baseline architecture described in the 2024 study *"Improving Plant Disease Classification Using Realistic Data Augmentation"*. The system trains a deep **ResNet50** Convolutional Neural Network backbone on leaf imagery to benchmark model robustness against geometric and color transformations under field conditions.

---

## 💻 Environment & Hardware Specifications
* **Cloud Platform:** Google Colab
* **Hardware Accelerator:** NVIDIA T4 Tensor Core GPU (16 GB VRAM)
* **System Hardware:** Intel Xeon CPU @ 2.20GHz | 13 GB System RAM
* **Software Environment:** Python 3.10+, TensorFlow 2.21.0, Keras, NumPy, Matplotlib, Scikit-Learn

---

## 📊 Evaluation & Verification Results

### Metric Matrix Comparison Table

| Evaluation Metric | Original Paper (Islam et al., 2024) | Our Reproduced Model | Performance Difference ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Training Accuracy** | 98.2% | **94.41%** | **-3.79%** |
| **Validation Accuracy** | 96.2% | **81.84%** | **-14.36%** |
| **Final Loss Value** | 0.084 | **0.7466** | **+0.6626** |

---

## 🚀 Execution & Setup Instructions

To execute this replication code pipeline inside Google Colab:

1. Open a new notebook session and select an **NVIDIA T4 GPU** runtime configuration under *Runtime -> Change runtime type*.
2. Run the data ingestion cell block to automatically fetch and unpack the target dataset directory:
   ```bash
   !wget --no-check-certificate "https://github.com" -O plantvillage.zip
   !unzip -q plantvillage.zip
   ```
3. Initialize the `ImageDataGenerator` pipeline configuration to parse training and validation data paths with horizontal/vertical flipping and color shifts.
4. Execute the optimization loop by fitting the model over `5` epochs using the Adam optimizer configuration ($LR = 0.001$).

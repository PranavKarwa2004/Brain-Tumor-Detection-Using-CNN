# Brain Tumor Detection Using CNN

## 👥 Contributors
- [Pranav Karwa](https://github.com/PranavKarwa2004)
- [Nabhya Sharma](https://github.com/NabhyaIoT2026)
- [Aniruddha Bolakhe](https://github.com/AniruddhaBolakhe)

---

## 📌 Overview
This project focuses on **classification of brain tumor MRI images** using **Convolutional Neural Networks (CNNs)**.  
Multiple deep learning architectures including **InceptionV3, ResNet50, VGG16, and a custom AlexNet** have been trained and evaluated to distinguish between tumor and non-tumor images.  

The aim is to provide a **robust, automated diagnostic tool** to support healthcare professionals in early and accurate tumor detection.

---

## ✨ Features
- **Deep Learning-Based Detection** – Utilizes state-of-the-art CNNs for accurate classification.  
- **Multiple Models** – Fine-tuned pretrained models (InceptionV3, ResNet50) and custom-built architectures (VGG16, AlexNet).  
- **Dataset Handling** – Automated preprocessing with resizing, normalization, augmentation, and dataset splitting.  
- **Performance Evaluation** – Includes accuracy, precision, recall, F1-score, confusion matrices, and a **custom weighted score** with special focus on "no tumor" precision.  
- **User-Friendly Workflow** – Fully documented **Google Colab Notebook** with reproducible pipelines and visualization tools.  

---

## 🗂 Dataset
The dataset used consists of **MRI images labeled into four categories**:
- `glioma_tumor`
- `meningioma_tumor`
- `no_tumor`
- `pituitary_tumor`

### 🔄 Preprocessing Steps
- Conversion to grayscale/RGB as required.  
- Image resizing (**224×224** or **512×512**, depending on model).  
- Pixel normalization to improve convergence.  
- Data augmentation: random flips, rotations, scaling.  
- Splitting into **training, validation, and testing sets**.

---

## 🏗 Model Architectures

### 🔹 1. InceptionV3
- Extracts **multi-scale features** using inception modules.  
- Pretrained on **ImageNet**, then fine-tuned.  
- Outperforms VGG16 in terms of **accuracy and custom score**.  

### 🔹 2. ResNet50
- Deep residual connections to prevent vanishing gradients.  
- Transfer learning with **gradual unfreezing** of layers.  
- Achieved **best results** with fine-tuning and augmentation:
  - Nearly **98% custom score**  
  - **Perfect precision for "no tumor" detection**.  

### 🔹 3. VGG16
- Classic **16-layer CNN architecture**.  
- Trained **from scratch** (no pretrained weights used).  
- Performs reasonably well but more prone to **overfitting**.  

### 🔹 4. AlexNet (Custom)
- Implemented **from scratch** as an experiment.  
- Achieves good results but is **computationally heavy**.  
- Inferior to ResNet50 and InceptionV3 in overall performance.  

---

## 📦 Dependencies
Make sure the following Python packages are installed:

- `tensorflow` / `keras`  
- `numpy`  
- `matplotlib`  
- `pandas`  
- `seaborn`  
- `scikit-learn`  
- `yellowbrick`  

---

## ⚙️ Model Training & Evaluation

### 🚀 Training Workflow
1. Mount Google Drive and load dataset (`Training.zip`, `Testing.zip`).  
2. Extract, preprocess, augment, and normalize images.  
3. Split dataset into **training, validation, and test sets**.  
4. Train models with:
   - Early stopping  
   - Checkpointing  
   - Data caching and prefetching  

### 📊 Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Custom Weighted Score.  
- **Visualization Tools**:  
  - Confusion Matrices  
  - Learning Curves (Loss/Accuracy)  
  - Metric Summaries  

---

## 📈 Results

| Model         | Accuracy | No Tumor Precision | Custom Score |
|---------------|----------|-------------------|--------------|
| InceptionV3   | 0.92     | 0.91              | 0.91         |
| ResNet50      | 0.93     | 0.95              | 0.94         |
| **ResNet50 FT** | **0.96** | **1.00**          | **0.98**     |
| VGG16         | 0.89     | 0.83              | 0.86         |
| AlexNet       | 0.91     | 0.87              | 0.89         |

👉 **ResNet50 (fine-tuned + augmented) delivers the best performance** with perfect "no tumor" detection.

---

## 🔮 Future Improvements
- Incorporate **lightweight models** like EfficientNet or MobileNet.  
- Deploy as a **web application** for real-time diagnosis.  
- Expand dataset for better generalization and robustness.  
- Integrate **clinical data (patient demographics)** for multi-modal learning.  

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgments
- Open medical imaging datasets (e.g., **Kaggle sources**).  
- **TensorFlow/Keras developers** and contributors.  
- The **open-source community** for supporting libraries and tools.  

---

## 🤝 Contributions
Contributions, feedback, and suggestions are welcome!  
Please **open an issue** or **submit a pull request** for enhancements, bug fixes, or new features.  


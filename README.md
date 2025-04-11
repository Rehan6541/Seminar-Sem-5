# 🫁 Pneumonia Detection using CNN-Based Feature Extraction

This repository contains the implementation of a deep learning-based pneumonia detection system developed as part of the **Seminar and Communication Skills** subject at **Sanjivani College of Engineering**, Kopargaon, for the academic year 2024–2025. The project leverages Convolutional Neural Networks (CNNs), specifically **DenseNet-169**, for feature extraction from chest X-ray images to detect pneumonia cases with high accuracy.


---

## 📌 Abstract

Pneumonia is a life-threatening respiratory infection, especially in underdeveloped regions where expert radiological diagnosis is not always available. This project aims to build an automated diagnostic system using deep learning, specifically CNNs, to detect pneumonia from chest X-ray images. The model employs DenseNet-169 for robust feature extraction and utilizes a Support Vector Machine (SVM) classifier for final prediction, achieving excellent accuracy and reliability while reducing diagnostic delays. The system is designed to be scalable and deployable in real-world healthcare environments, especially in resource-constrained settings.

---

## 🎯 Objectives

- Detect pneumonia automatically from chest X-ray images using CNN-based feature extraction.
- Improve diagnostic accuracy with deep learning over traditional methods.
- Design a user-friendly, scalable, and interpretable system for clinical use.
- Enable deployment in mobile and cloud environments.
- Ensure compliance with healthcare data privacy and security regulations.

---

## 🧪 Technologies Used

- **Deep Learning Models:** DenseNet-169, CNN  
- **Classifier:** Support Vector Machine (SVM)  
- **Libraries:** TensorFlow, Keras, OpenCV, Scikit-learn  
- **Deployment (optional):** TensorFlow Lite, CoreML, Cloud Hosting  
- **Data:** Chest X-ray Dataset (Kaggle or NIH ChestX-ray14)

---

## 🧱 System Architecture

1. **Preprocessing** – Resize images to 224×224, normalize, and augment for training.
2. **Feature Extraction** – Use DenseNet-169 (without final classification layer) to extract deep features.
3. **Classification** – Use SVM with RBF kernel for final decision-making.
4. **Performance Evaluation** – Metrics used: Accuracy, Precision, Recall, F1-Score, AUC.
5. **Deployment** – Scalable to both mobile and cloud platforms.

---

## 📊 Results

| Model              | Accuracy | Precision | Recall | F1-Score | AUC   |
|--------------------|----------|-----------|--------|----------|--------|
| Proposed CNN Model | 97.2%    | 97.4%     | 97.3%  | 97.37%   | 0.982 |
| SVM (with features)| 94.96%   | 92%       | 93.5%  | 92.75%   | 0.94  |
| KNN                | 91.5%    | 90%       | 92.1%  | 91.05%   | N/A   |
| Random Forest      | 92.3%    | 91.2%     | 90.9%  | 91.05%   | N/A   |

> DenseNet-169 + SVM performed best among all compared models.

---

## 📚 Literature References

1. Varshni et al., "Pneumonia Detection Using CNN Based Feature Extraction"
2. Szepesi & Szilagyi, "Detection of Pneumonia Using CNN and Deep Learning"
3. Lamia & Fawaz, "Pneumonia Detection on a Mobile Platform"
4. Sharma & Guleria, "VGG-16 Based CNN for Pneumonia Detection"
5. Qiuyu An et al., "CNN with Attention Ensemble for Pneumonia Detection"

---

## ✅ Future Enhancements

- Integrate 3D volumetric scan analysis using 3D CNNs.
- Include patient demographics and history for personalized prediction.
- Build Grad-CAM visualizations for explainable AI.
- Real-world clinical testing and feedback loop for continual learning.

---

## 🙏 Acknowledgements

Gratitude to **Dr. S.R. Deshmukh** (Guide), **Dr. D.B. Kshirsagar** (HOD), **Dr. A.G. Thakur** (Director), and all faculty members for their support and guidance throughout the seminar. Special thanks to peers and family for their constant motivation.

---


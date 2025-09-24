# Brain Metastasis Detection from MRI

A machine learning pipeline for detecting brain metastases from T1-pre and T1-post contrast MRI scans using advanced feature extraction and ensemble methods.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [License](#license)
- [Citation](#citation)
- [Author](#author)

## Overview

This project implements a comprehensive machine learning pipeline for detecting brain metastases from MRI scans. The system processes T1-pre and T1-post contrast MRI images, extracts advanced radiomic features, and employs ensemble learning techniques with cross-validation to achieve high diagnostic accuracy.

**Publication**: [Feature-Based Machine Learning for Brain Metastasis Detection Using Clinical MRI](https://www.medrxiv.org/content/10.1101/2025.09.22.25336307v1)  
**DOI**: https://doi.org/10.1101/2025.09.22.25336307  
**Publication Date**:  September 22, 2025

## Features

- **Advanced Image Processing**: Automated cropping, padding, and normalization of 3D MRI volumes
- **Comprehensive Feature Extraction**: Intensity-based, enhancement, texture, and histogram features
- **Data Augmentation**: Specialized techniques for handling class imbalance
- **Cross-Validation**: Robust 5-fold stratified cross-validation for reliable performance estimation
- **Multiple Classifiers**: Comparison of Random Forest, SVM, and Gradient Boosting models
- **Comprehensive Evaluation**: ROC curves, confusion matrices, and detailed classification reports

## Installation

### Prerequisites

- Python 3.8+
- NIfTI file support for medical imaging data

### Dependencies

Install required packages:

```bash
pip install nibabel numpy pandas scikit-learn matplotlib seaborn joblib imbalanced-learn
```

### Data Setup

1. Organize your dataset in the following structure:
```
dataset_directory/
├── patient_1/
│   ├── T1pre.nii.gz (or similar naming)
│   ├── T1post.nii.gz (or similar naming)
│   └── segmentation.nii.gz (or mask.nii.gz)
├── patient_2/
└── ...
```

2. Update the `base_directory` path in the `train_with_cross_validation()` function to point to your dataset.

## Usage

### Basic Training

Run the complete training and evaluation pipeline:

```bash
python brain_mets_detection.py
```

### Customization

Modify these parameters in the code for your specific needs:

```python
# Adjust these parameters as needed
dataset = BrainMetsDataset(
    base_dir=base_directory, 
    max_patients=50,           # Set to None to use all patients
    target_shape=(128, 128, 64) # Adjust based on your data characteristics
)
```

### Outputs

The script generates:
- Model file: `final_model.pkl`
- ROC curve: `roc_curve_cv.png`
- Confusion matrix: `confusion_matrix_cv.png`
- Console output with comprehensive evaluation metrics

## Methodology

### Data Processing
1. **Image Loading**: NIfTI files are loaded and converted to numpy arrays
2. **Preprocessing**: Images are cropped/padded to a consistent size (128×128×64)
3. **Normalization**: Intensity values are clipped and standardized

### Feature Extraction
- **Intensity Features**: Mean, standard deviation, percentiles
- **Enhancement Features**: Difference between T1-post and T1-pre contrasts
- **Texture Features**: Gradient-based texture analysis
- **Histogram Features**: Intensity distribution characteristics

### Modeling
- **Class Imbalance Handling**: Manual data augmentation with noise injection
- **Model Selection**: Comparison of Random Forest, SVM, and Gradient Boosting
- **Validation**: 5-fold stratified cross-validation
- **Evaluation**: Comprehensive metrics including ROC AUC, precision, recall, and F1-score

## Results

The model achieves the following performance metrics:

| Evaluation Metric | Value | Description & Interpretation |
|-------------------|-------|------------------------------|
| **Overall Accuracy** | 96.67% (87/90) | The proportion of total correct predictions (both classes) out of all predictions. Indicates a highly accurate model. |
| **ROC AUC Score** | 0.988 | Area Under the ROC Curve. A score near 1.0 indicates excellent class separation capability. |
| **Precision (No Metastasis)** | 1.00 | No False Alarms: When the model predicts "No Metastasis," it is always correct. Crucial to avoid unnecessary stress or procedures. |
| **Recall/Sensitivity (No Metastasis)** | 0.93 | Coverage: Correctly identifies 93% of actual healthy cases; 7% (3 cases) missed. |
| **F1-Score (No Metastasis)** | 0.96 | Harmonic mean of Precision and Recall for the "No Metastasis" class, showing strong balance. |
| **Precision (Metastasis)** | 0.94 | Reliability: When predicting "Metastasis," correct 94% of the time; few false positives occur. |
| **Recall/Sensitivity (Metastasis)** | 1.00 | Critical Performance: Identifies 100% of actual metastatic cases; vital to avoid missing serious conditions. |
| **F1-Score (Metastasis)** | 0.97 | Harmonic mean of Precision and Recall for the "Metastasis" class, indicating strong balance. |
| **Macro Avg F1-Score** | 0.97 | Unweighted average F1-score across both classes, showing excellent overall performance. |
| **False Positives (FP)** | 3 | Healthy cases incorrectly predicted as metastatic (Type I error). |
| **False Negatives (FN)** | 1 | Metastatic cases incorrectly predicted as healthy (Type II error); critical to minimize. |
| **Total Support** | 90 | Evaluation performed on 44 "No Metastasis" and 46 "Metastasis" samples. |

## License

This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0).

This means you are free to:

- **Share** — copy and redistribute the material in any medium or format for non-commercial purposes.

Under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial** — You may not use the material for commercial purposes.
- **NoDerivatives** — If you remix, transform, or build upon the material, you may not distribute the modified material.

**Summary**: This work may be read and downloaded for personal use only. It may be shared in its complete and unaltered form for non-commercial purposes, provided that the author's name, the title of the work, and a link to the original source (this repository) and the license are clearly cited. Any modification, adaptation, commercial use, or distribution for profit is strictly prohibited.

For permissions beyond the scope of this license, please contact the author directly.

![CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)

## Citation

If you use this work in your research, please cite the paper:

> **Rahi, A., & Shafiabadi, M. H.** (2025). *Feature-Based Machine Learning for Brain Metastasis Detection Using Clinical MRI*. medRxiv. September 22, 2025. https://doi.org/10.1101/2025.09.22.25336307

If you use the code implementation (software, scripts, etc.), please also cite:

> **Rahi, A.**(2025). *Feature-Based Machine Learning for Brain Metastasis Detection Using Clinical MRI* [Computer software]. GitHub repository, *AlirezaRahi/Feature-Based-Machine-Learning-for-Brain-Metastasis-Detection-Using-Clinical-MRI*. Retrieved from https://github.com/AlirezaRahi/Feature-Based-Machine-Learning-for-Brain-Metastasis-Detection-Using-Clinical-MRI


## Author

### Alireza Rahi 
**Independent Researcher**  
Tehran, Iran  
Email: Alireza.rahi@outlook.com  
LinkedIn: [https://www.linkedin.com/in/alireza-rahi-6938b4154/](https://www.linkedin.com/in/alireza-rahi-6938b4154/)  
GitHub: [https://github.com/AlirezaRahi](https://github.com/AlirezaRahi)  


## Acknowledgments

We thank the contributors to the open-source medical imaging and machine learning libraries that made this work possible. Special thanks to the UCSF Brain Metastases dataset providers for the training data used in this research.

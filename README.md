# Breast Cancer Diagnosis: TabNet Deep Learning & Feature Selection Optimization

## Project Overview
This repository contains the official codebase for my undergraduate thesis and published academic research. The project focuses on building an **Interpretable Deep Learning model** to classify breast cancer tumors (Benign vs. Malignant) using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. 

Unlike traditional "black-box" neural networks, this project utilizes **TabNet** (an attentive interpretable tabular learning architecture) combined with rigorous feature selection and hyperparameter tuning to ensure both high predictive performance and clinical transparency.

🔗 **[Read the Full Published Paper Here](https://teknosi.fti.unand.ac.id/index.php/teknosi/article/view/3880)**

## Tech Stack & Libraries
* **Language:** Python
* **Deep Learning Framework:** PyTorch, `pytorch-tabnet`
* **Machine Learning & Tuning:** Scikit-Learn, Optuna (Hyperparameter Optimization)
* **Data Resampling:** `imbalanced-learn` (SMOTE-ENN)
* **Environment:** Google Colaboratory

## The Machine Learning Pipeline
Dealing with sensitive medical data requires a robust pipeline to prevent data leakage and overfitting. The pipeline includes:
1. **Data Pre-processing:** Feature scaling using `StandardScaler`.
2. **Handling Imbalanced Data:** Applied **SMOTE-ENN** (Synthetic Minority Over-sampling Technique combined with Edited Nearest Neighbors) exclusively on the training data to balance the classes and clean noisy synthetic borders.
3. **Feature Selection Comparison:** Evaluated and compared two filter-based methods:
   * **Chi-Square:** For statistical independence evaluation.
   * **Information Gain:** For measuring entropy reduction.
4. **Modeling & Tuning:** Trained the **TabNetClassifier**. Automated the hyperparameter search (learning rate, steps, gamma, etc.) using **Optuna** with Stratified 5-Fold Cross-Validation.

## Key Results & Clinical Impact
The baseline model (without optimization) achieved 82.46% accuracy. After feature selection and Optuna optimization, the performance increased drastically:

* **TabNet + Chi-Square (75% features):** Achieved **98.25% Accuracy and 100% Precision**. Best configuration for computational efficiency and performance balance.
* **TabNet + Information Gain (75% features):** Achieved **98.25% Accuracy and 100% Recall (Sensitivity)**. 

**Conclusion:** In medical diagnostics like cancer detection, minimizing False Negatives is the absolute highest priority. Therefore, the Information Gain configuration (achieving 100% Recall) is the most viable model for real-world clinical applications.

## Repository Structure
* `fs_chi2.ipynb`: The one of Google Colab notebook containing the end-to-end pipeline (EDA, Pre-processing, Optuna Tuning, Evaluation).
* `README.md`: Project documentation.
* `wdbc.data`: The WDBC dataset
*(Note: The WDBC dataset is publicly available via the UCI Machine Learning Repository).*

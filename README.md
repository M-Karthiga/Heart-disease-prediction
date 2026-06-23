# Heart Disease Prediction

> An end-to-end machine learning classification project to predict the presence of heart disease from clinical patient data. Built as part of Andrei Neagoie and DAniel's course on ML and DS in Udemy

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Problem Statement

> Given clinical parameters about a patient, can we predict whether or not they have heart disease?

This project explores whether machine learning can reliably flag at-risk patients using only routine diagnostic measurements — without invasive procedures.

**Target metric:** ≥ 88% cross-validated accuracy on the Cleveland Heart Disease dataset. 

---

## Results Summary

| Model | Accuracy |
|---|---|
| Logistic Regression (baseline) | ~82% |
| K-Nearest Neighbours | ~68% |
| Random Forest | ~83% |
| **Logistic Regression (tuned, GridSearchCV)** | **~88.5%** |

**Final model — 5-fold cross-validated metrics:**

| Metric | Score |
|---|---|
| Accuracy | ~88% |
| Precision | ~87% |
| Recall | ~90% |
| F1-Score | ~88% |

---

## Dataset

- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (Cleveland subset)
- **Samples:** 303 patients
- **Features:** 13 clinical attributes
- **Target:** Binary — `1` = heart disease present, `0` = absent

### Feature Dictionary

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Thalassemia type (normal / fixed defect / reversible defect) |

---

## Project Structure

```
Heart-disease-prediction/
├── End-to-end-heart-disease-classification.ipynb   # Full analysis notebook
├── heart-disease.csv                                # Dataset
├── requirements.txt                                 # Dependencies
└── README.md
```

---

## Workflow

```
1. Problem Definition
2. Data Loading & Exploration (EDA)
3. Feature Analysis
   ├── Heart disease by sex
   ├── Age vs max heart rate scatter
   ├── Chest pain type frequency
   └── Correlation heatmap
4. Modelling
   ├── Logistic Regression
   ├── K-Nearest Neighbours
   └── Random Forest
5. Hyperparameter Tuning
   ├── RandomizedSearchCV (Random Forest)
   └── GridSearchCV (Logistic Regression)
6. Evaluation
   ├── ROC curve & AUC
   ├── Confusion matrix
   ├── Classification report
   └── Cross-validated precision / recall / F1
7. Feature Importance
8. Experimentation (CatBoost notes)
```

---

## Key Findings

- **Chest pain type (`cp`)** is the strongest positive predictor (correlation +0.43 with target)
- **Exercise-induced angina (`exang`)** is the strongest negative predictor (−0.44)
- **Fasting blood sugar (`fbs`)** shows near-zero correlation and contributes minimally
- **Slope of ST segment:** downsloping (type 2) is strongly associated with disease presence
- Tuned Logistic Regression outperforms more complex models, consistent with the dataset's modest size (n=303)

---

## Installation

```bash
git clone https://github.com/M-Karthiga/Heart-disease-prediction.git
cd Heart-disease-prediction
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook End-to-end-heart-disease-classification.ipynb
```

---

## References

- Detrano, R. et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology.
- UCI ML Repository — [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Scikit-learn documentation — [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

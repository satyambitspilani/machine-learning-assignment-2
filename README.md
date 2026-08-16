# Machine Learning Assignment 2 — Classification Models

## a. Problem Statement

The objective of this assignment is to implement multiple machine learning
classification models on a common classification dataset, evaluate their
performance using multiple evaluation metrics, and build an interactive
Streamlit web application to demonstrate the trained models.

The application allows users to:

- Upload test data in CSV format.
- Select a machine learning model.
- Select a test-data row.
- View the actual class.
- View the predicted class.
- View prediction probabilities.
- View model evaluation metrics.
- View a classification report.
- View a confusion matrix.
- Compare the performance of all implemented models.

---

## b. Dataset Description

### Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset**

The dataset is a binary classification dataset containing characteristics
computed from digitized images of fine needle aspirate (FNA) of breast masses.

The dataset contains:

- 569 instances
- 30 numerical features
- 2 target classes

The target variable is:

- `malignant`
- `benign`

The dataset satisfies the assignment requirement of at least 500 instances
and at least 12 features.

For reproducibility, the project uses the Breast Cancer Wisconsin
(Diagnostic) dataset available through `scikit-learn` using
`load_breast_cancer()`.

### Dataset Statistics

| Property | Value |
|---|---:|
| Number of Instances | 569 |
| Number of Features | 30 |
| Number of Classes | 2 |
| Classification Type | Binary Classification |

### Target Distribution

| Class | Number of Instances |
|---|---:|
| Benign | 357 |
| Malignant | 212 |

---

## c. GitHub Repository Link

**Repository:**  
https://github.com/satyambitspilani/machine-learning-assignment-2

This link will be updated after the project is pushed to GitHub.

---

## d. Models Used

The assignment requires six classification models. The PDF explicitly names
Logistic Regression, Decision Tree, K-Nearest Neighbor, Naive Bayes and Random
Forest, while also requiring six models. Therefore, SVM is included as the
sixth classification model in this implementation.

### 1. Logistic Regression

Logistic Regression is a linear classification algorithm that estimates the
probability of an observation belonging to a particular class.

Feature scaling is applied using `StandardScaler` before classification.

### 2. Decision Tree Classifier

Decision Tree Classifier makes predictions by recursively splitting the
dataset based on feature values.

It is an interpretable tree-based classification algorithm.

### 3. K-Nearest Neighbor (kNN)

K-Nearest Neighbor classifies an observation based on the classes of its
nearest training observations.

Feature scaling is applied using `StandardScaler` because KNN is
distance-based.

### 4. Naive Bayes

Gaussian Naive Bayes is used for this classification problem.

It applies Bayes' theorem while assuming conditional independence between
features.

### 5. Random Forest

Random Forest is an ensemble learning method that combines multiple decision
trees to improve predictive performance and reduce overfitting.

### 6. Support Vector Machine (SVM)

Support Vector Machine is used as the sixth classification model.

An RBF kernel is used, with `StandardScaler` applied before classification.

---

## Evaluation Metrics

Each model is evaluated using the following metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison

The following results were obtained using the test dataset generated from an
80/20 stratified train-test split with `random_state=42`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| kNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest | 0.9737 | 0.9944 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| SVM | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved strong overall performance with 96.49% accuracy, 99.60% AUC and 92.45% MCC. It provides a strong linear baseline for the dataset. |
| Decision Tree | Achieved 92.11% accuracy. It performed reasonably well but had the lowest overall performance among the implemented models based on Accuracy, F1 and MCC. |
| kNN | Achieved 95.61% accuracy and 93.83% F1. Feature scaling is important because the algorithm is distance-based. |
| Naive Bayes | Achieved 93.86% accuracy and 100% precision, but recall was 83.33%. Its conditional-independence assumption may not fully represent the relationships between the dataset features. |
| Random Forest | Achieved 97.37% accuracy, 100% precision, 92.86% recall, 96.30% F1 and 94.42% MCC. It was one of the strongest models in the experiment. |
| SVM | Achieved 97.37% accuracy, 99.47% AUC, 100% precision, 92.86% recall, 96.30% F1 and 94.42% MCC. It was one of the strongest models and achieved the highest AUC among the six models. |

---

## Overall Winner

Based on the test-set results, **SVM is selected as the overall winner**.

SVM achieved:

- Accuracy: **97.37%**
- AUC: **99.47%**
- Precision: **100.00%**
- Recall: **92.86%**
- F1 Score: **96.30%**
- MCC: **94.42%**

Random Forest achieved the same Accuracy, Precision, Recall, F1 and MCC,
but SVM achieved a slightly higher AUC.

Therefore, SVM is selected as the overall winner for this experiment.

---

# Project Structure

```text
machine_Learning_Assignment_2/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_metrics.csv
├── model_results.json
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    └── svm.joblib

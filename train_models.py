from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "model"

MODEL_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

data = load_breast_cancer(
    as_frame=True
)

df = data.frame.copy()


# ============================================================
# 3. PREPARE TARGET COLUMN
# ============================================================

df["diagnosis"] = df["target"].map({
    0: "malignant",
    1: "benign"
})

df = df.drop(
    columns=["target"]
)


# Put diagnosis as the last column

df = df[
    [
        column
        for column in df.columns
        if column != "diagnosis"
    ]
    + ["diagnosis"]
]


# ============================================================
# 4. DISPLAY DATASET INFORMATION
# ============================================================

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nTarget distribution:")
print(df["diagnosis"].value_counts())


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["diagnosis"]
)


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X_train = train_df.drop(
    columns=["diagnosis"]
)

y_train = train_df["diagnosis"]

X_test = test_df.drop(
    columns=["diagnosis"]
)

y_test = test_df["diagnosis"]


# ============================================================
# 7. CREATE CLASSIFICATION MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
                random_state=42
            )
        )
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "kNN": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=7
            )
        )
    ]),

    "Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            )
        )
    ])
}


# ============================================================
# 8. CREATE RESULT CONTAINERS
# ============================================================

metrics_rows = []

reports = {}


# ============================================================
# 9. TRAIN AND EVALUATE MODELS
# ============================================================

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Predictions
    predictions = model.predict(
        X_test
    )

    # Prediction probabilities
    probabilities = model.predict_proba(
        X_test
    )

    # Identify malignant class
    classes = list(
        model.classes_
    )

    positive_index = classes.index(
        "malignant"
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # AUC
    auc = roc_auc_score(
        (y_test == "malignant").astype(int),
        probabilities[:, positive_index]
    )

    # Precision
    precision = precision_score(
        y_test,
        predictions,
        pos_label="malignant",
        zero_division=0
    )

    # Recall
    recall = recall_score(
        y_test,
        predictions,
        pos_label="malignant",
        zero_division=0
    )

    # F1
    f1 = f1_score(
        y_test,
        predictions,
        pos_label="malignant",
        zero_division=0
    )

    # MCC
    mcc = matthews_corrcoef(
        y_test,
        predictions
    )

    # Store metrics
    metrics_rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    })

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[
            "malignant",
            "benign"
        ]
    )

    # Classification report
    report = classification_report(
        y_test,
        predictions,
        labels=[
            "malignant",
            "benign"
        ],
        target_names=[
            "malignant",
            "benign"
        ],
        output_dict=True,
        zero_division=0
    )

    # Save report
    reports[name] = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

    # Save trained model
    model_filename = (
        name
        .lower()
        .replace(" ", "_")
        + ".joblib"
    )

    joblib.dump(
        model,
        MODEL_DIR / model_filename
    )


# ============================================================
# 10. CREATE MODEL COMPARISON TABLE
# ============================================================

metrics_df = pd.DataFrame(
    metrics_rows
)

print("\nModel Comparison:")

print(
    metrics_df.to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE MODEL METRICS
# ============================================================

metrics_df.to_csv(
    ROOT / "model_metrics.csv",
    index=False
)


# ============================================================
# 12. SAVE TEST DATA
# ============================================================

test_df.to_csv(
    ROOT / "test_data.csv",
    index=False
)


# ============================================================
# 13. SAVE MODEL RESULTS
# ============================================================

results = {
    "metrics": metrics_rows,
    "reports": reports
}

with open(
    ROOT / "model_results.json",
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=2
    )


# ============================================================
# 14. COMPLETION MESSAGE
# ============================================================

print("\nTraining completed successfully.")

print(
    f"Saved models to: {MODEL_DIR}"
)

print(
    f"Saved test data to: "
    f"{ROOT / 'test_data.csv'}"
)

print(
    f"Saved metrics to: "
    f"{ROOT / 'model_metrics.csv'}"
)
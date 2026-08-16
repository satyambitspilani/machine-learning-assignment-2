import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "model"

TEST_DATA_PATH = ROOT / "test_data.csv"

METRICS_PATH = ROOT / "model_metrics.csv"

RESULTS_PATH = ROOT / "model_results.json"


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ML Classification App",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# 3. APPLICATION TITLE
# ============================================================

st.title(
    "Machine Learning Classification App"
)

st.write(
    "Compare six classification models "
    "and generate predictions using test data."
)


# ============================================================
# 4. LOAD DEFAULT TEST DATA
# ============================================================

@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_DATA_PATH)


default_test_df = load_test_data()


# ============================================================
# 5. TEST DATA CSV UPLOAD
# ============================================================

st.sidebar.header(
    "Test Data"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Data CSV",
    type=["csv"],
    help="Upload the test-data CSV file used for prediction."
)


if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    st.sidebar.success(
        "Uploaded test data loaded successfully."
    )

    st.sidebar.caption(
        f"File: {uploaded_file.name}"
    )

else:

    test_df = default_test_df

    st.sidebar.info(
        "Using default test_data.csv"
    )


# ============================================================
# 6. VALIDATE TEST DATA
# ============================================================

expected_feature_columns = [
    column
    for column in default_test_df.columns
    if column != "diagnosis"
]


missing_columns = [
    column
    for column in expected_feature_columns
    if column not in test_df.columns
]


if missing_columns:

    st.error(
        "The uploaded CSV is missing required feature columns."
    )

    st.write(
        "Missing columns:"
    )

    st.write(
        missing_columns
    )

    st.stop()


feature_columns = expected_feature_columns


if "diagnosis" not in test_df.columns:

    st.warning(
        "The uploaded CSV does not contain the 'diagnosis' "
        "column. Actual-vs-predicted comparison and "
        "evaluation metrics cannot be calculated for "
        "the uploaded data."
    )


# ============================================================
# 7. LOAD MODEL RESULTS
# ============================================================

@st.cache_data
def load_results():

    with open(
        RESULTS_PATH,
        "r"
    ) as file:

        return json.load(file)


results = load_results()


# ============================================================
# 8. LOAD MODEL METRICS
# ============================================================

@st.cache_data
def load_metrics():

    return pd.read_csv(
        METRICS_PATH
    )


metrics_df = load_metrics()


# ============================================================
# 9. MODEL FILE MAPPING
# ============================================================

MODEL_FILES = {

    "Logistic Regression":
        "logistic_regression.joblib",

    "Decision Tree":
        "decision_tree.joblib",

    "kNN":
        "knn.joblib",

    "Naive Bayes":
        "naive_bayes.joblib",

    "Random Forest":
        "random_forest.joblib",

    "SVM":
        "svm.joblib"
}


# ============================================================
# 10. MODEL SELECTION
# ============================================================

st.sidebar.header(
    "Model Selection"
)

selected_model = st.sidebar.selectbox(
    "Select ML Model",
    list(MODEL_FILES.keys())
)


# ============================================================
# 11. LOAD SELECTED MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):

    return joblib.load(
        model_path
    )


selected_model_path = (
    MODEL_DIR /
    MODEL_FILES[selected_model]
)

model = load_model(
    selected_model_path
)


# ============================================================
# 12. TEST DATA ROW SELECTION
# ============================================================

st.sidebar.header(
    "Test Data Selection"
)

row_number = st.sidebar.number_input(
    "Select Test Row",
    min_value=0,
    max_value=len(test_df) - 1,
    value=0,
    step=1
)


# ============================================================
# 13. SELECT TEST ROW
# ============================================================

selected_row = test_df.iloc[
    row_number
]


# ============================================================
# 14. PREPARE INPUT DATA
# ============================================================

X_selected = selected_row[
    feature_columns
].to_frame().T


if "diagnosis" in test_df.columns:

    actual_class = selected_row[
        "diagnosis"
    ]

else:

    actual_class = None


# ============================================================
# 15. MAKE PREDICTION
# ============================================================

prediction = model.predict(
    X_selected
)[0]


# ============================================================
# 16. PREDICTION PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_selected
)[0]

class_probabilities = dict(
    zip(
        model.classes_,
        probabilities
    )
)


# ============================================================
# 17. DISPLAY PREDICTION
# ============================================================

st.header(
    "Prediction"
)

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Actual Class"
    )

    if actual_class is not None:

        st.write(
            actual_class
        )

    else:

        st.write(
            "Not available"
        )


with col2:

    st.subheader(
        "Predicted Class"
    )

    st.write(
        prediction
    )


# ============================================================
# 18. DISPLAY PREDICTION PROBABILITIES
# ============================================================

st.subheader(
    "Prediction Probabilities"
)


probability_df = pd.DataFrame({

    "Class":
        list(
            class_probabilities.keys()
        ),

    "Probability":
        list(
            class_probabilities.values()
        )
})


probability_df[
    "Probability"
] = (
    probability_df["Probability"] * 100
).round(2)


st.dataframe(
    probability_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 19. MODEL EVALUATION METRICS
# ============================================================

st.header(
    "Model Evaluation Metrics"
)


selected_metrics = metrics_df[
    metrics_df["ML Model Name"]
    == selected_model
]


st.dataframe(
    selected_metrics,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 20. CLASSIFICATION REPORT
# ============================================================

st.header(
    "Classification Report"
)


classification_report_data = results[
    "reports"
][selected_model][
    "classification_report"
]


classification_report_df = pd.DataFrame(
    classification_report_data
).transpose()


st.dataframe(
    classification_report_df,
    use_container_width=True
)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

st.header(
    "Confusion Matrix"
)


confusion_matrix_data = results[
    "reports"
][selected_model][
    "confusion_matrix"
]


cm_df = pd.DataFrame(
    confusion_matrix_data,
    index=[
        "Actual Malignant",
        "Actual Benign"
    ],
    columns=[
        "Predicted Malignant",
        "Predicted Benign"
    ]
)


st.dataframe(
    cm_df,
    use_container_width=True
)


# ============================================================
# 22. SELECTED TEST DATA ROW
# ============================================================

st.header(
    "Selected Test Data Row"
)


st.dataframe(
    selected_row.to_frame().T,
    use_container_width=True
)
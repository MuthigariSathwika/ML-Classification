import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

st.set_page_config(page_title="ML Classification Models", layout="wide")
AVG_SCORE_COL = "Average Score"
MODEL_DIR = Path(__file__).resolve().parent / "model"
SCALER_FILE = "scaler.pkl"
MODEL_FILE_MAP = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree Classifier": "decision_tree.pkl",
    "K-Nearest Neighbor Classifier": "knn.pkl",
    "Naive Bayes Classifier (Gaussian)": "naive_bayes.pkl",
    "Random Forest Classifier": "random_forest.pkl",
    "XGBoost Classifier": "xgboost.pkl",
}


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
    except Exception:
        auc_score = 0.0

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC Score": auc_score,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC Score": mcc,
    }


def get_model_definitions():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            ccp_alpha=0.0,
        ),
        "K-Nearest Neighbor Classifier": KNeighborsClassifier(
            n_neighbors=5, weights="distance"
        ),
        "Naive Bayes Classifier (Gaussian)": GaussianNB(),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features="sqrt",
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=100,
            random_state=42,
            learning_rate=0.1,
            max_depth=6,
            eval_metric="logloss",
        ),
    }


def all_artifacts_exist():
    return (MODEL_DIR / SCALER_FILE).exists() and all(
        (MODEL_DIR / filename).exists() for filename in MODEL_FILE_MAP.values()
    )


def save_artifacts(scaler, models):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / SCALER_FILE)
    for model_name, model in models.items():
        joblib.dump(model, MODEL_DIR / MODEL_FILE_MAP[model_name])


def load_artifacts():
    scaler = joblib.load(MODEL_DIR / SCALER_FILE)
    models = {
        model_name: joblib.load(MODEL_DIR / filename)
        for model_name, filename in MODEL_FILE_MAP.items()
    }
    return scaler, models


@st.cache_resource
def load_or_train_and_evaluate_models():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    X = df.drop("target", axis=1)
    y = df["target"]

    data_source = "Loaded pre-trained .pkl models"

    if all_artifacts_exist():
        scaler, trained_models = load_artifacts()
    else:
        scaler = StandardScaler()
        scaler.fit(X)
        trained_models = get_model_definitions()

        x_scaled_train = scaler.transform(X)
        x_scaled_train = pd.DataFrame(x_scaled_train, columns=X.columns)

        x_train_fit, _, y_train_fit, _ = train_test_split(
            x_scaled_train,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        for model in trained_models.values():
            model.fit(x_train_fit, y_train_fit)

        save_artifacts(scaler, trained_models)
        data_source = "Artifacts were missing, so models were trained and saved as .pkl"

    x_scaled = scaler.transform(X)
    x_scaled = pd.DataFrame(x_scaled, columns=X.columns)

    _, X_test, _, y_test = train_test_split(
        x_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = []
    for model_name, model in trained_models.items():
        result = evaluate_model(model, X_test, y_test, model_name)
        results.append(result)

    results_df = pd.DataFrame(results)
    metric_cols = [
        "Accuracy",
        "AUC Score",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC Score",
    ]
    results_df[AVG_SCORE_COL] = results_df[metric_cols].mean(axis=1)
    best_model_name = results_df.loc[results_df[AVG_SCORE_COL].idxmax(), "Model"]

    return df, X, scaler, trained_models, results_df, best_model_name, data_source


df, X_raw, scaler, trained_models, results_df, best_model_name, data_source = (
    load_or_train_and_evaluate_models()
)

st.title("Breast Cancer Classification - Model Evaluation Dashboard")
st.write(
    "This app reproduces notebook pipeline: StandardScaler preprocessing, "
    "6 classification models, and metric-based comparison."
)
st.caption(data_source)

col1, col2, col3 = st.columns(3)
col1.metric("Instances", f"{df.shape[0]}")
col2.metric("Features", f"{X_raw.shape[1]}")
# a. Dataset upload option [1 mark]
st.subheader("📤 Upload Custom Test Data (CSV)")
uploaded_file = st.file_uploader(
    "Upload a CSV file with test data (optional - demo built-in data used by default)",
    type=["csv"],
    help="CSV must have same 30 features as Breast Cancer dataset + 'target' column",
)

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        if "target" in uploaded_df.columns and uploaded_df.shape[1] == 31:
            X_uploaded = uploaded_df.drop("target", axis=1)
            y_uploaded = uploaded_df["target"]
            st.success(f"✓ Uploaded data: {uploaded_df.shape[0]} rows, {X_uploaded.shape[1]} features")
            # Use uploaded data for evaluation
            X_test_eval = scaler.transform(X_uploaded)
            y_test_eval = y_uploaded
            data_mode = "uploaded"
        else:
            st.error("CSV must have 30 feature columns + 'target' column (31 total)")
            X_test_eval = scaler.transform(X_raw)
            _, X_test_eval, _, y_test_eval = train_test_split(
                X_test_eval, df["target"], test_size=0.2, random_state=42, stratify=df["target"]
            )
            data_mode = "built-in"
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        X_test_eval = scaler.transform(X_raw)
        _, X_test_eval, _, y_test_eval = train_test_split(
            X_test_eval, df["target"], test_size=0.2, random_state=42, stratify=df["target"]
        )
        data_mode = "built-in"
else:
    X_test_eval = scaler.transform(X_raw)
    _, X_test_eval, _, y_test_eval = train_test_split(
        X_test_eval, df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )
    data_mode = "built-in"

st.subheader("Model Performance Comparison")
st.dataframe(results_df.sort_values(AVG_SCORE_COL, ascending=False), use_container_width=True)

best_row = results_df.loc[results_df["Model"] == best_model_name].iloc[0]
st.success(
    f"Best overall model: {best_model_name} | Average Score: {best_row[AVG_SCORE_COL]:.4f}"
)

# b. Model selection dropdown
st.subheader("Select Model for Detailed Evaluation")
selected_eval_model = st.selectbox(
    "Choose a model to evaluate on test data",
    list(trained_models.keys()),
    index=0,
    key="eval_model_select",
)
eval_model = trained_models[selected_eval_model]

# Generate predictions for selected model
y_pred_eval = eval_model.predict(X_test_eval)
y_pred_proba_eval = (
    eval_model.predict_proba(X_test_eval)[:, 1]
    if hasattr(eval_model, "predict_proba")
    else None
)

# c. Display of evaluation metrics
st.subheader("📊 Evaluation Metrics")
metric_col1, metric_col2, metric_col3 = st.columns(3)
acc = accuracy_score(y_test_eval, y_pred_eval)
prec = precision_score(y_test_eval, y_pred_eval, average="weighted", zero_division=0)
rec = recall_score(y_test_eval, y_pred_eval, average="weighted", zero_division=0)
metric_col1.metric("Accuracy", f"{acc:.4f}")
metric_col2.metric("Precision", f"{prec:.4f}")
metric_col3.metric("Recall", f"{rec:.4f}")

metric_col4, metric_col5, metric_col6 = st.columns(3)
f1_val = f1_score(y_test_eval, y_pred_eval, average="weighted", zero_division=0)
mcc_val = matthews_corrcoef(y_test_eval, y_pred_eval)
auc_val = roc_auc_score(y_test_eval, y_pred_proba_eval) if y_pred_proba_eval is not None else 0.0
metric_col4.metric("F1 Score", f"{f1_val:.4f}")
metric_col5.metric("MCC", f"{mcc_val:.4f}")
metric_col6.metric("AUC", f"{auc_val:.4f}")

# d. Confusion matrix or classification report [1 mark]
st.subheader("📋 Confusion Matrix & Classification Report")
tab1, tab2 = st.tabs(["Confusion Matrix", "Classification Report"])

with tab1:
    cm = confusion_matrix(y_test_eval, y_pred_eval)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Malignant (0)", "Benign (1)"],
        yticklabels=["Malignant (0)", "Benign (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix - {selected_eval_model}")
    st.pyplot(fig)

with tab2:
    report = classification_report(
        y_test_eval,
        y_pred_eval,
        target_names=["Malignant (0)", "Benign (1)"],
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

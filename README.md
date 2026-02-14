# Machine Learning Classification Models Evaluation

## a) Problem statement

Implement and evaluate six machine learning classification models on the Breast Cancer Wisconsin (Diagnostic) dataset and compare their performance using the following metrics:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## b) Dataset description [1 mark]

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset (UCI / sklearn)
- **Total instances:** 569
- **Total features:** 30 (all numeric)
- **Task type:** Binary classification
- **Target classes:** 0 = Malignant, 1 = Benign
- **Train-test split:** 80% training, 20% testing (`random_state=42`, stratified)
- **Preprocessing:** StandardScaler feature scaling

## c) Models used and comparison table [6 marks]

The following 6 models were implemented and evaluated:
1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison Table (from notebook outputs)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |
| Decision Tree | 0.9035 | 0.9216 | 0.9090 | 0.9035 | 0.9045 | 0.8011 |
| kNN | 0.9649 | 0.9785 | 0.9652 | 0.9649 | 0.9647 | 0.9245 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9298 | 0.9298 | 0.9298 | 0.8492 |
| Random Forest (Ensemble) | 0.9474 | 0.9947 | 0.9474 | 0.9474 | 0.9474 | 0.8869 |
| XGBoost (Ensemble) | 0.9474 | 0.9924 | 0.9474 | 0.9474 | 0.9471 | 0.8864 |

### Overall best model

- **Best model:** Logistic Regression
- **Average score across all metrics:** 0.9812

## Observations on model performance [3 marks]

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer across all six metrics; highest Accuracy, AUC, Precision, Recall, F1, and MCC. |
| Decision Tree | Lowest scores among all models; likely more variance/overfitting compared with ensemble and linear alternatives. |
| kNN | Strong second-best overall performance with high Accuracy and MCC; effective on this scaled numeric dataset. |
| Naive Bayes | Moderate Accuracy but strong AUC; simple probabilistic assumptions limit classification strength compared with top models. |
| Random Forest (Ensemble) | Robust and stable performance across all metrics; slightly below Logistic Regression and kNN on this dataset. |
| XGBoost (Ensemble) | Performance close to Random Forest, with strong AUC and balanced metrics; marginally lower than RF in F1/MCC. |

## Model artifact files (.pkl) for deployment

To avoid retraining on every Streamlit startup, save all models and scaler as `.pkl` files once:

```bash
python model/save_models.py
```

This generates the following files inside `model/`:
- `logistic_regression.pkl`
- `decision_tree.pkl`
- `knn.pkl`
- `naive_bayes.pkl`
- `random_forest.pkl`
- `xgboost.pkl`
- `scaler.pkl`

The app automatically loads these `.pkl` files if present. If missing, it trains models once and creates them.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

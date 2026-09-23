# Customer Churn Prediction and Analytics Using Machine Learning

**AICTE + IBM SkillsBuild Data Analytics with AI Internship**

---

## 1. Project Overview

This project applies data analytics and supervised machine learning to predict customer churn in the telecommunications industry. Using the IBM Telco Customer Churn dataset, it combines exploratory data analysis, feature engineering, a leak-free preprocessing pipeline, and five classification models to identify customers who are likely to discontinue their service.

The project covers the complete data science workflow — from raw data inspection through model evaluation and business interpretation — and is implemented as a professional Jupyter Notebook suitable for an academic internship submission.

---

## 2. Problem Statement

Customer churn — the decision by a customer to stop using a service — is a major source of revenue loss in the telecommunications industry. Retaining an existing customer is significantly more cost-effective than acquiring a new one. Without a predictive model, retention efforts are reactive and untargeted.

**The goal is to answer:** *Given a customer's demographic profile, account details, and service subscriptions, can we predict whether that customer will churn?*

---

## 3. Objectives

- Perform thorough exploratory data analysis to understand the data and identify patterns associated with churn.
- Build and evaluate multiple supervised machine learning classifiers.
- Select the best-performing model based on objective evaluation metrics appropriate for an imbalanced classification task.
- Generate feature importance explanations to understand which factors drive churn.
- Derive business insights and recommendations directly from the data and model results.

---

## 4. Dataset

| Property | Detail |
|---|---|
| **Dataset name** | Telco Customer Churn |
| **Source** | [Kaggle — IBM Sample Data](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Records** | 7,043 customers |
| **Columns** | 21 |
| **Target variable** | `Churn` (Yes / No → 1 / 0) |
| **Class balance** | 26.5% churn, 73.5% no churn |

### Feature categories

| Group | Columns |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Phone services | `PhoneService`, `MultipleLines` |
| Internet services | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

> **Note:** `TotalCharges` is stored as a string in the raw CSV. Eleven rows contain whitespace values that are converted to `NaN` and imputed during preprocessing.

---

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3 | Core programming language |
| Jupyter Notebook | Interactive development and presentation |
| NumPy | Numerical computing |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Scikit-learn | Machine learning, preprocessing, and evaluation |

---

## 6. Machine Learning Approach

### Data cleaning
- `customerID` dropped (non-predictive identifier).
- `TotalCharges` converted from string to numeric; 11 blank values become `NaN`.
- Target `Churn` encoded as `Yes → 1`, `No → 0`.
- Redundant category levels (`"No internet service"`, `"No phone service"`) consolidated to `"No"` across all service add-on columns.

### Feature engineering
- `NumAddOnServices` — count of `"Yes"` values across the seven internet service add-on columns.
- `TenureGroup` — tenure binned into four groups (0–12, 13–24, 25–48, 49–72 months) for EDA visualisation only.

### Train / test split
- **80% train / 20% test**, stratified on the target to preserve the 26.5% churn ratio.
- Split performed **before** any preprocessing to prevent data leakage.
- Result: 5,634 training rows, 1,409 test rows.

### Preprocessing pipeline
A scikit-learn `ColumnTransformer` is fitted **exclusively on training data** and then applied to the test set. It applies:
- `SimpleImputer (median)` + `StandardScaler` to numeric columns.
- `OrdinalEncoder` to binary Yes/No and gender columns.
- `OneHotEncoder (drop='first')` to nominal multi-level columns (`InternetService`, `PaymentMethod`).
- `OrdinalEncoder` with explicit order to `Contract`.
- Pass-through for `SeniorCitizen` (already 0/1).

This produces **23 features** after preprocessing.

### Model training
Five classifiers are trained on the same processed training set with `random_state=42`:

- **Logistic Regression** — linear baseline; `class_weight='balanced'`
- **Decision Tree** — interpretable tree; `max_depth=5`, `class_weight='balanced'`
- **Random Forest** — ensemble of 200 trees; `class_weight='balanced'`
- **Gradient Boosting** — sequential boosting; 200 estimators, `learning_rate=0.1`
- **Support Vector Machine** — RBF kernel; `class_weight='balanced'`, `probability=True`

`class_weight='balanced'` is applied where available to mitigate the 26.5/73.5 class imbalance without discarding data.

### Model evaluation
Each model is evaluated on the held-out test set using five metrics plus a confusion matrix. The best model is selected programmatically from the actual test-set results — not assumed in advance.

---

## 7. Evaluation Metrics

Because the dataset is class-imbalanced (26.5% churn), **accuracy alone is an insufficient metric**. A trivial classifier that always predicts "No Churn" would achieve ~73.5% accuracy while identifying zero churners.

| Metric | Why it matters |
|---|---|
| **Accuracy** | Overall fraction correct; reported for completeness |
| **Precision (Churn=1)** | Of customers predicted to churn, how many actually did — controls false alarms |
| **Recall (Churn=1)** | Of actual churners, how many were correctly identified — controls missed churners |
| **F1-score (Churn=1)** | Harmonic mean of Precision and Recall — balances both concerns |
| **ROC-AUC** | Model's discrimination ability across all thresholds; robust to class imbalance |
| **Confusion Matrix** | Full breakdown of true/false positives and negatives |

**Primary selection criteria: ROC-AUC and F1-score (Churn=1).**

---

## 8. Results

### Model comparison (test set — 1,409 rows)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7374 | 0.5034 | 0.7834 | 0.6130 | 0.8418 |
| Decision Tree | 0.7566 | 0.5276 | 0.7914 | 0.6332 | 0.8346 |
| **Random Forest** | **0.7700** | **0.5517** | **0.7139** | **0.6224** | **0.8423** |
| Gradient Boosting | 0.7956 | 0.6378 | 0.5321 | 0.5802 | 0.8342 |
| SVM | 0.7459 | 0.5140 | 0.7834 | 0.6208 | 0.8260 |

### Selected model

**Random Forest** was selected as the best model based on the highest ROC-AUC (0.8423) and competitive F1-score for the churn class (0.6224). Selection was performed programmatically from the actual test-set results.

### Example customer predictions (Random Forest)

| Profile | Description | Churn Probability | Prediction |
|---|---|---|---|
| Profile A | New customer, month-to-month, fiber optic, no add-ons, electronic check | 82.1% | **CHURN** |
| Profile B | Mid-tenure, month-to-month, DSL, some add-ons | 31.8% | No Churn |
| Profile C | Long-tenure, two-year contract, many add-ons, auto-pay | 4.8% | No Churn |

---

## 9. Project Structure

```
Customer-Churn-Prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Original Kaggle dataset (unmodified)
│
├── notebook/
│   └── customer_churn_prediction.ipynb          # Main Jupyter Notebook
│
├── requirements.txt                             # Python dependencies
└── README.md                                    # This file
```

---

## 10. Installation

```bash
git clone <repository-url>
cd Customer-Churn-Prediction
pip install -r requirements.txt
```

> Replace `<repository-url>` with the actual GitHub repository URL once the repository is created.

---

## 11. Running the Project

1. Ensure all dependencies are installed (see Installation above).
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Navigate to `notebook/customer_churn_prediction.ipynb` and open it.
4. Select **Kernel → Restart & Run All** to execute the notebook from first cell to last.

All outputs, charts, model metrics, and predictions will be generated automatically from the data.

---

## 12. Key Insights

The following insights are derived directly from the data analysis performed in the notebook.

**Contract type:**
Month-to-month customers have a churn rate of **42.7%**, compared to **11.3%** for one-year contracts and **2.8%** for two-year contracts. Contract type is one of the most differentiated factors in the dataset.

**Customer tenure:**
Customers in their first year (0–12 months) have a churn rate of **47.4%**. This drops to **28.7%** for months 13–24, **20.4%** for months 25–48, and **9.5%** for months 49–72. Newer customers are at significantly higher risk.

**Internet service:**
Fiber optic customers churn at **41.9%**, compared to **19.0%** for DSL customers and **7.4%** for customers with no internet service.

**Payment method:**
Electronic check customers have the highest churn rate at **45.3%**, compared to **19.1%** for mailed check, **16.7%** for bank transfer (automatic), and **15.2%** for credit card (automatic).

**Add-on services:**
Customers with no add-on services churn at **20.8%**. The churn rate rises for customers with 1–2 add-ons, suggesting that the type and combination of services — not just the count — matters for retention.

---

## 13. Limitations

- **Cross-sectional data only** — the dataset is a single snapshot with no time dimension. It cannot capture how customer behaviour changes over time.
- **No customer lifetime value** — all customers are treated equally regardless of revenue contribution.
- **Class imbalance** — the 26.5/73.5 class split is addressed via `class_weight='balanced'`. Oversampling techniques such as SMOTE were not applied.
- **Manual hyperparameters** — model hyperparameters were set manually. No systematic hyperparameter search (e.g. GridSearchCV) was performed.
- **No external validation** — the model was evaluated on a single train/test split, not cross-validated.

---

## 14. Future Improvements

- **Hyperparameter tuning** with `GridSearchCV` or `RandomizedSearchCV` to optimise each model.
- **SHAP values** for granular, per-prediction explainability beyond global feature importances.
- **SMOTE oversampling** to address class imbalance at the data level.
- **Cross-validation** (e.g. stratified k-fold) for more robust model evaluation.
- **Cost-sensitive modelling** incorporating customer lifetime value to weight false negatives appropriately.
- **Deployment** as a REST API using Flask or FastAPI for real-time churn scoring.

---

## 15. Author

**Satyajit Panda**

B.Tech Computer Science and Engineering (Internet of Things)

*AICTE + IBM SkillsBuild Data Analytics with AI Internship*

---

## License

This project is submitted as part of an academic internship programme. The dataset is sourced from Kaggle and is subject to its original terms of use.

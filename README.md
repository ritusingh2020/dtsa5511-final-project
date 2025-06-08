# Home Equity Loan Default Prediction

This project aims to predict the likelihood of a home equity loan default using deep learning models, with a strong focus on balancing financial risk and opportunity. It tries to apply advanced machine learning concepts—especially model evaluation beyond simple accuracy.

---

##  Project Objectives

### Business Objective
- **Minimize financial loss** from bad loans (defaults or severe delinquencies)
- **Preserve revenue** from good loans by avoiding unnecessary rejections

### Technical Objective
- Build and evaluate models that **maximize recall for both classes**: bad loans (`BAD = 1`) and good loans (`BAD = 0`)
- Handle **imbalanced data** using class weighting and SMOTE
- Avoid overfitting while improving generalization and interpretability

---

## Dataset Overview

- **Source:** Home Equity Loan dataset (`hmeq.csv`)
- **Records:** 5,960 loan applicants
- **Target Variable:** `BAD` (1 = Defaulted, 0 = Repaid)
- **Features:** 12 financial and personal attributes including loan amount, credit history, job type, and debt-to-income ratio
- ~20% of the loans are labeled as `BAD = 1`

---

## Project Workflow

### 1. **Data Preparation**
- Missing value imputation (median/mode)
- Outlier handling (IQR method)
- Encoding of categorical variables
- Feature scaling using StandardScaler

### 2. **Exploratory Data Analysis (EDA)**
- Distribution analysis for numerical and categorical features
- Correlation matrix and multicollinearity review
- Bivariate analysis with respect to the `BAD` target

### 3. **Model Development**
- Baseline: Logistic Regression
- Deep Learning Models (Keras Sequential API):
  - Layer variations (64-32, 128-64-32, etc.)
  - Use of dropout and ReLU activations
  - Optional additional layer
  - Sigmoid output layer
- Class weighting and SMOTE for imbalance handling
- Multiple model versions tested with different configurations

### 4. **Evaluation Metrics**
- Accuracy
- Recall (with emphasis on positive class)
- Train-test recall drop monitoring
- Business impact analysis (missed good loans vs. prevented bad loans)



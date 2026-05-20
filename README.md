# Customer Churn Prediction Using Decision Tree
# GROUP 5

## Project Overview

This project aims to predict customer churn using Machine Learning techniques, specifically the Decision Tree Classifier.

The project includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Decision Tree architecture visualization
- Model training
- Model optimization
- Final model evaluation

The dataset used is the Telco Customer Churn dataset.

---

# Project Structure

```text
KD34403-ML-PROJECT/
│
├── data/
│   ├── Telco_Customer_Churn.csv
│   └── cleaned_data.csv
│
├── src/
│   └── main.py
│
├── churn_distribution.png
├── tenure_distribution.png
├── monthlycharges_distribution.png
├── decision_tree_visualization.png
│
├── README.md
└── requirements.txt
```

---

# Dataset

Dataset used:
- Telco Customer Churn Dataset

Dataset location:

```text
data/Telco_Customer_Churn.csv
```

The dataset contains customer information such as:
- Customer tenure
- Monthly charges
- Total charges
- Internet services
- Contract types
- Customer churn status

---

# Machine Learning Workflow

## Milestone 1 — Data Pipeline
Contributor:
- Charmaine Chia Yun Shan (BI23110106)

Tasks:
- Data cleaning
- Handling missing values
- Feature encoding
- Feature scaling
- Train-test split
- Exploratory Data Analysis

---

## Milestone 2 — Architecture Logic
Contributor:
- Nurshurayani binti Samsudin (BI23110065)

Tasks:
- Decision Tree architecture design
- Decision Tree visualization
- Feature interpretation

---

## Milestone 3 — Model Training
Contributor:
- Yusrina binti Mohammad Yuseri (BI23110273)

Tasks:
- Model training
- Performance evaluation
- Confusion matrix analysis

---

## Milestone 4 — Model Optimization
Contributor:
- Esther Christine Jude Valentine (BI23110060)

Tasks:
- Decision Tree regularization
- Overfitting reduction
- Hyperparameter tuning

---

## Milestone 5 — Final Model Evaluation
Contributor:
- Nurnisriza binti De Afendi (BI23110047)

Tasks:
- Final model testing
- ROC Curve analysis
- Precision, Recall, F1-score evaluation
- Error analysis

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
- VS Code
- GitHub

---

# Requirements

Install required libraries using:

```python
pip install -r requirements.txt
```

---

# How to Run the Project in Google Colab

## Step 1 — Open Google Colab

Open:

https://colab.research.google.com

---

## Step 2 — Clone the repository

Run:

```python
!git clone https://github.com/serinayoung/KD34403-ML-Project.git
%cd KD34403-ML-Project
```

---

## Step 3 — Install dependencies

Run:

```python
!pip install -r requirements.txt
```

---

## Step 4 — Run the main pipeline

Run:

```python
!python src/main.py
```

---

# Expected Outputs

The program will generate:
- Printed logs for each milestone (EDA, cleaning, training, optimization, evaluation).
- Saved plots: churn_distribution.png, tenure_distribution.png, monthlycharges_distribution.png, decision_tree_visualization.png.
- Confusion matrix and ROC curve plots.
- Final metrics: Accuracy, Precision, Recall, F1, ROC‑AUC.

---

# Final Conclusion

The optimized Decision Tree model successfully predicts customer churn using customer behavior and subscription-related features.

Regularization techniques reduced overfitting and improved the model’s generalization performance.

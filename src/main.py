import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

#Setup
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("Libraries imported successfully.")
print(f"Random seed set to {RANDOM_STATE}")

# Problem Definition
print("\nPROBLEM DEFINITION")
print("-" * 50)
print("Business Question: Can we predict customer churn?")
print("ML Problem Type: Binary Classification")
print("Target Variable: Churn (Yes/No)")

# Data Collection
print("\nDATA COLLECTION")
print("-" * 50)

file_path = "data/Telco_Customer_Churn.csv"
df = pd.read_csv(file_path)

print(f"Dataset loaded successfully.")
print(f"Shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# Basic EDA
print("\nEXPLORATORY DATA ANALYSIS")
print("-" * 50)

print("\nTarget distribution:")
print(df["Churn"].value_counts())
print("\nTarget distribution (%):")
print(df["Churn"].value_counts(normalize=True) * 100)

# Churn count plot
plt.figure()
sns.countplot(x="Churn", data=df)
plt.title("Distribution of Churn")
plt.savefig("churn_distribution.png")
plt.show()

# Tenure distribution
plt.figure()
sns.histplot(df["tenure"], bins=30, kde=True)
plt.title("Distribution of Tenure")
plt.savefig("tenure_distribution.png")
plt.show()

# Monthly charges distribution
plt.figure()
sns.histplot(df["MonthlyCharges"], bins=30, kde=True)
plt.title("Distribution of Monthly Charges")
plt.savefig("monthlycharges_distribution.png")
plt.show()

# Data Cleaning
print("\nDATA CLEANING")
print("-" * 50)

# Drop customerID
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)
    print("Dropped column: customerID")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Converted TotalCharges to numeric.")

print("\nMissing values after conversion:")
print(df.isnull().sum())

# Drop rows with missing values
before_rows = df.shape[0]
df.dropna(inplace=True)
after_rows = df.shape[0]

print(f"\nRows before dropping missing values: {before_rows}")
print(f"Rows after dropping missing values: {after_rows}")
print(f"Rows removed: {before_rows - after_rows}")


#Data Preprocessing
print("\nDATA PREPROCESSING")
print("-" * 50)

# Convert target variable to binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print("Converted Churn into binary values.")

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)
print("Applied one-hot encoding to categorical variables.")

print(f"\nProcessed feature shape: {X.shape}")
print("Processed feature columns:")
print(X.columns.tolist())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("\nTrain-test split completed.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Scale numeric features
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print("Scaled numeric columns: tenure, MonthlyCharges, TotalCharges")

# Save cleaned data
df.to_csv("data/cleaned_data.csv", index=False)
print("\nCleaned dataset saved as cleaned_data.csv")

print("\nMilestone 1 pipeline completed successfully.")


# =====================================================
# MODEL TRAINING
# =====================================================
# >> Contributor: Nurshurayani binti Samsudin (BI23110065)
print("\nMODEL TRAINING")
print("=" * 50)

# Train Decision Tree model
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))

# >> Contributor: Yusrina binti Mohammad Yuseri (BI23110273)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix - Model Training")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

print("\nModel training completed successfully.")

# >> Contributor: Esther Christine (BI23110060)
# MODEL OPTIMIZATION 
print("\nMODEL OPTIMIZATION")
print("=" * 50)

# Train optimized model
optimized_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

optimized_model.fit(X_train, y_train)

# Training accuracy
y_train_pred = optimized_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Testing accuracy
y_test_pred = optimized_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n=== Regularized Decision Tree ===")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_test_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Optimized Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("\nModel is regularized using max_depth, min_samples_split, and min_samples_leaf to reduce overfitting.")

# FINAL MODEL EVALUATION
print("\nFINAL MODEL EVALUATION")
print("=" * 50)

final_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# Train finalized model
final_model.fit(X_train, y_train)

print("\nFinal optimized model trained successfully.")

# TEST MODEL ON NEW DATA (TEST SET)

y_pred = final_model.predict(X_test)

# Probability prediction for ROC-AUC
y_prob = final_model.predict_proba(X_test)[:, 1]

# MODEL EVALUATION METRICS

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n====================================")
print("FINAL MODEL EVALUATION RESULTS")
print("====================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# CLASSIFICATION REPORT

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix - Final Model")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.show()

# ROC CURVE

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Final Model")

plt.legend()
plt.tight_layout()
plt.show()

# ERROR ANALYSIS

print("\n====================================")
print("ERROR ANALYSIS")
print("====================================")

# False Positives
false_positive = ((y_test == 0) & (y_pred == 1)).sum()

# False Negatives
false_negative = ((y_test == 1) & (y_pred == 0)).sum()

print(f"False Positives : {false_positive}")
print(f"False Negatives : {false_negative}")

print("\nAll pipelines completed successfully")


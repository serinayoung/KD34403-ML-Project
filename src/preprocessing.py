import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

print("\nMilestone 1 pipeline completed successfully.")

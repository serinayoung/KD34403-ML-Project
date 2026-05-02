import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# Load cleaned data
# =========================
df = pd.read_csv("data/cleaned_data.csv")

# =========================
# Split features and target
# =========================
X = df.drop("Churn", axis=1)
y = df["Churn"]

# =========================
# Convert categorical variables to numeric
# =========================
X = pd.get_dummies(X, drop_first=True)

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Train model (Milestone 4 – Regularized Decision Tree)
# =========================
model = DecisionTreeClassifier(
    max_depth=3,            # Reduce model complexity
    min_samples_split=10,   # Prevent splits on small noisy samples
    min_samples_leaf=5,     # Avoid overfitting to individual samples
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# Training accuracy (overfitting check)
# =========================
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# =========================
# Testing accuracy
# =========================
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n=== Milestone 4: Regularized Decision Tree ===")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nClassification Report (Test Data):\n")
print(classification_report(y_test, y_test_pred))

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Milestone 4)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
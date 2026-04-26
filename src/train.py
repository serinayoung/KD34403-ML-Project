# train.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Training Loop Simulation
# =========================
# Logistic Regression supports iterative training via max_iter
train_accuracies = []
test_accuracies = []
iterations = [10, 50, 100, 200, 300]

for iters in iterations:
    model = LogisticRegression(max_iter=iters, random_state=42)
    model.fit(X_train, y_train)

    # Accuracy on train and test
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"Iteration {iters}: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")

# =========================
# Plot Training Progress
# =========================
plt.figure()
plt.plot(iterations, train_accuracies, marker="o", label="Train Accuracy")
plt.plot(iterations, test_accuracies, marker="o", label="Test Accuracy")
plt.title("Training Progress over Iterations")
plt.xlabel("Iterations (max_iter)")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_progress.png")
plt.show()

# =========================
# Final Model Evaluation
# =========================
final_model = LogisticRegression(max_iter=300, random_state=42)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

print("\nFinal Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Final Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix_final.png")
plt.show()

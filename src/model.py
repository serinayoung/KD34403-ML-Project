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
 X, y, test_size=0.2, random_state=42, stratify=y
 )
 
 # =========================
 # Train model
 # =========================
 model = DecisionTreeClassifier(max_depth=5, random_state=42)
 model.fit(X_train, y_train)
 
 # =========================
 # Predictions
 # =========================
 y_pred = model.predict(X_test)
 
 # =========================
 # Evaluation
 # =========================
 print("Accuracy:", accuracy_score(y_test, y_pred))
 print("\nClassification Report:\n", classification_report(y_test, y_pred))
 
 # =========================
 # Confusion Matrix
 # =========================
 cm = confusion_matrix(y_test, y_pred)
 
 plt.figure()
 sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
 plt.title("Confusion Matrix")
 plt.xlabel("Predicted")
 plt.ylabel("Actual")
 plt.show()

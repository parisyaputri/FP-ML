import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# --- Step 1: Read and Prepare Data ---
dataset = pd.read_csv("data.csv", encoding='latin1')

# Calculate total purchase per customer
dataset["TotalSpent"] = dataset["Quantity"] * dataset["UnitPrice"]
spending_data = dataset.groupby("CustomerID")["TotalSpent"].sum().reset_index()

# --- Step 2: Define Spending Tiers ---
def classify_spender(amount):
    if amount < 3000:
        return "Low Spender"
    elif amount < 15000:
        return "Medium Spender"
    return "High Spender"

spending_data["SpenderType"] = spending_data["TotalSpent"].apply(classify_spender)

# --- Step 3: Feature & Target Extraction ---
features = spending_data[["TotalSpent"]]
labels = spending_data["SpenderType"]
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# To collect results
model_results = {}

# --- Step 4: Model A - Default Decision Tree (No Scaling) ---
tree_default = DecisionTreeClassifier(criterion="entropy", random_state=42)
tree_default.fit(x_train, y_train)
pred_default = tree_default.predict(x_test)

model_results["Unscaled Decision Tree"] = {
    "Accuracy": accuracy_score(y_test, pred_default),
    "Precision": precision_score(y_test, pred_default, average="macro"),
    "Recall": recall_score(y_test, pred_default, average="macro"),
    "F1": f1_score(y_test, pred_default, average="macro")
}

# --- Step 5: Model B - Scaled Inputs ---
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

tree_scaled = DecisionTreeClassifier(criterion="entropy", random_state=42)
tree_scaled.fit(x_train_scaled, y_train)
pred_scaled = tree_scaled.predict(x_test_scaled)

model_results["Scaled Decision Tree"] = {
    "Accuracy": accuracy_score(y_test, pred_scaled),
    "Precision": precision_score(y_test, pred_scaled, average="macro"),
    "Recall": recall_score(y_test, pred_scaled, average="macro"),
    "F1": f1_score(y_test, pred_scaled, average="macro")
}

# --- Step 6: Model C - Tuned with Grid Search ---
param_grid = {
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10]
}

grid_cv = GridSearchCV(
    estimator=DecisionTreeClassifier(criterion="entropy", random_state=42),
    param_grid=param_grid,
    cv=10,
    scoring="accuracy"
)

grid_cv.fit(x_train_scaled, y_train)
best_tree = grid_cv.best_estimator_
pred_tuned = best_tree.predict(x_test_scaled)

model_results["Grid Search Tuned Tree"] = {
    "Accuracy": accuracy_score(y_test, pred_tuned),
    "Precision": precision_score(y_test, pred_tuned, average="macro"),
    "Recall": recall_score(y_test, pred_tuned, average="macro"),
    "F1": f1_score(y_test, pred_tuned, average="macro")
}

# --- Step 7: Print Results Summary ---
print("\n=== Model Comparison Summary ===")
summary_df = pd.DataFrame(model_results).T
print(summary_df)

# --- Step 8: Plot Best Tree ---
plt.figure(figsize=(12, 8))
plot_tree(best_tree, feature_names=["TotalSpent (scaled)"],
          class_names=best_tree.classes_, filled=True)
plt.title("Optimized Decision Tree Structure")
plt.tight_layout()
plt.show()

# --- Step 9: Confusion Matrix ---
conf = confusion_matrix(y_test, pred_tuned)
plt.figure(figsize=(7, 6))
sns.heatmap(conf, annot=True, cmap="Blues", fmt="d",
            xticklabels=best_tree.classes_, yticklabels=best_tree.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Tuned Tree")
plt.tight_layout()
plt.show()

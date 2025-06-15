import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# === Load Dataset ===
data = pd.read_csv('data.csv', encoding='latin1')

# === Feature Engineering ===
data['TotalSpent'] = data['Quantity'] * data['UnitPrice']
grouped_data = data.groupby('CustomerID')['TotalSpent'].sum().reset_index()

# === Labeling ===
def assign_category(amount):
    if amount < 3000:
        return 'Low Spender'
    elif amount < 15000:
        return 'Medium Spender'
    return 'High Spender'

grouped_data['SpendingCategory'] = grouped_data['TotalSpent'].apply(assign_category)

# === Prepare Dataset ===
features = grouped_data[['TotalSpent']]
labels = grouped_data['SpendingCategory']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# === Store Result Metrics ===
evaluation = {}

# === 1. Base Decision Tree ===
clf_base = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_base.fit(X_train, y_train)
pred_base = clf_base.predict(X_test)

evaluation['Base DT (Unscaled)'] = {
    'Accuracy': accuracy_score(y_test, pred_base),
    'Precision (Macro)': precision_score(y_test, pred_base, average='macro'),
    'Recall (Macro)': recall_score(y_test, pred_base, average='macro'),
    'F1 Score (Macro)': f1_score(y_test, pred_base, average='macro')
}

# === 2. Scaled Features ===
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

clf_scaled = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_scaled.fit(X_train_std, y_train)
pred_scaled = clf_scaled.predict(X_test_std)

evaluation['DT with Scaling'] = {
    'Accuracy': accuracy_score(y_test, pred_scaled),
    'Precision (Macro)': precision_score(y_test, pred_scaled, average='macro'),
    'Recall (Macro)': recall_score(y_test, pred_scaled, average='macro'),
    'F1 Score (Macro)': f1_score(y_test, pred_scaled, average='macro')
}

# === 3. Grid Search Optimization ===
param_options = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(criterion='entropy', random_state=42),
    param_grid=param_options,
    cv=10,
    scoring='accuracy'
)

grid_search.fit(X_train_std, y_train)
best_clf = grid_search.best_estimator_
pred_best = best_clf.predict(X_test_std)

evaluation['Optimized DT'] = {
    'Accuracy': accuracy_score(y_test, pred_best),
    'Precision (Macro)': precision_score(y_test, pred_best, average='macro'),
    'Recall (Macro)': recall_score(y_test, pred_best, average='macro'),
    'F1 Score (Macro)': f1_score(y_test, pred_best, average='macro')
}

# === Print Results ===
print("\nEvaluation Results Summary:")
print(pd.DataFrame(evaluation).T)

# === Visualize Best Decision Tree ===
plt.figure(figsize=(12, 8))
plot_tree(best_clf, feature_names=['TotalSpent (scaled)'], class_names=best_clf.classes_, filled=True)
plt.title('Optimized Decision Tree Visualization')
plt.show()

# === Confusion Matrix ===
conf_matrix = confusion_matrix(y_test, pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_clf.classes_, yticklabels=best_clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Optimized DT')
plt.show()

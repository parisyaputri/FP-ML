# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset (ubah nama jika berbeda)
df = pd.read_csv('data.csv', encoding='latin1')

# Hitung total pengeluaran
df['TotalSpent'] = df['Quantity'] * df['UnitPrice']

# Kategorikan pengeluaran
def categorize_spending(total_spent):
    if total_spent <= 3000:
        return 'low'
    elif total_spent <= 15000:
        return 'medium'
    else:
        return 'high'

df['SpendingCategory'] = df['TotalSpent'].apply(categorize_spending)

# Fitur & Label
features = df[['Quantity', 'UnitPrice']]
labels = df['SpendingCategory']

# Encode label
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Standarisasi fitur
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, labels_encoded,
    test_size=0.3, random_state=42, stratify=labels_encoded
)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Evaluasi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

print("=== Evaluation Metrics ===")
print(f"Accuracy         : {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro)   : {recall:.4f}")
print(f"F1 Score (macro) : {f1:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=[0, 1, 2],
    target_names=label_encoder.classes_,
    zero_division=1
))

# Confusion Matrix (normalized)
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Tampilkan probabilitas
low_idx = label_encoder.transform(['low'])[0]
medium_idx = label_encoder.transform(['medium'])[0]
high_idx = label_encoder.transform(['high'])[0]

probs_df = pd.DataFrame({
    'LowProb (%)': y_prob[:, low_idx] * 100,
    'MediumProb (%)': y_prob[:, medium_idx] * 100,
    'HighProb (%)': y_prob[:, high_idx] * 100
})
print("\n=== Contoh Probabilitas Prediksi ===")
print(probs_df.head())

# Visualisasi hasil klasifikasi
X_test_plot = pd.DataFrame(X_test, columns=['Quantity (scaled)', 'UnitPrice (scaled)'])
X_test_plot['PredictedCategory'] = label_encoder.inverse_transform(y_pred)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=X_test_plot, x='Quantity (scaled)', y='UnitPrice (scaled)', hue='PredictedCategory', palette='viridis')
plt.title('Predicted Spending Categories')
plt.show()

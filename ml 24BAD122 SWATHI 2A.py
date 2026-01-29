import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

df = pd.read_csv(r"C:\Users\namiy\Downloads\LICI - 10 minute data.csv")
open_col = [c for c in df.columns if 'open' in c.lower()][0]
close_col = [c for c in df.columns if 'close' in c.lower()][0]
df['Price_Movement'] = np.where(df[close_col] > df[open_col], 1, 0)
feature_candidates = ['high', 'low', 'volume']
features = [c for c in df.columns if any(f in c.lower() for f in feature_candidates)]
target = 'Price_Movement'
data = df[features + [target]]
data = data.fillna(data.mean())
X = data[features]
y = data[target]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
logreg = LogisticRegression(random_state=42, solver='liblinear')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:, 1]
print("\n--- Logistic Regression Performance ---")
print("Model Run By: swathi-24bad122")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-Score :", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - swathi-24bad122')
plt.legend(loc='lower right')
plt.show()
coef = pd.Series(logreg.coef_[0], index=features).sort_values()
plt.figure(figsize=(6, 4))
coef.plot(kind='barh')
plt.title("Feature Importance (Logistic Regression) - swathi-24bad122")
plt.xlabel("Coefficient Value")
plt.show()
param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)
print("\n--- Best Model Performance ---")
print("Model Run By: swathi-24bad122")
print("Best Parameters:", grid.best_params_)
print("Test Accuracy with Best Model:", accuracy_score(y_test, y_pred_best))

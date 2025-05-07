import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib

# Load preprocessed data
df = pd.read_csv('data/diabetic_data_mapped.csv')
X = df.drop(['readmitted', 'encounter_id', 'patient_nbr'], axis=1, errors='ignore')
y = df['readmitted']
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Use a random sample for tuning
sample_size = 10000
if len(X) > sample_size:
    X_tune, _, y_tune, _ = train_test_split(X, y, train_size=sample_size, stratify=y, random_state=42)
else:
    X_tune, y_tune = X, y

# Split sample for tuning
X_train_tune, X_test_tune, y_train_tune, y_test_tune = train_test_split(X_tune, y_tune, test_size=0.2, stratify=y_tune, random_state=42)

# Parameter grid
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.5, 1, 2]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=20, scoring='roc_auc', cv=3, verbose=2, n_jobs=-1, random_state=42)
search.fit(X_train_tune, y_train_tune)

print('Best parameters (sample tuning):', search.best_params_)
print('Best CV ROC-AUC (sample tuning):', search.best_score_)

# Now retrain best model on full training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
best_params = search.best_params_
best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
best_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
print('Test Accuracy:', accuracy_score(y_test, y_pred))
print('Test Precision:', precision_score(y_test, y_pred))
print('Test Recall:', recall_score(y_test, y_pred))
print('Test F1:', f1_score(y_test, y_pred))
print('Test Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
test_roc_auc = roc_auc_score(y_test, y_prob)
print('Test ROC-AUC:', test_roc_auc)

# Save the best model
joblib.dump(best_model, 'xgb_model_tuned.joblib')
print('Tuned XGBoost model saved as xgb_model_tuned.joblib') 
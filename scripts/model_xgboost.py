import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Load preprocessed data
df = pd.read_csv('data/diabetic_data_mapped.csv')

# Drop columns not for modeling (IDs)
X = df.drop(['readmitted', 'encounter_id', 'patient_nbr'], axis=1, errors='ignore')
y = df['readmitted']

# One-hot encode categorical variables
df_cat = X.select_dtypes(include=['object'])
X = pd.get_dummies(X, columns=df_cat.columns, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# Metrics
print('XGBoost Results:')
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('ROC-AUC:', roc_auc_score(y_test, y_prob)) 
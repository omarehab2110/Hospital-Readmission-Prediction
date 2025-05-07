import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# Load preprocessed data to get columns and categories
df = pd.read_csv('data/diabetic_data_mapped.csv')

# Drop target and IDs for input fields
input_cols = [col for col in df.columns if col not in ['readmitted', 'encounter_id', 'patient_nbr']]

# For categorical columns, get unique values
cat_cols = df[input_cols].select_dtypes(include=['object']).columns.tolist()
cat_values = {col: sorted(df[col].dropna().unique()) for col in cat_cols}

# For numerical columns, get min/max/mean for hints
num_cols = [col for col in input_cols if col not in cat_cols]
num_stats = {col: (df[col].min(), df[col].max(), df[col].mean()) for col in num_cols}

# Load or train model (XGBoost)
def get_model():
    try:
        model = joblib.load('xgb_model.joblib')
    except Exception:
        # Train if not found
        X = df.drop(['readmitted', 'encounter_id', 'patient_nbr'], axis=1, errors='ignore')
        y = df['readmitted']
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X, y)
        joblib.dump(model, 'xgb_model.joblib')
    return model

model = get_model()

# For one-hot encoding user input
all_dummies = pd.get_dummies(df[input_cols], columns=cat_cols, drop_first=True)
all_columns = all_dummies.columns

def predict_from_input(user_input):
    # user_input: dict of {col: value}
    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    # Add missing columns
    for col in all_columns:
        if col not in input_df:
            input_df[col] = 0
    input_df = input_df[all_columns]
    prob = model.predict_proba(input_df)[0,1]
    pred = int(prob >= 0.5)
    return pred, prob

def get_feature_importance():
    # Return top 10 features
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    return [(all_columns[i], importances[i]) for i in indices]

# --- GUI ---
class ReadmissionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Hospital Readmission Predictor')
        self.geometry('700x700')
        self.configure(bg='#f0f4f8')
        self.inputs = {}
        self.create_widgets()

    def create_widgets(self):
        title = tk.Label(self, text='Hospital Readmission Prediction', font=('Arial', 22, 'bold'), bg='#f0f4f8', fg='#2d415a')
        title.pack(pady=20)
        form = tk.Frame(self, bg='#f0f4f8')
        form.pack(pady=10)
        # Input fields
        for i, col in enumerate(input_cols):
            label = tk.Label(form, text=col, font=('Arial', 12), bg='#f0f4f8', anchor='w')
            label.grid(row=i, column=0, sticky='w', padx=5, pady=3)
            if col in cat_cols:
                cb = ttk.Combobox(form, values=cat_values[col], state='readonly', width=20)
                cb.grid(row=i, column=1, padx=5, pady=3)
                self.inputs[col] = cb
            else:
                minv, maxv, meanv = num_stats[col]
                entry = tk.Entry(form, width=22)
                entry.insert(0, str(meanv))
                entry.grid(row=i, column=1, padx=5, pady=3)
                self.inputs[col] = entry
        # Predict button
        btn = tk.Button(self, text='Predict Readmission', font=('Arial', 14, 'bold'), bg='#2d415a', fg='white', command=self.on_predict)
        btn.pack(pady=20)
        # Output
        self.result_label = tk.Label(self, text='', font=('Arial', 16, 'bold'), bg='#f0f4f8')
        self.result_label.pack(pady=10)
        # Feature importance
        self.feat_frame = tk.Frame(self, bg='#f0f4f8')
        self.feat_frame.pack(pady=10)
        self.feat_label = tk.Label(self.feat_frame, text='', font=('Arial', 12), bg='#f0f4f8')
        self.feat_label.pack()

    def on_predict(self):
        user_input = {}
        try:
            for col in input_cols:
                if col in cat_cols:
                    val = self.inputs[col].get()
                    if val == '':
                        raise ValueError(f'Missing value for {col}')
                    user_input[col] = val
                else:
                    val = float(self.inputs[col].get())
                    user_input[col] = val
        except Exception as e:
            messagebox.showerror('Input Error', str(e))
            return
        pred, prob = predict_from_input(user_input)
        if pred == 1:
            msg = f'Prediction: Readmitted (within 30 days)\nConfidence: {prob:.2%}'
            color = 'red'
        else:
            msg = f'Prediction: Not Readmitted\nConfidence: {1-prob:.2%}'
            color = 'green'
        self.result_label.config(text=msg, fg=color)
        # Feature importance
        feat_imp = get_feature_importance()
        feat_txt = 'Top Features:\n' + '\n'.join([f'{f}: {w:.3f}' for f, w in feat_imp])
        self.feat_label.config(text=feat_txt)

if __name__ == '__main__':
    app = ReadmissionApp()
    app.mainloop() 
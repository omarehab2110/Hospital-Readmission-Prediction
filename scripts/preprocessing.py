# Data Preprocessing Script for Hospital Readmission Prediction

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the mapping file for decoding IDs manually by sections
print('Loading mapping file manually...')

# Read specific sections of the CSV file
admission_type_mapping = pd.read_csv('data/IDs_mapping.csv', nrows=9, names=['id', 'description'])

# Skip to discharge_disposition_id section (lines 11-40 in the file)
discharge_disposition_mapping = pd.read_csv('data/IDs_mapping.csv', skiprows=10, nrows=30, names=['id', 'description'])

# Skip to admission_source_id section (lines 42-67 in the file)
admission_source_mapping = pd.read_csv('data/IDs_mapping.csv', skiprows=41, nrows=26, names=['id', 'description'])

# Function to decode IDs using the mapping file
def decode_ids(df, column, mapping_df):
    mapping_dict = dict(zip(mapping_df['id'], mapping_df['description']))
    df[column] = df[column].map(mapping_dict)
    return df

# Load a small subset of the data for initial exploration 
print('Loading a subset of the data for preprocessing setup...')
data_subset = pd.read_csv('data/diabetic_data.csv')

# Display initial data info
print('Initial Data Info:')
print(data_subset.info())

# Handle missing values (example strategy: fill with mode for categorical, mean for numerical)
print('Handling missing values...')
for column in data_subset.columns:
    if data_subset[column].dtype == 'object':
        data_subset[column].fillna(data_subset[column].mode()[0], inplace=True)
    else:
        data_subset[column].fillna(data_subset[column].mean(), inplace=True)

# Decode ID columns using mapping file
print('Decoding ID columns...')
data_subset = decode_ids(data_subset, 'admission_type_id', admission_type_mapping)
data_subset = decode_ids(data_subset, 'discharge_disposition_id', discharge_disposition_mapping)
data_subset = decode_ids(data_subset, 'admission_source_id', admission_source_mapping)

# Encode categorical features
print('Encoding categorical features...')
le = LabelEncoder()
categorical_cols = data_subset.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data_subset[col] = le.fit_transform(data_subset[col])

# Normalize numerical columns
print('Normalizing numerical columns...')
scaler = MinMaxScaler()
numerical_cols = data_subset.select_dtypes(include=['int64', 'float64']).columns
if len(numerical_cols) > 0:
    data_subset[numerical_cols] = scaler.fit_transform(data_subset[numerical_cols])

# Feature selection by correlation
print('Selecting features by correlation with the target (readmitted)...')
correlation_threshold = 0.05

# Ensure 'readmitted' is not dropped and is encoded as numeric for correlation
if 'readmitted' in data_subset.columns and data_subset['readmitted'].dtype == 'object':
    data_subset['readmitted'] = le.fit_transform(data_subset['readmitted'])

# Fix the 'readmitted' column to be binary: 1 for <30, 0 for others
readmit_map = {'<30': 1, '>30': 0, 'NO': 0, 1.0: 1, 0.5: 0, 0.0: 0}
data_subset['readmitted'] = data_subset['readmitted'].map(readmit_map).astype(int)

correlations = data_subset.corr()['readmitted'].abs()
selected_features = correlations[correlations > correlation_threshold].index.tolist()

print(f'Selected features: {selected_features}')
data_subset = data_subset[selected_features]

# Save the preprocessed subset for reference
print('Saving preprocessed subset...')
data_subset.to_csv('preprocessed_subset.csv', index=False)

print('Preprocessing script setup complete. This script processes only a subset of data.')
print('To process the full dataset, modify the script to handle larger data in chunks or use the full file.')

# Show a sample after mapping
data_subset.head().to_csv('data/diabetic_data_sample_mapped.csv', index=False)
print('\nSample with mapped columns saved to data/diabetic_data_sample_mapped.csv')

# Save the entire mapped dataset with corrected 'readmitted'
data_subset.to_csv('data/diabetic_data_mapped.csv', index=False)
print('Full mapped dataset saved to data/diabetic_data_mapped.csv') 
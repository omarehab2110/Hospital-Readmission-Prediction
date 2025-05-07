import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import os

# --- CONFIG ---
DATA_PATH = os.path.join('data', 'diabetic_data.csv')
MAPPING_PATH = os.path.join('data', 'IDs_mapping.csv')
CHUNK_SIZE = 10000
PARQUET_PATH = os.path.join('data', 'diabetic_data_cleaned.parquet')
CSV_PATH = os.path.join('data', 'diabetic_data_cleaned.csv')

# --- 1. Load Mapping Dictionaries ---
def load_mapping(section):
    mapping = {}
    lines = []
    with open(MAPPING_PATH) as f:
        found = False
        for line in f:
            if line.strip() == section:
                found = True
                continue
            if found:
                if line.strip() == '':
                    break
                lines.append(line.strip().split(','))
    for row in lines:
        if len(row) == 2 and row[0] and row[1]:
            mapping[row[0]] = row[1]
    return mapping

admission_type_map = load_mapping('admission_type_id,description')
discharge_disposition_map = load_mapping('discharge_disposition_id,description')
admission_source_map = load_mapping('admission_source_id,description')

# --- 2. Process Data in Chunks ---
def map_column(df, col, mapping):
    df[col + '_desc'] = df[col].astype(str).map(mapping).fillna('Unknown')
    return df

chunks = []
for chunk in pd.read_csv(DATA_PATH, na_values='?', low_memory=False, chunksize=CHUNK_SIZE):
    # Handle missing values
    cat_cols = chunk.select_dtypes(include=['object']).columns.tolist()
    num_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
    for col in cat_cols:
        chunk[col] = chunk[col].fillna('Unknown')
    for col in num_cols:
        chunk[col] = chunk[col].fillna(chunk[col].median())
    # Map ID columns
    chunk = map_column(chunk, 'admission_type_id', admission_type_map)
    chunk = map_column(chunk, 'discharge_disposition_id', discharge_disposition_map)
    chunk = map_column(chunk, 'admission_source_id', admission_source_map)
    chunks.append(chunk)

# --- 3. Concatenate All Chunks ---
df = pd.concat(chunks, ignore_index=True)

# --- 4. Drop Useless Columns ---
# Drop original ID columns and columns with only one unique value
drop_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
drop_cols += [col for col in df.columns if df[col].nunique() <= 1]
df = df.drop(columns=drop_cols)

# --- 5. Encode Categorical Features Efficiently ---
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col != 'readmitted']
low_card_cols = [col for col in categorical_cols if df[col].nunique() <= 10]
high_card_cols = [col for col in categorical_cols if df[col].nunique() > 10]

# One-hot encode low-cardinality categoricals
if low_card_cols:
    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[low_card_cols])
    encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded, columns=encoder.get_feature_names_out(low_card_cols))
    df = pd.concat([df.drop(low_card_cols, axis=1).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Label encode high-cardinality categoricals
for col in high_card_cols:
    df[col] = df[col].astype('category').cat.codes

# --- 6. Scale Numerical Features ---
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# --- 7. Outlier Handling ---
for col in num_cols:
    df[col] = np.clip(df[col], -4, 4)

# --- 8. Feature Selection: Remove Highly Correlated Features ---
corr_matrix = df[num_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
df = df.drop(to_drop, axis=1)

# --- 9. Downcast Numeric Types ---
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')
for col in df.select_dtypes(include=['int64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='integer')

# --- 10. Save Cleaned Data ---
try:
    df.to_parquet(PARQUET_PATH, index=False)
except Exception as e:
    print(f"Could not save as Parquet: {e}")
df.to_csv(CSV_PATH, index=False)

print(f"Preprocessing complete. Saved to {PARQUET_PATH} and {CSV_PATH}.")

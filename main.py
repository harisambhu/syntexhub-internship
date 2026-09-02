import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("Housing.csv")

print("HOUSE PRICE PREDICTION")
print("=" * 40)

# ==========================================
# 2. BASIC INFORMATION
# ==========================================

print("\nDataset Shape:")
print(data.shape)

print("\nFirst 5 Rows:")
print(data.head())

print("\nColumn Names:")
print(data.columns.tolist())

# ==========================================
# 3. CHECK DATA TYPES
# ==========================================

print("\nData Types:")
print(data.dtypes)

# ==========================================
# 4. CHECK MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(data.isnull().sum())

# ==========================================
# 5. CHECK DUPLICATES
# ==========================================

print("\nDuplicate Rows:")
print(data.duplicated().sum())

# ==========================================
# 6. STATISTICAL SUMMARY
# ==========================================

print("\nStatistical Summary:")
print(data.describe())
print("\nDataset Shape:")
print(data.shape)
print("\nFirst 5 Rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nStatistical Summary:")
print(data.describe())
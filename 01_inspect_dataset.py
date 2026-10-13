import pandas as pd
from pathlib import Path

dataset_path = Path("dataset/final.csv")

df = pd.read_csv(dataset_path)

print("=" * 60)
print("MACHINE HEALTH MONITORING DATASET")
print("=" * 60)

print("\n1. Dataset Shape:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n2. Column Names:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print("\n3. First 10 Rows:")
print(df.head(10).to_string())

print("\n4. Data Types:")
print(df.dtypes)

print("\n5. Missing Values:")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("Dataset inspection completed successfully!")
print("=" * 60)
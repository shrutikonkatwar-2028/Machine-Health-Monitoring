import pandas as pd

print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# 1. Load the real dataset
df = pd.read_csv("dataset/final.csv")

print("\nOriginal dataset shape:")
print(df.shape)

# 2. Convert datetime column
df["datetime"] = pd.to_datetime(df["datetime"])

# 3. Remove completely empty column
df = df.drop(columns=["Tachometer_Gap_volt"])

print("\nRemoved: Tachometer_Gap_volt")

# 4. Create useful time features
df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute

# 5. Remove original datetime
df = df.drop(columns=["datetime"])

# 6. Handle missing numerical values
numeric_columns = df.select_dtypes(include=["number"]).columns

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())

# 7. Check dataset shape
print("\nCleaned dataset shape:")
print(df.shape)

# 8. Check status classes
print("\nStatus classes:")
print(df["status"].value_counts())

# 9. Save cleaned dataset
df.to_csv("dataset/cleaned_machine_data.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("Location: dataset/cleaned_machine_data.csv")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)
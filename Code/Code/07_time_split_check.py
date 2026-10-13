import pandas as pd

print("=" * 60)
print("TIME-BASED DATA SPLIT CHECK")
print("=" * 60)

df = pd.read_csv("dataset/final.csv")

# Convert datetime back to datetime format
df["datetime"] = pd.to_datetime(df["datetime"])

# Sort according to actual time
df = df.sort_values("datetime").reset_index(drop=True)

# 80% training, 20% testing
split_index = int(len(df) * 0.80)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

print("\nTotal records:", len(df))
print("Training records:", len(train))
print("Testing records:", len(test))

print("\nTraining time period:")
print("From:", train["datetime"].min())
print("To  :", train["datetime"].max())

print("\nTesting time period:")
print("From:", test["datetime"].min())
print("To  :", test["datetime"].max())

print("\nTraining status distribution:")
print(train["status"].value_counts())

print("\nTesting status distribution:")
print(test["status"].value_counts())

print("\nTraining classes:")
print(sorted(train["status"].unique()))

print("\nTesting classes:")
print(sorted(test["status"].unique()))

print("\n" + "=" * 60)
print("TIME-BASED SPLIT CHECK COMPLETED")
print("=" * 60)
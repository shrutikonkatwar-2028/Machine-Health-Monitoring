import pandas as pd

# Load dataset
df = pd.read_csv("dataset/final.csv")

print("=" * 60)
print("MACHINE STATUS ANALYSIS")
print("=" * 60)

# Count each machine status
status_counts = df["status"].value_counts()

print("\nStatus Classes:")
print(status_counts)

print("\nNumber of Different Status Classes:")
print(df["status"].nunique())

print("\nStatus Names:")
for status in df["status"].unique():
    print("-", status)

print("\nPercentage Distribution:")
print((df["status"].value_counts(normalize=True) * 100).round(2))

print("\n" + "=" * 60)
print("Status analysis completed!")
print("=" * 60)
import pandas as pd

print("=" * 60)
print("STATUS TIME-RANGE ANALYSIS")
print("=" * 60)

df = pd.read_csv("dataset/final.csv")

df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values("datetime")

print("\nStatus-wise time ranges:\n")

for status in df["status"].unique():
    data = df[df["status"] == status]

    print("-" * 60)
    print("Status:", status)
    print("Records:", len(data))
    print("Start:", data["datetime"].min())
    print("End  :", data["datetime"].max())

print("\n" + "=" * 60)
print("STATUS TIME ANALYSIS COMPLETED")
print("=" * 60)
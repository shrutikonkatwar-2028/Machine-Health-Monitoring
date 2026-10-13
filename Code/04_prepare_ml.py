import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("MACHINE HEALTH MONITORING - ML PREPARATION")
print("=" * 60)

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_machine_data.csv")

print("\nCleaned dataset shape:")
print(df.shape)

# Target column
target = "status"

# Features and target
X = df.drop(columns=[target, "hour", "minute"])
y = df[target]

print("\nNumber of features:", X.shape[1])
print("Target:", target)

print("\nTarget distribution:")
print(y.value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\n" + "=" * 60)
print("ML DATA PREPARATION COMPLETED")
print("=" * 60)
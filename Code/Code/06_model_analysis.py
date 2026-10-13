import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("=" * 60)
print("MODEL ANALYSIS")
print("=" * 60)

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_machine_data.csv")

# Features and target
X = df.drop(columns=["status", "hour", "minute"])
y = df["status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -------------------------------
# 1. Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "output/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -------------------------------
# 2. Feature Importance
# -------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance.head(10).to_string(index=False))

# Plot top 10 features
top10 = importance.head(10)

plt.figure(figsize=(10, 6))
plt.barh(
    top10["Feature"][::-1],
    top10["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importance - Random Forest")
plt.tight_layout()

plt.savefig(
    "output/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraphs saved successfully:")
print("1. output/confusion_matrix.png")
print("2. output/feature_importance.png")

print("\n" + "=" * 60)
print("MODEL ANALYSIS COMPLETED")
print("=" * 60)
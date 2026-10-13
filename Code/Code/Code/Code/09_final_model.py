# ============================================================
# MACHINE HEALTH MONITORING SYSTEM
# Using Temperature + Vibration Sensors
# 7-Day Health Analysis + Threshold Analysis + ML Model
# ============================================================

import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = "dataset/final.csv"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 75)
print(" MACHINE HEALTH MONITORING SYSTEM")
print(" Temperature + Vibration Sensor Analysis")
print("=" * 75)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

df["datetime"] = pd.to_datetime(df["datetime"])

print("\nDataset loaded successfully!")
print("Total records:", len(df))

print(
    "Date range:",
    df["datetime"].min().strftime("%d-%m-%Y"),
    "to",
    df["datetime"].max().strftime("%d-%m-%Y")
)


# ============================================================
# 3. SENSOR FEATURES
# ============================================================

temperature_features = [
    "Thermocouple_1_Temperature_F",
    "Thermocouple_2_Temperature_F",
    "Thermocouple_3_Temperature_F",
    "Thermocouple_4_Temperature_F"
]


vibration_features = [
    "Accelerometer_1_Crest_Factor_g_g",
    "Accelerometer_1_Derived_Peak_g",
    "Accelerometer_1_Peak-Peak_g",
    "Accelerometer_1_RMS_g",
    "Accelerometer_1_True_Peak_g",
    "Accelerometer_1_High_Frequency_g_rms",

    "Accelerometer_2_Crest_Factor_g_g",
    "Accelerometer_2_Derived_Peak_g",
    "Accelerometer_2_Peak-Peak_g",
    "Accelerometer_2_RMS_g",
    "Accelerometer_2_True_Peak_g",
    "Accelerometer_2_High_Frequency_g_rms",

    "Accelerometer_3_Crest_Factor_g_g",
    "Accelerometer_3_Derived_Peak_g",
    "Accelerometer_3_Peak-Peak_g",
    "Accelerometer_3_RMS_g",
    "Accelerometer_3_True_Peak_g",
    "Accelerometer_3_High_Frequency_g_rms",

    "Accelerometer_4_Crest_Factor_g_g",
    "Accelerometer_4_Derived_Peak_g",
    "Accelerometer_4_Peak-Peak_g",
    "Accelerometer_4_RMS_g",
    "Accelerometer_4_True_Peak_g",
    "Accelerometer_4_High_Frequency_g_rms"
]


vibration_rms_features = [
    "Accelerometer_1_RMS_g",
    "Accelerometer_2_RMS_g",
    "Accelerometer_3_RMS_g",
    "Accelerometer_4_RMS_g"
]


# ============================================================
# 4. CALCULATE MACHINE SENSOR VALUES
# ============================================================

df["Average_Temperature_F"] = df[temperature_features].mean(axis=1)

df["Average_Vibration_RMS_g"] = df[vibration_rms_features].mean(axis=1)

df["Date"] = df["datetime"].dt.date


# ============================================================
# 5. HEALTHY BASELINE
# ============================================================
# Baseline condition is used as the reference condition.
# Thresholds are derived statistically from healthy data.
# ============================================================

baseline_data = df[df["status"] == "baseline"].copy()

temperature_baseline_mean = baseline_data[
    "Average_Temperature_F"
].mean()

temperature_baseline_std = baseline_data[
    "Average_Temperature_F"
].std()

vibration_baseline_mean = baseline_data[
    "Average_Vibration_RMS_g"
].mean()

vibration_baseline_std = baseline_data[
    "Average_Vibration_RMS_g"
].std()


# ============================================================
# 6. THRESHOLDS
# ============================================================
# +4 SD = Warning
# +5 SD = Critical
#
# This avoids marking every small variation as critical.
# ============================================================

temperature_warning = (
    temperature_baseline_mean +
    4 * temperature_baseline_std
)

temperature_critical = (
    temperature_baseline_mean +
    5 * temperature_baseline_std
)

vibration_warning = (
    vibration_baseline_mean +
    4 * vibration_baseline_std
)

vibration_critical = (
    vibration_baseline_mean +
    5 * vibration_baseline_std
)


print("\n" + "=" * 75)
print(" HEALTHY BASELINE & THRESHOLDS")
print("=" * 75)

print(
    f"\nTemperature Healthy Baseline : "
    f"{temperature_baseline_mean:.2f} °F"
)

print(
    f"Temperature Warning Threshold : "
    f"{temperature_warning:.2f} °F"
)

print(
    f"Temperature Critical Threshold: "
    f"{temperature_critical:.2f} °F"
)

print(
    f"\nVibration Healthy Baseline : "
    f"{vibration_baseline_mean:.4f} g"
)

print(
    f"Vibration Warning Threshold : "
    f"{vibration_warning:.4f} g"
)

print(
    f"Vibration Critical Threshold: "
    f"{vibration_critical:.4f} g"
)


# ============================================================
# 7. 7-DAY REPORTING WINDOW
# ============================================================

start_date = pd.Timestamp("2025-08-01")
end_date = pd.Timestamp("2025-08-07")

seven_days = pd.date_range(
    start=start_date,
    end=end_date,
    freq="D"
)


# ============================================================
# 8. DAILY ANALYSIS
# ============================================================

daily_records = []

for current_date in seven_days:

    day_data = df[
        df["datetime"].dt.normalize() ==
        current_date
    ]

    # --------------------------------------------------------
    # NO DATA
    # --------------------------------------------------------

    if len(day_data) == 0:

        daily_records.append({
            "Date": current_date.date(),
            "Temperature_F": np.nan,
            "Vibration_g": np.nan,
            "Actual_Status": "NO DATA",
            "Temperature_Stage": "NO DATA",
            "Vibration_Stage": "NO DATA",
            "Overall_Stage": "NO DATA",
            "Health_Level": "NO DATA",
            "Threshold_Crossed": "NO DATA",
            "Records": 0
        })

        continue


    # --------------------------------------------------------
    # DAILY SENSOR VALUES
    # --------------------------------------------------------

    daily_temperature = day_data[
        "Average_Temperature_F"
    ].mean()

    daily_vibration = day_data[
        "Average_Vibration_RMS_g"
    ].mean()

    actual_status = day_data["status"].mode()[0]


    # --------------------------------------------------------
    # TEMPERATURE STAGE
    # --------------------------------------------------------

    if daily_temperature >= temperature_critical:

        temperature_stage = "CRITICAL"
        temperature_score = 3

    elif daily_temperature >= temperature_warning:

        temperature_stage = "WARNING"
        temperature_score = 2

    else:

        temperature_stage = "SAFE"
        temperature_score = 1


    # --------------------------------------------------------
    # VIBRATION STAGE
    # --------------------------------------------------------

    if daily_vibration >= vibration_critical:

        vibration_stage = "CRITICAL"
        vibration_score = 3

    elif daily_vibration >= vibration_warning:

        vibration_stage = "WARNING"
        vibration_score = 2

    else:

        vibration_stage = "SAFE"
        vibration_score = 1


    # --------------------------------------------------------
    # OVERALL HEALTH
    # --------------------------------------------------------

    overall_score = max(
        temperature_score,
        vibration_score
    )


    if overall_score == 3:

        overall_stage = "CRITICAL"
        health_level = "HIGH RISK"

    elif overall_score == 2:

        overall_stage = "WARNING"
        health_level = "ATTENTION"

    else:

        overall_stage = "SAFE"
        health_level = "NORMAL"


    # --------------------------------------------------------
    # THRESHOLD MESSAGE
    # --------------------------------------------------------

    crossed = []

    if temperature_stage == "CRITICAL":
        crossed.append("Temperature Critical")

    elif temperature_stage == "WARNING":
        crossed.append("Temperature Warning")


    if vibration_stage == "CRITICAL":
        crossed.append("Vibration Critical")

    elif vibration_stage == "WARNING":
        crossed.append("Vibration Warning")


    if len(crossed) == 0:
        threshold_crossed = "None"

    else:
        threshold_crossed = " + ".join(crossed)


    # --------------------------------------------------------
    # SAVE DAILY RECORD
    # --------------------------------------------------------

    daily_records.append({

        "Date": current_date.date(),

        "Temperature_F":
            daily_temperature,

        "Vibration_g":
            daily_vibration,

        "Actual_Status":
            actual_status,

        "Temperature_Stage":
            temperature_stage,

        "Vibration_Stage":
            vibration_stage,

        "Overall_Stage":
            overall_stage,

        "Health_Level":
            health_level,

        "Threshold_Crossed":
            threshold_crossed,

        "Records":
            len(day_data)
    })


# ============================================================
# 9. DAILY DATAFRAME
# ============================================================

daily_df = pd.DataFrame(daily_records)


# ============================================================
# 10. PRINT DAILY REPORT
# ============================================================

print("\n" + "=" * 75)
print(" 7-DAY MACHINE HEALTH REPORT")
print("=" * 75)

for _, row in daily_df.iterrows():

    print("\nDate:", row["Date"])

    if row["Overall_Stage"] == "NO DATA":

        print("Status: NO DATA")
        print("No sensor readings available for this date.")

    else:

        print(
            f"Temperature : "
            f"{row['Temperature_F']:.2f} °F"
        )

        print(
            f"Vibration   : "
            f"{row['Vibration_g']:.4f} g"
        )

        print(
            f"Actual Fault Status : "
            f"{row['Actual_Status']}"
        )

        print(
            f"Temperature Stage   : "
            f"{row['Temperature_Stage']}"
        )

        print(
            f"Vibration Stage     : "
            f"{row['Vibration_Stage']}"
        )

        print(
            f"Overall Health      : "
            f"{row['Overall_Stage']}"
        )

        print(
            f"Health Level        : "
            f"{row['Health_Level']}"
        )

        print(
            f"Threshold Crossed   : "
            f"{row['Threshold_Crossed']}"
        )


# ============================================================
# 11. SAVE DAILY REPORT
# ============================================================

daily_df.to_csv(
    f"{OUTPUT_DIR}/7_day_machine_health_report.csv",
    index=False
)

print(
    "\nDaily report saved:"
    " output/7_day_machine_health_report.csv"
)


# ============================================================
# 12. GRAPH 1
# TEMPERATURE HEALTH ANALYSIS
# ============================================================

plt.figure(figsize=(15, 8))

dates = pd.to_datetime(daily_df["Date"])

plt.plot(
    dates,
    daily_df["Temperature_F"],
    marker="o",
    linewidth=3,
    markersize=8,
    label="Daily Average Temperature"
)

plt.axhline(
    temperature_baseline_mean,
    linestyle=":",
    linewidth=2.5,
    label=f"Healthy Baseline ({temperature_baseline_mean:.2f} °F)"
)

plt.axhline(
    temperature_warning,
    linestyle="--",
    linewidth=2.5,
    label=f"Warning Threshold ({temperature_warning:.2f} °F)"
)

plt.axhline(
    temperature_critical,
    linestyle="-.",
    linewidth=3,
    label=f"Critical Threshold ({temperature_critical:.2f} °F)"
)


# Highlight data points
for i, row in daily_df.iterrows():

    if row["Overall_Stage"] == "NO DATA":

        plt.scatter(
            dates[i],
            temperature_baseline_mean,
            marker="X",
            s=180,
            color="gray",
            zorder=5
        )

        plt.text(
            dates[i],
            temperature_baseline_mean + 0.25,
            "NO DATA",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="gray"
        )

    elif row["Temperature_Stage"] == "CRITICAL":

        plt.scatter(
            dates[i],
            row["Temperature_F"],
            s=180,
            color="red",
            edgecolor="black",
            zorder=5
        )

    elif row["Temperature_Stage"] == "WARNING":

        plt.scatter(
            dates[i],
            row["Temperature_F"],
            s=180,
            color="orange",
            edgecolor="black",
            zorder=5
        )

    else:

        plt.scatter(
            dates[i],
            row["Temperature_F"],
            s=180,
            color="green",
            edgecolor="black",
            zorder=5
        )


plt.title(
    "7-Day Machine Temperature Health Analysis",
    fontsize=22,
    fontweight="bold"
)

plt.xlabel(
    "Date",
    fontsize=14
)

plt.ylabel(
    "Average Temperature (°F)",
    fontsize=14
)

plt.xticks(
    dates,
    dates.dt.strftime("%d %b")
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.25
)

plt.legend(
    loc="upper left",
    fontsize=11
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/7_day_temperature_health.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 13. GRAPH 2
# VIBRATION HEALTH ANALYSIS
# ============================================================

plt.figure(figsize=(15, 8))

plt.plot(
    dates,
    daily_df["Vibration_g"],
    marker="o",
    linewidth=3,
    markersize=8,
    label="Daily Average Vibration"
)

plt.axhline(
    vibration_baseline_mean,
    linestyle=":",
    linewidth=2.5,
    label=f"Healthy Baseline ({vibration_baseline_mean:.4f} g)"
)

plt.axhline(
    vibration_warning,
    linestyle="--",
    linewidth=2.5,
    label=f"Warning Threshold ({vibration_warning:.4f} g)"
)

plt.axhline(
    vibration_critical,
    linestyle="-.",
    linewidth=3,
    label=f"Critical Threshold ({vibration_critical:.4f} g)"
)


for i, row in daily_df.iterrows():

    if row["Overall_Stage"] == "NO DATA":

        plt.scatter(
            dates[i],
            vibration_baseline_mean,
            marker="X",
            s=180,
            color="gray",
            zorder=5
        )

        plt.text(
            dates[i],
            vibration_baseline_mean + 0.0008,
            "NO DATA",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="gray"
        )

    elif row["Vibration_Stage"] == "CRITICAL":

        plt.scatter(
            dates[i],
            row["Vibration_g"],
            s=180,
            color="red",
            edgecolor="black",
            zorder=5
        )

    elif row["Vibration_Stage"] == "WARNING":

        plt.scatter(
            dates[i],
            row["Vibration_g"],
            s=180,
            color="orange",
            edgecolor="black",
            zorder=5
        )

    else:

        plt.scatter(
            dates[i],
            row["Vibration_g"],
            s=180,
            color="green",
            edgecolor="black",
            zorder=5
        )


plt.title(
    "7-Day Machine Vibration Health Analysis",
    fontsize=22,
    fontweight="bold"
)

plt.xlabel(
    "Date",
    fontsize=14
)

plt.ylabel(
    "Average Vibration RMS (g)",
    fontsize=14
)

plt.xticks(
    dates,
    dates.dt.strftime("%d %b")
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.25
)

plt.legend(
    loc="upper left",
    fontsize=11
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/7_day_vibration_health.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. GRAPH 3
# OVERALL MACHINE HEALTH
# ============================================================

stage_numbers = {
    "NO DATA": 0,
    "SAFE": 1,
    "WARNING": 2,
    "CRITICAL": 3
}

daily_df["Stage_Number"] = daily_df[
    "Overall_Stage"
].map(stage_numbers)


plt.figure(figsize=(15, 8))


# ------------------------------------------------------------
# Background health zones
# ------------------------------------------------------------

plt.axhspan(
    0.5,
    1.5,
    alpha=0.12,
    color="green",
    label="Safe Zone"
)

plt.axhspan(
    1.5,
    2.5,
    alpha=0.12,
    color="orange",
    label="Warning Zone"
)

plt.axhspan(
    2.5,
    3.5,
    alpha=0.12,
    color="red",
    label="Critical Zone"
)


# ------------------------------------------------------------
# Health line
# ------------------------------------------------------------

plt.plot(
    dates,
    daily_df["Stage_Number"],
    linewidth=4,
    marker="o",
    markersize=11,
    color="black",
    label="Machine Health Stage"
)


# ------------------------------------------------------------
# Individual points
# ------------------------------------------------------------

for i, row in daily_df.iterrows():

    stage = row["Overall_Stage"]

    if stage == "SAFE":

        point_color = "green"

    elif stage == "WARNING":

        point_color = "orange"

    elif stage == "CRITICAL":

        point_color = "red"

    else:

        point_color = "gray"


    plt.scatter(
        dates[i],
        row["Stage_Number"],
        s=240,
        color=point_color,
        edgecolor="black",
        linewidth=1.5,
        zorder=5
    )


    if stage == "NO DATA":

        label = "NO DATA"

    else:

        label = stage


    plt.text(
        dates[i],
        row["Stage_Number"] + 0.18,
        label,
        ha="center",
        fontsize=11,
        fontweight="bold"
    )


plt.yticks(
    [0, 1, 2, 3],
    ["NO DATA", "SAFE", "WARNING", "CRITICAL"]
)

plt.ylim(
    -0.2,
    3.6
)

plt.xticks(
    dates,
    dates.dt.strftime("%d %b")
)

plt.xlabel(
    "Date",
    fontsize=14
)

plt.ylabel(
    "Machine Health Stage",
    fontsize=14
)

plt.title(
    "7-Day Machine Health Monitoring",
    fontsize=22,
    fontweight="bold"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.legend(
    loc="upper left",
    fontsize=11
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/7_day_machine_health_stage.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 15. MACHINE LEARNING MODEL
# ============================================================
# Only temperature + vibration features are used.
# Tachometer and voltage are intentionally excluded.
# ============================================================

ml_features = (
    temperature_features +
    vibration_features
)

X = df[ml_features]
y = df["status"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


model = Pipeline([

    (
        "imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "random_forest",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    )
])


print("\n" + "=" * 75)
print(" TRAINING MACHINE LEARNING MODEL")
print("=" * 75)

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)


# ============================================================
# 16. ML METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\nModel Performance")

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

classes = sorted(
    y.unique()
)


plt.figure(
    figsize=(9, 7)
)

plt.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues"
)

plt.title(
    "Machine Health Classification - Confusion Matrix",
    fontsize=18,
    fontweight="bold"
)

plt.colorbar()

plt.xticks(
    range(len(classes)),
    classes,
    rotation=25,
    ha="right"
)

plt.yticks(
    range(len(classes)),
    classes
)

plt.xlabel(
    "Predicted Status",
    fontsize=13
)

plt.ylabel(
    "Actual Status",
    fontsize=13
)


for i in range(len(classes)):

    for j in range(len(classes)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold"
        )


plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/machine_health_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 18. SAVE MODEL METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})


metrics_df.to_csv(
    f"{OUTPUT_DIR}/machine_learning_metrics.csv",
    index=False
)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print(" PROJECT OUTPUT FILES")
print("=" * 75)

print(
    "\n1. 7_day_machine_health_report.csv"
)

print(
    "2. 7_day_temperature_health.png"
)

print(
    "3. 7_day_vibration_health.png"
)

print(
    "4. 7_day_machine_health_stage.png"
)

print(
    "5. machine_health_confusion_matrix.png"
)

print(
    "6. machine_learning_metrics.csv"
)


print("\n" + "=" * 75)
print(" MACHINE HEALTH MONITORING ANALYSIS COMPLETED")
print("=" * 75)
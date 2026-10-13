import streamlit as st
import pandas as pd
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Machine Health Monitoring",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    h1 {
        color: #172033;
        font-weight: 800;
    }

    h2, h3 {
        color: #172033;
    }

    .subtitle {
        color: #64748b;
        font-size: 1.15rem;
        margin-bottom: 1.5rem;
    }

    .status-card {
        padding: 25px;
        border-radius: 16px;
        margin: 10px 0 25px 0;
        border-left: 7px solid;
    }

    .status-normal {
        background-color: #ecfdf5;
        border-left-color: #16a34a;
    }

    .status-warning {
        background-color: #fff7ed;
        border-left-color: #f59e0b;
    }

    .status-critical {
        background-color: #fef2f2;
        border-left-color: #ef4444;
    }

    .status-no-data {
        background-color: #f1f5f9;
        border-left-color: #64748b;
    }

    .status-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .status-description {
        font-size: 1rem;
        color: #475569;
    }

    .section-box {
        background-color: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }

    .recommendation-box {
        background-color: #fff7ed;
        border: 1px solid #fdba74;
        border-left: 6px solid #f97316;
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
        margin-bottom: 25px;
    }

    .summary-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 18px;
        border-radius: 12px;
        margin-top: 10px;
    }

    .small-note {
        color: #64748b;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "final.csv"
REPORT_PATH = BASE_DIR / "output" / "7_day_machine_health_report.csv"
METRICS_PATH = BASE_DIR / "output" / "machine_learning_metrics.csv"

TEMP_GRAPH = BASE_DIR / "output" / "7_day_temperature_health.png"
VIB_GRAPH = BASE_DIR / "output" / "7_day_vibration_health.png"
STAGE_GRAPH = BASE_DIR / "output" / "7_day_machine_health_stage.png"
CONFUSION_GRAPH = BASE_DIR / "output" / "machine_health_confusion_matrix.png"


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_report():
    df = pd.read_csv(REPORT_PATH)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.rename(columns={"date": "Date"})

    return df


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)


# ============================================================
# FILE CHECK
# ============================================================

required_files = [
    DATASET_PATH,
    REPORT_PATH,
    METRICS_PATH,
    TEMP_GRAPH,
    VIB_GRAPH,
    STAGE_GRAPH,
    CONFUSION_GRAPH
]

missing_files = [str(file) for file in required_files if not file.exists()]

if missing_files:

    st.error("Some required project files are missing.")

    st.write("Missing files:")

    for file in missing_files:
        st.code(file)

    st.info(
        "Run the machine analysis script again so that the required "
        "CSV reports and graphs are generated."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

df = load_dataset()
report_df = load_report()
metrics_df = load_metrics()


# ============================================================
# DATASET PROCESSING
# ============================================================

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

df["date_only"] = df["datetime"].dt.date


temperature_columns = [
    "Thermocouple_1_Temperature_F",
    "Thermocouple_2_Temperature_F",
    "Thermocouple_3_Temperature_F",
    "Thermocouple_4_Temperature_F"
]

vibration_columns = [
    "Accelerometer_1_RMS_g",
    "Accelerometer_2_RMS_g",
    "Accelerometer_3_RMS_g",
    "Accelerometer_4_RMS_g"
]


# ============================================================
# DAILY SENSOR DATA
# ============================================================

df["Avg_Temperature"] = df[temperature_columns].mean(axis=1)

df["Avg_Vibration_RMS"] = df[vibration_columns].mean(axis=1)


daily_sensor_data = (
    df.groupby("date_only")
    .agg(
        Avg_Temperature=("Avg_Temperature", "mean"),
        Avg_Vibration_RMS=("Avg_Vibration_RMS", "mean"),
        Records=("status", "count")
    )
    .reset_index()
)

daily_sensor_data["Date"] = pd.to_datetime(
    daily_sensor_data["date_only"]
)

daily_sensor_data = daily_sensor_data.drop(
    columns=["date_only"]
)


# ============================================================
# THRESHOLD VALUES
# ============================================================
# Dataset-derived thresholds based on healthy baseline.
#
# These are monitoring thresholds, NOT manufacturer safety limits.
# Warning = baseline + 4 standard deviations
# Critical = baseline + 5 standard deviations
# ============================================================

TEMP_BASELINE = 73.37
TEMP_WARNING = 76.47
TEMP_CRITICAL = 77.24

VIB_BASELINE = 0.0897
VIB_WARNING = 0.0937
VIB_CRITICAL = 0.0947


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_status(value, warning, critical):

    if pd.isna(value):
        return "NO DATA"

    if value >= critical:
        return "CRITICAL"

    if value >= warning:
        return "WARNING"

    return "NORMAL"


def get_status_emoji(status):

    if status == "CRITICAL":
        return "🔴"

    if status == "WARNING":
        return "🟠"

    if status == "NORMAL":
        return "🟢"

    return "⚪"


def calculate_sensor_score(value, warning, critical):

    """
    Threshold-based monitoring score.

    This is an indicator for dashboard visualization.
    It is NOT an ML metric and is not a certified engineering
    health score.
    """

    if pd.isna(value):
        return None

    if value <= warning:
        return 100.0

    if value <= critical:

        ratio = (
            (value - warning)
            / (critical - warning)
        )

        return max(50.0, 100.0 - (50.0 * ratio))

    # Above critical threshold
    ratio = (
        (value - critical)
        / critical
    )

    score = 50.0 - (100.0 * ratio)

    return max(0.0, min(50.0, score))


def get_health_class(score):

    if score is None:
        return "NO DATA"

    if score >= 80:
        return "NORMAL"

    if score >= 50:
        return "ATTENTION"

    return "HIGH RISK"


def safe_column(df, possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Dashboard Controls")

st.sidebar.markdown("### Monitoring Period")

available_dates = sorted(
    report_df["Date"].dropna().dt.date.unique()
)

if not available_dates:

    st.sidebar.error("No monitoring dates available.")
    st.stop()


selected_date = st.sidebar.selectbox(
    "Select Date",
    available_dates,
    index=len(available_dates) - 1
)


st.sidebar.markdown("---")

st.sidebar.markdown("### Sensor Information")

st.sidebar.write("🌡️ **Temperature Sensors:** 4")

st.sidebar.write("📳 **Accelerometers:** 4")

st.sidebar.write("📊 **Total Sensor Features:** 28")

st.sidebar.markdown("---")

st.sidebar.markdown("### Dataset Source")

st.sidebar.write(
    "UTK-ASL Digital Twin Predictive Maintenance Dataset"
)

st.sidebar.markdown(
    """
    <div class="small-note">
    Monitoring is based on vibration and temperature sensor data.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    "# ⚙️ Machine Health Monitoring System"
)

st.markdown(
    '<div class="subtitle">'
    'Temperature & Vibration Based Predictive Maintenance Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ============================================================
# SELECTED DATE DATA
# ============================================================

selected_rows = report_df[
    report_df["Date"].dt.date == selected_date
]


# ============================================================
# IF NO DATA
# ============================================================

if selected_rows.empty:

    st.markdown(
        """
        <div class="status-card status-no-data">

        <div class="status-title">
        ⚪ MACHINE HEALTH: NO DATA
        </div>

        <div class="status-description">
        No sensor readings are available for the selected date.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    selected_sensor = daily_sensor_data[
        daily_sensor_data["Date"].dt.date == selected_date
    ]

    if not selected_sensor.empty:

        pass

    st.stop()


# ============================================================
# SELECTED REPORT ROW
# ============================================================

selected_row = selected_rows.iloc[0]


# ============================================================
# FIND REPORT COLUMNS SAFELY
# ============================================================

temp_column = safe_column(
    report_df,
    [
        "Avg_Temperature",
        "Average_Temperature",
        "Temperature",
        "avg_temperature"
    ]
)

vib_column = safe_column(
    report_df,
    [
        "Avg_Vibration_RMS",
        "Average_Vibration_RMS",
        "Vibration_RMS",
        "avg_vibration"
    ]
)

stage_column = safe_column(
    report_df,
    [
        "Overall_Stage",
        "Machine_Stage",
        "Health_Stage",
        "Stage"
    ]
)

health_column = safe_column(
    report_df,
    [
        "Health_Level",
        "Health",
        "health_level"
    ]
)

condition_column = safe_column(
    report_df,
    [
        "Recorded_Condition",
        "Condition",
        "status"
    ]
)

message_column = safe_column(
    report_df,
    [
        "Threshold_Message",
        "Message",
        "Alert"
    ]
)


# ============================================================
# CURRENT SENSOR VALUES
# ============================================================

if temp_column:

    current_temperature = float(
        selected_row[temp_column]
    )

else:

    current_temperature = float(
        daily_sensor_data[
            daily_sensor_data["Date"].dt.date == selected_date
        ]["Avg_Temperature"].iloc[0]
    )


if vib_column:

    current_vibration = float(
        selected_row[vib_column]
    )

else:

    current_vibration = float(
        daily_sensor_data[
            daily_sensor_data["Date"].dt.date == selected_date
        ]["Avg_Vibration_RMS"].iloc[0]
    )


temperature_status = get_status(
    current_temperature,
    TEMP_WARNING,
    TEMP_CRITICAL
)

vibration_status = get_status(
    current_vibration,
    VIB_WARNING,
    VIB_CRITICAL
)


# ============================================================
# OVERALL MACHINE HEALTH
# ============================================================

if (
    temperature_status == "CRITICAL"
    or vibration_status == "CRITICAL"
):

    overall_status = "CRITICAL"

elif (
    temperature_status == "WARNING"
    or vibration_status == "WARNING"
):

    overall_status = "WARNING"

else:

    overall_status = "NORMAL"


# ============================================================
# RECORDED CONDITION
# ============================================================

if condition_column:

    recorded_condition = str(
        selected_row[condition_column]
    )

else:

    recorded_condition = "Not available"


# ============================================================
# MACHINE STAGE
# ============================================================

if stage_column:

    machine_stage = str(
        selected_row[stage_column]
    )

else:

    machine_stage = overall_status


# ============================================================
# CURRENT MACHINE STATUS CARD
# ============================================================

if overall_status == "CRITICAL":

    st.markdown(
        """
        <div class="status-card status-critical">

        <div class="status-title">
        🔴 MACHINE HEALTH: CRITICAL
        </div>

        <div class="status-description">
        One or more sensor parameters have crossed the critical
        monitoring threshold.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

elif overall_status == "WARNING":

    st.markdown(
        """
        <div class="status-card status-warning">

        <div class="status-title">
        🟠 MACHINE HEALTH: WARNING
        </div>

        <div class="status-description">
        Sensor readings have crossed the warning monitoring threshold.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="status-card status-normal">

        <div class="status-title">
        🟢 MACHINE HEALTH: NORMAL
        </div>

        <div class="status-description">
        Temperature and vibration readings are within the normal
        monitoring range.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SENSOR READINGS
# ============================================================

st.subheader(
    f"Sensor Readings — {selected_date}"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "🌡️ Temperature",
        f"{current_temperature:.2f} °F"
    )

with col2:

    st.metric(
        "📳 Vibration RMS",
        f"{current_vibration:.4f} g"
    )

with col3:

    st.metric(
        "Machine Stage",
        machine_stage
    )

with col4:

    st.metric(
        "Recorded Condition",
        recorded_condition
    )


# ============================================================
# SENSOR THRESHOLD ANALYSIS
# ============================================================

st.markdown("---")

st.subheader("🚨 Sensor Threshold Analysis")

st.caption(
    "Thresholds are derived from the healthy baseline in the "
    "selected dataset and are used for monitoring visualization."
)


# ------------------------------------------------------------
# TEMPERATURE
# ------------------------------------------------------------

st.markdown("### 🌡️ Temperature Monitoring")

temp_col1, temp_col2, temp_col3, temp_col4 = st.columns(4)

with temp_col1:

    st.metric(
        "Healthy Baseline",
        f"{TEMP_BASELINE:.2f} °F"
    )

with temp_col2:

    st.metric(
        "Warning Threshold",
        f"{TEMP_WARNING:.2f} °F"
    )

with temp_col3:

    st.metric(
        "Critical Threshold",
        f"{TEMP_CRITICAL:.2f} °F"
    )

with temp_col4:

    st.metric(
        "Current Value",
        f"{current_temperature:.2f} °F"
    )

st.write(
    f"**Temperature Status:** "
    f"{get_status_emoji(temperature_status)} "
    f"{temperature_status}"
)


# ------------------------------------------------------------
# VIBRATION
# ------------------------------------------------------------

st.markdown("### 📳 Vibration RMS Monitoring")

vib_col1, vib_col2, vib_col3, vib_col4 = st.columns(4)

with vib_col1:

    st.metric(
        "Healthy Baseline",
        f"{VIB_BASELINE:.4f} g"
    )

with vib_col2:

    st.metric(
        "Warning Threshold",
        f"{VIB_WARNING:.4f} g"
    )

with vib_col3:

    st.metric(
        "Critical Threshold",
        f"{VIB_CRITICAL:.4f} g"
    )

with vib_col4:

    st.metric(
        "Current Value",
        f"{current_vibration:.4f} g"
    )

st.write(
    f"**Vibration Status:** "
    f"{get_status_emoji(vibration_status)} "
    f"{vibration_status}"
)


# ============================================================
# THRESHOLD ALERT
# ============================================================

if (
    temperature_status == "CRITICAL"
    or vibration_status == "CRITICAL"
):

    critical_messages = []

    if temperature_status == "CRITICAL":

        critical_messages.append(
            "Temperature has exceeded the critical threshold."
        )

    if vibration_status == "CRITICAL":

        critical_messages.append(
            "Vibration RMS has exceeded the critical threshold."
        )

    st.error(
        "⚠️ **Critical Threshold Alert**\n\n"
        + " ".join(critical_messages)
    )


elif (
    temperature_status == "WARNING"
    or vibration_status == "WARNING"
):

    warning_messages = []

    if temperature_status == "WARNING":

        warning_messages.append(
            "Temperature has crossed the warning threshold."
        )

    if vibration_status == "WARNING":

        warning_messages.append(
            "Vibration RMS has crossed the warning threshold."
        )

    st.warning(
        "⚠️ **Warning Threshold Alert**\n\n"
        + " ".join(warning_messages)
    )

else:

    st.success(
        "✅ Temperature and vibration are within the normal "
        "monitoring range."
    )


# ============================================================
# MACHINE HEALTH SCORE
# ============================================================

st.markdown("---")

st.subheader("❤️ Machine Health Score")

temperature_score = calculate_sensor_score(
    current_temperature,
    TEMP_WARNING,
    TEMP_CRITICAL
)

vibration_score = calculate_sensor_score(
    current_vibration,
    VIB_WARNING,
    VIB_CRITICAL
)


if (
    temperature_score is not None
    and vibration_score is not None
):

    health_score = min(
        temperature_score,
        vibration_score
    )

    health_class = get_health_class(
        health_score
    )

    score_col1, score_col2, score_col3 = st.columns(3)

    with score_col1:

        st.metric(
            "Monitoring Health Score",
            f"{health_score:.0f} / 100"
        )

    with score_col2:

        st.metric(
            "Temperature Score",
            f"{temperature_score:.0f} / 100"
        )

    with score_col3:

        st.metric(
            "Vibration Score",
            f"{vibration_score:.0f} / 100"
        )

    if health_class == "NORMAL":

        st.success(
            "🟢 Overall monitoring indicator: NORMAL"
        )

    elif health_class == "ATTENTION":

        st.warning(
            "🟠 Overall monitoring indicator: ATTENTION"
        )

    else:

        st.error(
            "🔴 Overall monitoring indicator: HIGH RISK"
        )

    st.caption(
        "This score is a threshold-based dashboard indicator. "
        "It is not an ML accuracy score or a certified engineering "
        "health index."
    )


# ============================================================
# MAINTENANCE RECOMMENDATION
# ============================================================

st.markdown("---")

st.subheader("🔧 Maintenance Recommendation")


if overall_status == "CRITICAL":

    recommendation_text = """
    <div class="recommendation-box">

    <h4>🔴 Critical Monitoring Condition</h4>

    <p>
    One or more monitored sensor parameters have crossed the
    critical threshold.
    </p>

    <b>Recommended inspection actions:</b>

    <ul>
        <li>Inspect the machine before continued operation.</li>
        <li>Check for mechanical looseness or imbalance.</li>
        <li>Inspect bearings, coupling and shaft alignment.</li>
        <li>Review recent vibration and temperature trends.</li>
        <li>Schedule maintenance inspection based on operating conditions.</li>
    </ul>

    <p>
    <b>Note:</b> These are monitoring recommendations and do not
    represent a confirmed mechanical diagnosis.
    </p>

    </div>
    """

    st.markdown(
        recommendation_text,
        unsafe_allow_html=True
    )


elif overall_status == "WARNING":

    recommendation_text = """
    <div class="recommendation-box">

    <h4>🟠 Attention Required</h4>

    <p>
    One or more monitored parameters have crossed the warning
    threshold.
    </p>

    <ul>
        <li>Continue monitoring the machine closely.</li>
        <li>Check for changes in vibration or temperature.</li>
        <li>Review the recent health trend.</li>
        <li>Plan an inspection if the condition persists.</li>
    </ul>

    </div>
    """

    st.markdown(
        recommendation_text,
        unsafe_allow_html=True
    )


else:

    st.success(
        "🟢 Current sensor readings are within the normal "
        "monitoring range. Continue routine monitoring."
    )


# ============================================================
# MONITORING SUMMARY
# ============================================================

st.markdown("---")

st.subheader("📊 Monitoring Summary")

date_min = df["datetime"].min()
date_max = df["datetime"].max()

recorded_days = df["date_only"].nunique()

total_days = (
    date_max.date() - date_min.date()
).days + 1


summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:

    st.metric(
        "Total Sensor Records",
        f"{len(df):,}"
    )

with summary_col2:

    st.metric(
        "Recorded Days",
        recorded_days
    )

with summary_col3:

    st.metric(
        "Calendar Period",
        f"{total_days} days"
    )

with summary_col4:

    st.metric(
        "Sensor Features",
        "28"
    )


st.markdown(
    f"""
    <div class="summary-box">

    <b>Monitoring Period:</b>
    {date_min.strftime("%d %b %Y, %H:%M")}
    →
    {date_max.strftime("%d %b %Y, %H:%M")}

    <br><br>

    <b>Temperature Sensors:</b> 4

    &nbsp;&nbsp;&nbsp;

    <b>Accelerometers:</b> 4

    &nbsp;&nbsp;&nbsp;

    <b>Temperature Features:</b> 4

    &nbsp;&nbsp;&nbsp;

    <b>Vibration Features:</b> 24

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 7-DAY / MONITORING HEALTH TREND
# ============================================================

st.markdown("---")

st.subheader("📈 Machine Health Trend")

st.image(
    str(STAGE_GRAPH),
    width="stretch"
)


st.caption(
    "Daily health trend based on temperature and vibration "
    "threshold monitoring."
)


# ============================================================
# TEMPERATURE GRAPH
# ============================================================

st.markdown("---")

st.subheader("🌡️ Temperature Monitoring")

st.image(
    str(TEMP_GRAPH),
    width="stretch"
)


st.caption(
    "Daily average temperature compared with monitoring thresholds."
)


# ============================================================
# VIBRATION GRAPH
# ============================================================

st.markdown("---")

st.subheader("📳 Vibration RMS Monitoring")

st.image(
    str(VIB_GRAPH),
    width="stretch"
)


st.caption(
    "Daily average vibration RMS compared with monitoring thresholds."
)


# ============================================================
# DAILY HEALTH REPORT
# ============================================================

st.markdown("---")

st.subheader("📋 Daily Machine Health Report")

display_df = report_df.copy()

display_df["Date"] = display_df["Date"].dt.strftime(
    "%Y-%m-%d"
)

# Rename columns for user-friendly dashboard display

rename_map = {
    "Date": "Date",
    "Avg_Temperature": "Temperature (°F)",
    "Average_Temperature": "Temperature (°F)",
    "Avg_Vibration_RMS": "Vibration RMS (g)",
    "Average_Vibration_RMS": "Vibration RMS (g)",
    "Temperature_Stage": "Temperature Stage",
    "Vibration_Stage": "Vibration Stage",
    "Overall_Stage": "Overall Stage",
    "Machine_Stage": "Overall Stage",
    "Health_Level": "Health Level",
    "Recorded_Condition": "Recorded Condition",
    "Threshold_Message": "Threshold Message",
    "Records": "Records"
}

display_df = display_df.rename(
    columns=rename_map
)

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True
)


# ============================================================
# MACHINE LEARNING PERFORMANCE
# ============================================================

st.markdown("---")

st.subheader("🤖 Machine Learning Performance")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)


def get_metric_value(metric_name):

    possible_metric_columns = [
        "Metric",
        "metric",
        "Name",
        "name"
    ]

    possible_value_columns = [
        "Value",
        "value",
        "Score",
        "score"
    ]

    metric_col = None
    value_col = None

    for col in possible_metric_columns:

        if col in metrics_df.columns:

            metric_col = col
            break

    for col in possible_value_columns:

        if col in metrics_df.columns:

            value_col = col
            break

    if metric_col is None or value_col is None:

        return None

    match = metrics_df[
        metrics_df[metric_col]
        .astype(str)
        .str.lower()
        .str.contains(
            metric_name.lower(),
            regex=False
        )
    ]

    if match.empty:

        return None

    try:

        value = float(
            match.iloc[0][value_col]
        )

        # Handle percentage values
        if value <= 1:

            value = value * 100

        return value

    except:

        return None


accuracy = get_metric_value("accuracy")

precision = get_metric_value("precision")

recall = get_metric_value("recall")

f1 = get_metric_value("f1")


with metric_col1:

    if accuracy is not None:

        st.metric(
            "Accuracy",
            f"{accuracy:.2f}%"
        )

    else:

        st.metric(
            "Accuracy",
            "N/A"
        )


with metric_col2:

    if precision is not None:

        st.metric(
            "Precision",
            f"{precision:.2f}%"
        )

    else:

        st.metric(
            "Precision",
            "N/A"
        )


with metric_col3:

    if recall is not None:

        st.metric(
            "Recall",
            f"{recall:.2f}%"
        )

    else:

        st.metric(
            "Recall",
            "N/A"
        )


with metric_col4:

    if f1 is not None:

        st.metric(
            "F1 Score",
            f"{f1:.2f}%"
        )

    else:

        st.metric(
            "F1 Score",
            "N/A"
        )


st.info(
    "The ML metrics represent performance on the current "
    "stratified hold-out test split."
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("🧩 Confusion Matrix")

st.image(
    str(CONFUSION_GRAPH),
    width="stretch"
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.markdown("---")

st.subheader("📚 Dataset Information")

info_col1, info_col2, info_col3, info_col4 = st.columns(4)

with info_col1:

    st.metric(
        "Total Records",
        f"{len(df):,}"
    )

with info_col2:

    st.metric(
        "Temperature Sensors",
        "4"
    )

with info_col3:

    st.metric(
        "Accelerometers",
        "4"
    )

with info_col4:

    st.metric(
        "ML Sensor Features",
        "28"
    )


st.markdown(
    """
    <div class="summary-box">

    <b>Dataset:</b>
    UTK-ASL Digital Twin Predictive Maintenance Dataset

    <br><br>

    <b>Monitoring Variables:</b>
    Temperature and vibration sensor measurements

    <br><br>

    <b>Machine Conditions in Dataset:</b>
    Baseline, Imbalance, Looseness and Angular Misalignment

    <br><br>

    <b>Machine Learning Model:</b>
    Random Forest Classifier

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding:20px;">

    <b>Machine Health Monitoring System</b>

    <br>

    Temperature & Vibration Based Predictive Maintenance

    <br><br>

    EAI Lab Project | PCCOE

    </div>
    """,
    unsafe_allow_html=True
)
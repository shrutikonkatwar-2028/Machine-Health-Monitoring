# Machine-Health-Monitoring
Predictive maintenance dashboard using real temperature and vibration sensor data, threshold-based health monitoring, and Random Forest machine fault classification.
# ⚙️ Machine Health Monitoring System

### Temperature & Vibration Based Predictive Maintenance

A machine health monitoring project that analyzes real industrial sensor data, tracks temperature and vibration trends, identifies threshold violations, and classifies machine operating conditions using Machine Learning.

## 📌 Project Overview

Unexpected machine failures can cause downtime, equipment damage, and maintenance costs. This project demonstrates a data-driven approach to predictive maintenance by combining sensor threshold monitoring, visual analytics, and machine learning.

The system includes an interactive Streamlit dashboard that displays machine health status, sensor readings, warning and critical thresholds, maintenance recommendations, historical trends, and model evaluation results.

## 🎯 Objectives

- Monitor temperature using four thermocouples.
- Analyze vibration using four accelerometers and 24 vibration features.
- Clean and preprocess real-world sensor data.
- Visualize safe, warning, and critical monitoring conditions.
- Classify machine conditions using a Random Forest Classifier.
- Present sensor trends and model performance through an interactive dashboard.

## 📊 Dataset

**Source:** [UTK-ASL Digital Twin Predictive Maintenance Dataset](https://github.com/UTK-ASL/Dataset_digital_twin_predictive_maintenance)

**Dataset file used:** `Cutsforth_InsightCM/final.csv`

| Dataset detail | Value |
|---|---|
| Sensor records | 2,083 |
| Original columns | 33 |
| Temperature sensors | 4 thermocouples |
| Accelerometers | 4 |
| Temperature features | 4 |
| Vibration features | 24 |
| Machine conditions | 4 |

### Machine Conditions

- `baseline`
- `imbalance`
- `looseness`
- `angular_misalignment`

The dataset includes timestamps, temperature measurements, vibration features, machine status, and additional recorded variables.

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend and dashboard:** Streamlit
- **Data processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualizations:** Matplotlib
- **Model:** Random Forest Classifier

## 🔄 System Workflow

```text
Real Sensor Dataset
        ↓
Data Inspection and Preprocessing
        ↓
Missing Value Handling
        ↓
Temperature and Vibration Feature Selection
        ↓
Threshold-Based Health Monitoring
        ↓
Random Forest Classification
        ↓
Model Evaluation
        ↓
Interactive Streamlit Dashboard
```

## 🧹 Data Preprocessing

The preprocessing workflow includes:

1. Converting and processing timestamp information.
2. Identifying missing sensor readings.
3. Removing `Tachometer_Gap_volt` because it contains no usable values.
4. Handling missing numerical values using median imputation.
5. Selecting temperature and vibration features for machine-health classification.
6. Splitting the data using a stratified 80:20 train-test split.

## 🚦 Threshold-Based Monitoring

The dashboard compares average temperature and vibration RMS with dataset-derived monitoring thresholds.

| Measurement | Healthy baseline | Warning threshold | Critical threshold |
|---|---:|---:|---:|
| Temperature | 73.37 °F | 76.47 °F | 77.24 °F |
| Vibration RMS | 0.0897 g | 0.0937 g | 0.0947 g |

- **Safe:** The monitored reading remains below the warning threshold.
- **Warning:** The reading has crossed the warning threshold.
- **Critical:** The reading has crossed the critical threshold.

These thresholds are dataset-derived monitoring indicators, not manufacturer-defined safety limits. They should not be used as the sole basis for operational safety decisions.

## 🤖 Machine Learning Model

A Random Forest Classifier is used to predict the machine condition from the selected temperature and vibration features.

### Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 99.76% |
| Precision | 99.76% |
| Recall | 99.76% |
| F1-score | 99.76% |

The reported metrics correspond to the current stratified hold-out test split. They do not guarantee equivalent performance on unseen machines or real-world operating conditions. Because sensor records may be temporally related, additional time-based and independent-machine validation is needed.

## 📈 Dashboard Features

- Current machine-health status
- Selected-date temperature and vibration readings
- Sensor threshold analysis
- Machine health score and monitoring risk indicator
- Maintenance inspection recommendations
- Daily machine health trend
- Temperature and vibration monitoring graphs
- Daily machine health report
- Machine-learning evaluation metrics
- Confusion matrix
- Dataset and sensor information

The dashboard's health score is a threshold-based monitoring indicator, not the model's accuracy or a certified engineering health index.

## 🖥️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Machine-Health-Monitoring-System.git
cd Machine-Health-Monitoring-System
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment (recommended)

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

`http://localhost:8501`

### Dataset and output files

Make sure the dataset and any generated report or image files are in the locations expected by `app.py`. If your application generates output files automatically, you may not need to include all generated files in the repository.

## 📁 Project Structure

```text
Machine-Health-Monitoring-System/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── dataset/
├── output/
└── screenshots/
```

## 🔮 Future Scope

- Real-time IoT sensor integration
- Automated maintenance notifications
- Time-series anomaly detection
- Remaining Useful Life (RUL) prediction
- Monitoring across multiple machines
- Validation under different loads and environmental conditions
- Integration with industrial monitoring systems

## ⚠️ Limitations

- The analysis uses the available test-bed dataset.
- Some calendar dates have no recorded data; missing days are not treated as healthy conditions.
- Thresholds are specific to the analyzed dataset and require engineering validation before operational use.
- High hold-out test performance should be verified using chronological splits and independent machine data.

## 📚 References

1. [UTK-ASL — Dataset for Digital Twin Predictive Maintenance](https://github.com/UTK-ASL/Dataset_digital_twin_predictive_maintenance)
2. Williams et al., *Cross-domain digital twin architecture for predictive maintenance via machine learning and Large Language Models*, Computers & Industrial Engineering (2026). DOI: [10.1016/j.cie.2026.111914](https://doi.org/10.1016/j.cie.2026.111914)

## 👩‍💻 Project Information

**Project:** Machine Health Monitoring System  
**Subject:** EAI Lab  
**Department:** Computer Engineering (AI & ML)  
**Institute:** Pimpri Chinchwad College of Engineering (PCCOE)  
**Academic Year:** 2025–26

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.

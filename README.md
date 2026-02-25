# HiveDrive — IMU Event Analysis (Hackathon Round-2 Resource)

This repository contains the **datasets, analysis code, and outputs** used to validate the **sensor-layer feasibility** of HiveDrive using smartphone-grade IMU signals.

The goal of this repo is **not** to ship a full app, but to show that our sensing pipeline can:
1) identify **candidate road events** from IMU data (anomaly detection), and  
2) reduce false alerts using a simple **ML filter** (pothole detection).

---

## What’s inside

### 1) Datasets
**Pothole trips (labeled):**
- `Pothole/trip1_sensors.csv` … `trip5_sensors.csv`
- `Pothole/trip1_potholes.csv` … `trip5_potholes.csv`

Each `*_sensors.csv` includes IMU readings (accelerometer + gyroscope) and timestamps.  
Each `*_potholes.csv` contains labeled pothole timestamps for that trip.

**Road condition segments (unlabeled):**
- `RoadCondition/good1_sensors.csv` … `good10_sensors.csv`
- `RoadCondition/bad1_sensors.csv` … `bad5_sensors.csv`

These are used as additional background segments for negative examples / robustness.

> Note: Some folders may contain duplicated copies (e.g., `_extracted_dataset/`) created during local ZIP extraction.  
> For clean runs, prefer the root `Pothole/` and `RoadCondition/` folders.

---

### 2) Programs (Python)
- `raw.py`  
  **Step 1:** Plots raw IMU signals and performs a simple adaptive-threshold detector to find “candidate bumps”.

- `step_2.py`  
  **Step 2:** Extracts event windows, computes simple features, trains a **Random Forest** pothole classifier, and shows how classification reduces false alerts compared to anomaly detection alone.

---

### 3) Output Images (PPT-ready)
From **Step 1**:
- `outputs_hivedrive/trip1_01_raw_acc_axes.png`
- `outputs_hivedrive/trip1_02_detection_overview.png`
- `outputs_hivedrive/trip1_03_zoom.png`

From **Step 2**:
- `outputs_hivedrive_step2/trip1_05_baseline_vs_rf_events.png` (main result image)
- `outputs_hivedrive_step2/rf_confusion_matrix.png`
- `outputs_hivedrive_step2/rf_feature_importances.png`
- `outputs_hivedrive_step2/trip1_04_window_examples.png`

---

## How to run (Windows)

### 1) Create virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate

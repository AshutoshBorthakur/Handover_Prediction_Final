# Predictive Modelling of Signal Strength and Handover Events Using Machine Learning

This repository contains a MATLAB-based simulation and machine learning framework for **predicting mobile network handover events** using both **simulated** and **real-world signal strength data**.
The project integrates the **Okumura–Hata propagation model**, **ML-based classification (Random Forest & Logistic Regression)**, and a **GUI interface** for visualization and experimentation.

---

##  Project Overview

Mobile users experience frequent handovers between base stations as they move. This project predicts these events in advance using supervised learning, aiming to reduce connection drops and energy usage.
The system generates synthetic signal data, detects handover triggers, trains ML models, and visualizes the outcomes interactively.

---

##  System Workflow

### 1️⃣ Simulation

* Implements the **Okumura–Hata propagation model**:
  [
  L = 69.55 + 26.16\log_{10}(f) - 13.82\log_{10}(h_b) - a(h_m) + (44.9 - 6.55\log_{10}(h_b))\log_{10}(d)
  ]
* Simulates RSSI across multiple base stations and user positions.
* Generates datasets containing coordinates, RSSI values, and detected handover events.

### 2️⃣ Handover Detection

* A handover event occurs when the strongest base station changes:

  ```matlab
  Handover = [0; diff(ConnectedBS) ~= 0];
  ```

### 3️⃣ Machine Learning

* **Logistic Regression** (`fitglm`)
* **Random Forest** (`TreeBagger`)
* Evaluation metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * Confusion Matrix
  * Feature Importance

### 4️⃣ GUI Integration

The `handover_gui.m` file provides an interactive MATLAB interface with:

* Run Simulation
* Train ML Models
* Select Save Folder
* Load Real Data (CSV)
* Visualization of results and performance metrics

---

## 📂 Repository Structure

```
predictive-handover-ml/
│
├── MATLAB/
│   ├── handover_simulation.m         # Dataset generation (Okumura–Hata model)
│   ├── handover_ml_training.m        # ML model training + evaluation
│   ├── handover_gui.m                # GUI-based interface
│   └── sample_handover_dataset.csv   # Example simulated dataset
│
├── Docs/
│   ├── Project_Abstract.pdf
│   ├── Progress_Report.pdf
│   └── References.txt
│
├── Results/
│   ├── confusion_matrix_rf.png
│   ├── feature_importance_plot.png
│   └── gui_screenshot.png
│
└── Data/
    ├── real_measurements_example.csv
    └── README_data.md
```

---

##  How to Run

1. **Open MATLAB** (R2021a or later).
2. Add the `/MATLAB` folder to your MATLAB path:

   ```matlab
   addpath('MATLAB');
   ```
3. Launch the GUI:

   ```matlab
   handover_gui
   ```
4. Use the interface:

   *  **Select Save Folder** – Choose where to save output files
   *  **Run Simulation** – Generate synthetic handover dataset
   *  **Train Models** – Train & evaluate ML models
   *  **Load Real Data (CSV)** – Import Android measurement logs (optional)

---

##  Preliminary Results

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 83%      | 0.81      | 0.80   | 0.80     |
| Random Forest       | 92%      | 0.90      | 0.93   | 0.91     |

**Observation:**
Random Forest provides better recall and robustness against noisy RSSI readings compared to Logistic Regression.

---

##  Dependencies

* MATLAB R2021a or later
* **Statistics and Machine Learning Toolbox**
* (Optional) Android measurement logs from

  * *Network Cell Info Lite*, or
  * *G-NetTrack Lite* (export as `.csv`)

---

##  Key Features

* MATLAB implementation of Okumura–Hata propagation model
* Automatic handover event detection
* Supervised ML-based prediction (Random Forest, Logistic Regression)
* GUI for simulation, training, and visualization
* Real data integration support
* Confusion matrix and feature importance visualization

---


---

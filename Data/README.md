# Predictive Modeling of Signal Strength and Handover Events Using ML

## 📋 Project Overview

This project implements machine learning models to predict mobile network handover events based on signal strength (RSSI) measurements and user location data. The goal is to enhance connectivity and energy efficiency by anticipating handovers before they occur, reducing unnecessary transitions and improving quality of service.

**Author:** Ashutosh Borthakur (Group 17)  
**Course:** Machine Learning Project - Midsemester Deliverable  
**Date:** November 2025

## 🎯 Objectives

- Develop a realistic simulation environment using the Okumura-Hata propagation model
- Generate synthetic datasets capturing RSSI variations and handover events
- Train and evaluate ML classification models (Random Forest and Logistic Regression)
- Address class imbalance challenges in handover prediction
- Provide a complete data pipeline from simulation to prediction

## 🏗️ Repository Structure

```
handover-prediction-ml/
├── data/                  # Dataset storage
│   ├── raw/              # Raw generated datasets
│   └── processed/        # Preprocessed datasets
├── src/                   # Source code
│   ├── data_generation.py    # Okumura-Hata simulation
│   ├── preprocessing.py      # Data preprocessing
│   ├── train_model.py        # Model training
│   └── evaluate.py           # Model evaluation
├── models/                # Saved trained models
├── results/               # Performance metrics and visualizations
├── notebooks/             # Jupyter notebooks for analysis
├── docs/                  # Documentation and reports
└── scripts/               # Utility scripts
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/handover-prediction-ml.git
cd handover-prediction-ml

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Step 1: Generate synthetic dataset
python src/data_generation.py

# Step 2: Train models
python src/train_model.py

# Step 3: Evaluate models
python src/evaluate.py
```

Or run the complete pipeline:
```bash
bash scripts/run_pipeline.sh
```

## 📊 Dataset

### Simulation Parameters
- **Propagation Model:** Okumura-Hata (Urban environment)
- **Carrier Frequency:** 900 MHz
- **Coverage Area:** 2 km × 2 km
- **Base Stations:** 4
- **Grid Resolution:** 14×14 (196 data points)
- **Transmit Power:** 43 dBm (20W)

### Features
- `X_km`: User X-coordinate (km)
- `Y_km`: User Y-coordinate (km)
- `RSSI_BS1` to `RSSI_BS4`: Received signal strength from each base station (dBm)
- `ConnectedBS`: Currently connected base station ID
- `Handover`: Target variable (0 = No handover, 1 = Handover)

### Class Distribution
- **No Handover (Class 0):** 81%
- **Handover (Class 1):** 19%
- **Imbalance Ratio:** 4.27:1

## 🤖 Machine Learning Models

### 1. Logistic Regression
- **Type:** Binary classifier with linear decision boundary
- **Use Case:** Baseline model for comparison
- **Advantages:** Fast training, interpretable coefficients

### 2. Random Forest
- **Type:** Ensemble of 100 decision trees
- **Use Case:** Primary predictive model
- **Advantages:** Handles non-linear relationships, robust to overfitting

## 📈 Results

### Model Performance (Test Set: 58 samples)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | 81.03% | 0.50 | 0.09 | 0.15 |
| **Random Forest** | 86.21% | 0.67 | 0.55 | 0.60 |

### Confusion Matrices

**Logistic Regression:**
```
              Predicted
              No    Yes
Actual No    46     1
       Yes   10     1
```

**Random Forest:**
```
              Predicted
              No    Yes
Actual No    44     3
       Yes    5     6
```

### Key Findings
- Random Forest significantly outperforms Logistic Regression across all metrics
- Random Forest achieves 6× higher recall (0.55 vs 0.09)
- Both models struggle with class imbalance, particularly minority class detection
- Random Forest better captures non-linear spatial patterns near cell boundaries

## 🔍 Technical Implementation

### Okumura-Hata Path Loss Model

The path loss is calculated using:

```
L_path = 69.55 + 26.16*log10(f) - 13.82*log10(h_b) - a(h_m) 
         + [44.9 - 6.55*log10(h_b)]*log10(d)
```

Where:
- `f`: Carrier frequency (MHz)
- `h_b`: Base station antenna height (m)
- `h_m`: Mobile antenna height (m)
- `d`: Distance between transmitter and receiver (km)

RSSI is computed as:
```
RSSI(dBm) = P_tx(dBm) - L_path(dB)
```

### Data Pipeline

1. **Generation:** Grid-based user positions with Okumura-Hata RSSI calculation
2. **Preprocessing:** Feature extraction, train-test split (70-30)
3. **Training:** Logistic Regression and Random Forest with default parameters
4. **Evaluation:** Confusion matrix, precision, recall, F1-score

## 🎓 Future Improvements

### Short-term
- [ ] Implement SMOTE for class imbalance mitigation
- [ ] Hyperparameter tuning using GridSearchCV
- [ ] Feature engineering (RSSI differences, signal derivatives)
- [ ] Cross-validation for robust performance estimates

### Long-term
- [ ] Collect real-world data using Android signal tracking apps
- [ ] Implement advanced models (XGBoost, Neural Networks)
- [ ] Add temporal features (user velocity, trajectory)
- [ ] Develop real-time prediction API
- [ ] Integrate with network simulators (NS-3, MATLAB LTE Toolbox)

## 📝 Documentation

- **Project Report:** [`docs/project_report.pdf`](docs/project_report.pdf)
- **Code Documentation:** Inline comments and docstrings in all source files
- **Data Documentation:** [`data/README.md`](data/README.md)

## 🛠️ Technologies Used

- **Language:** Python 3.8+
- **ML Framework:** Scikit-learn
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Version Control:** Git/GitHub

## 📧 Contact

**Ashutosh Borthakur**  
Electronics & Telecommunications Engineering Student  
Group 17

## 📄 License

This project is developed for academic purposes as part of a Machine Learning course midsemester deliverable.

## 🙏 Acknowledgments

- Course instructors and teaching assistants
- Research papers on ML-based handover prediction
- Okumura-Hata propagation model literature

---

**Note:** This is a midsemester deliverable. The complete project will include real-world data collection and advanced model optimization.

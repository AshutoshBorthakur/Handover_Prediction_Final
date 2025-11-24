import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV


# Load MATLAB-simulated dataset
sim_df = pd.read_csv(r'C:\Users\borth\Desktop\predictive-handover-ml\MATLAB\handover_dataset.csv')

sim_df["ConnectedBS"] = sim_df["ConnectedBS"].astype(int)
sim_df["ConnectedBS"] = sim_df["ConnectedBS"].astype(int)

# 1. Convert X_km, Y_km to pseudo GPS coords
lat0 = 15.3920       # real dataset starting latitude
lon0 = 73.8810       # real dataset starting longitude

# Convert km → degrees approx. (1 km ≈ 0.009 degrees)
sim_df["Latitude"]  = lat0  + sim_df["Y_km"] * 0.009
sim_df["Longitude"] = lon0  + sim_df["X_km"] * 0.009

# 2. Combine RSSI from the connected BS
bs_columns = ["RSSI_BS1", "RSSI_BS2", "RSSI_BS3", "RSSI_BS4"]

def pick_rssi(row):
    bs_index = int(row["ConnectedBS"]) - 1
    return row[bs_columns].iloc[bs_index]



sim_df["RSSI"] = sim_df.apply(pick_rssi, axis=1)

# 3. Simulate Speed (km movement per step)
sim_df["Speed"] = np.random.uniform(1, 5, size=len(sim_df))

# 4. Simulate Altitude (fake but harmless)
sim_df["Altitude"] = np.random.uniform(-25, 10, size=len(sim_df))

# 5. Keep only real-like columns
sim_ready = sim_df[["Longitude", "Latitude", "RSSI", "Speed", "Altitude", "Handover"]]

print(sim_ready.head())
print(sim_ready.shape)

real_df = pd.read_csv(r'C:\Users\borth\Documents\Machine Learning\Machine Learning\Jio_True5G_2025.11.03_21.58.16.txt', sep="\t")

# Rename Level → RSSI
real_df = real_df.rename(columns={"Level": "RSSI"})

# Compute RSSI changes
real_df["RSSI_diff"] = real_df["RSSI"].diff().abs().fillna(0)
real_df["RSSI_grad"] = real_df["RSSI"].diff().fillna(0)

# Create heuristic handover label
real_df["Handover"] = (
    (real_df["RSSI_diff"] > 5) |
    (real_df["RSSI_grad"] < -4) |
    ((real_df["RSSI_diff"] > 3) & (real_df["Speed"] > 2))
).astype(int)

# Keep only matching columns
real_ready = real_df[["Longitude", "Latitude", "RSSI", "Speed", "Altitude", "Handover"]]
combined = pd.concat([sim_ready, real_ready], ignore_index=True)
# Save merged dataset for inspection
output_path = "combined_dataset_debug.csv"
combined.to_csv(output_path, index=False)

print("\nSaved merged dataset to:", output_path)


print("Combined shape:", combined.shape)
print(combined.head())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X = combined[["Longitude", "Latitude", "RSSI", "Speed", "Altitude"]]
y = combined["Handover"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#rf = RandomForestClassifier(n_estimators=200)
#rf.fit(X_train, y_train)

#y_pred = rf.predict(X_test)

#print(classification_report(y_test, y_pred))

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False
)

param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'gamma': [0, 1, 5, 10],
}

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,                  # number of combinations to try
    scoring='f1',               # because handover class is imbalanced
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print("Training tuned XGBoost... please wait.")
random_search.fit(X_train, y_train)

best_xgb = random_search.best_estimator_
print("\nBest XGBoost Parameters:", random_search.best_params_)

y_pred = best_xgb.predict(X_test)

print("\n--- Tuned XGBoost Performance ---")
print(classification_report(y_test, y_pred))

import joblib
joblib.dump(best_xgb, "handover_model.pkl")
print("Model saved as handover_model.pkl")

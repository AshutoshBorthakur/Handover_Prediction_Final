import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------
# 1. LOAD REAL G-NETTRACK DATA
# ------------------------------------------------------

file = "Jio_True5G_2025.11.03_21.58.16.txt"   # change filename

# Auto-detect tab or comma delimiter
try:
    df = pd.read_csv(file, sep="\t")
except:
    df = pd.read_csv(file)

print("Dataset loaded:")
print(df.head())

# ------------------------------------------------------
# 2. Clean & preprocess
# ------------------------------------------------------

# Convert numeric fields
df["Level"] = pd.to_numeric(df["Level"], errors="coerce")     # RSSI
df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")
df["Altitude"] = pd.to_numeric(df["Altitude"], errors="coerce")

df = df.dropna()

# ------------------------------------------------------
# 3. Detect handover-like events
# ------------------------------------------------------
# G-NetTrack Lite DOES NOT give Cell ID → infer handover from RSSI jumps
# A ΔRSSI > 10 dB is considered a potential handover trigger

# --- Handover detection (Heuristic) ---
# MUST create these before using:
df["RSSI_diff"] = df["Level"].diff().abs().fillna(0)
df["RSSI_grad"] = df["Level"].diff().fillna(0)

df["Handover"] = (
    (df["RSSI_diff"] > 5) |
    (df["RSSI_grad"] < -4) |
    ((df["RSSI_diff"] > 3) & (df["Speed"] > 2))
).astype(int)

print("Detected handover-like events:", df["Handover"].sum())


# ------------------------------------------------------
# 4. Features and Labels
# ------------------------------------------------------

X = df[["Longitude", "Latitude", "Level", "Speed", "Altitude"]]
y = df["Handover"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------------------------------------------
# 5. Logistic Regression
# ------------------------------------------------------

log_model = LogisticRegression(max_iter=200, class_weight="balanced")
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\n--- Logistic Regression Results ---")
print(classification_report(y_test, y_pred_log))

# ------------------------------------------------------
# 6. Random Forest
# ------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest Results ---")
print(classification_report(y_test, y_pred_rf))

# ------------------------------------------------------
# 7. Confusion matrices
# ------------------------------------------------------

fig, ax = plt.subplots(1, 2, figsize=(12,5))

sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, cmap='Blues', ax=ax[0])
ax[0].set_title("Logistic Regression Confusion Matrix")
ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("True")

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, cmap='Greens', ax=ax[1])
ax[1].set_title("Random Forest Confusion Matrix")
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("True")

plt.show()

# ------------------------------------------------------
# 8. Feature Importance
# ------------------------------------------------------

plt.figure(figsize=(8,5))
sns.barplot(x=rf_model.feature_importances_,
            y=X.columns)
plt.title("Feature Importance (Random Forest)")
plt.show()

# ------------------------------------------------------
# 9. Plot signal strength over time
# ------------------------------------------------------

plt.figure(figsize=(12,5))
plt.plot(df["Level"], marker=".")
plt.title("RSSI / RSRP Over Time (Real Jio 5G)")
plt.xlabel("Sample Index")
plt.ylabel("Signal Level (dBm)")
plt.grid()
plt.show()

# ------------------------------------------------------
# 10. Plot detected handover-like events
# ------------------------------------------------------

plt.figure(figsize=(12,5))
plt.plot(df["Level"], label="RSSI")
plt.scatter(df.index[df["Handover"]==1], df["Level"][df["Handover"]==1],
            color="red", label="Handover-like Event")
plt.legend()
plt.title("Detected RSSI Jumps (Potential Handovers)")
plt.show()

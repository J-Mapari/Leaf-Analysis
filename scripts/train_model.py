import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



# -------------------------
# LOAD TRAIN ONLY
# -------------------------
train = pd.read_csv("outputs/train.csv")

# -------------------------
# CLEAN
# -------------------------
train = train.dropna(subset=["class_id"])
train = train.fillna(0)

# -------------------------
# SPLIT FEATURES / LABELS
# -------------------------
X = train.drop(columns=["image", "class_id", "type"], errors="ignore")
y = train["class_id"]

# -------------------------
# TRAIN / VALIDATION SPLIT
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# TRAIN MODEL
# -------------------------
model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
model.fit(X_train, y_train)

# -------------------------
# VALIDATION PERFORMANCE
# -------------------------
preds = model.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, preds))

import joblib
joblib.dump(model, "model.pkl")
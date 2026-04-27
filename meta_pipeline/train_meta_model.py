import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("outputs_meta/train_meta.csv")

# -------------------------
# CLEAN DATA
# -------------------------
df = df.dropna(subset=["class_id"])
df = df.fillna(0)

# -------------------------
# REMOVE RARE CLASSES (<2 samples)
# -------------------------
counts = df["class_id"].value_counts()

rare_classes = counts[counts < 2].index
print("Removing rare classes:", len(rare_classes))

df = df[~df["class_id"].isin(rare_classes)]

# -------------------------
# FEATURES / LABELS
# -------------------------
X = df.drop(columns=["image", "class_id"], errors="ignore")
y = df["class_id"]

print("Final samples:", len(df))
print("Final classes:", y.nunique())

# -------------------------
# TRAIN / VALIDATION SPLIT
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # now safe
)

# -------------------------
# TRAIN MODEL
# -------------------------
model = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# EVALUATION
# -------------------------
preds = model.predict(X_val)

acc = accuracy_score(y_val, preds)
print("\nMeta Model Accuracy:", acc)

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nTop features:")
print(importances.head(10))

# -------------------------
# SAVE MODEL
# -------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/meta_model.pkl")

print("\nModel saved to models/meta_model.pkl")
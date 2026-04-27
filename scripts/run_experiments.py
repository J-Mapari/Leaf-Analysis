import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("outputs/train.csv")

# clean
df = df.dropna(subset=["class_id"])
df = df.fillna(0)

# -------------------------
# SPLIT FEATURES / LABELS
# -------------------------
X = df.drop(columns=["image", "class_id", "type"], errors="ignore")
y = df["class_id"]

# -------------------------
# TRAIN / VALIDATION SPLIT
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# FEATURE GROUPS
# -------------------------
groups = {
    "shape": ["area", "perimeter", "compactness"],
    "colour": ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"],
    "texture": [c for c in X.columns if "lbp" in c or c in ["contrast","homogeneity","energy","correlation"]],
    "venation": ["vein_density"]
}

# -------------------------
# EXPERIMENTS
# -------------------------
experiments = {
    "full": X.columns.tolist(),

    "no_shape": [c for c in X.columns if c not in groups["shape"]],
    "no_colour": [c for c in X.columns if c not in groups["colour"]],
    "no_texture": [c for c in X.columns if c not in groups["texture"]],
    "no_venation": [c for c in X.columns if c not in groups["venation"]],

    "shape_only": groups["shape"],
    "texture_only": groups["texture"],
    "colour_only": groups["colour"],
    "venation_only": groups["venation"]
}

# -------------------------
# OUTPUT DIR
# -------------------------
BASE_OUT = "outputs/experiments"
os.makedirs(BASE_OUT, exist_ok=True)

results = []

# -------------------------
# RUN ALL EXPERIMENTS
# -------------------------
for name, cols in experiments.items():

    print(f"\nRunning: {name}")

    out_dir = os.path.join(BASE_OUT, name)
    os.makedirs(out_dir, exist_ok=True)

    # select features
    X_tr = X_train[cols]
    X_va = X_val[cols]

    # train model
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    model.fit(X_tr, y_train)

    preds = model.predict(X_va)
    acc = accuracy_score(y_val, preds)

    print(f"Accuracy: {acc:.4f}")

    # -------------------------
    # SAVE METRICS
    # -------------------------
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")

    # -------------------------
    # SAVE FEATURE IMPORTANCE
    # -------------------------
    importances = pd.DataFrame({
        "feature": cols,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    importances.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    # -------------------------
    # CONFUSION MATRIX
    # -------------------------
    cm = confusion_matrix(y_val, preds)

    disp = ConfusionMatrixDisplay(cm)
    disp.plot(xticks_rotation=90)

    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    # store results
    results.append((name, acc))

# -------------------------
# SAVE SUMMARY TABLE
# -------------------------
results_df = pd.DataFrame(results, columns=["experiment", "accuracy"])
results_df = results_df.sort_values(by="accuracy", ascending=False)

results_df.to_csv("outputs/experiments_summary.csv", index=False)

print("\nAll experiments complete!")
print(results_df)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("outputs/train.csv")

df = df.dropna(subset=["class_id"])
df = df.fillna(0)

# -------------------------
# SPLIT FEATURES / LABELS
# -------------------------
X = df.drop(columns=["image", "class_id", "type"], errors="ignore")
y = df["class_id"]

# -------------------------
# TRAIN / VAL SPLIT
# -------------------------
X_train, X_val, y_train, y_val, type_train, type_val = train_test_split(
    X, y, df["type"],
    test_size=0.2,
    random_state=42,
    stratify=df["type"]   # ← THIS IS THE FIX
)
# -------------------------
# DEFINE FEATURE GROUPS
# -------------------------
groups = {
    "shape": ["area", "perimeter", "compactness"],
    "colour": ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"],
    "texture": [c for c in X.columns if "lbp" in c or c in ["contrast","homogeneity","energy","correlation"]],
    "venation": ["vein_density"]
}

experiments = {
    "full": X.columns.tolist(),
    "shape_only": groups["shape"],
    "texture_only": groups["texture"],
    "colour_only": groups["colour"]
}

# -------------------------
# RUN
# -------------------------
results = []

for name, cols in experiments.items():

    print(f"\n=== {name} ===")

    model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    model.fit(X_train[cols], y_train)

    preds = model.predict(X_val[cols])

    for t in ["Scan", "pseudoscan", "photograph"]:
        mask = (type_val == t)

        if mask.sum() == 0:
            continue

        acc = accuracy_score(y_val[mask], preds[mask])

        print(f"{t}: {acc:.4f}")

        results.append({
            "experiment": name,
            "type": t,
            "accuracy": acc
        })

# -------------------------
# SAVE
# -------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/per_type_results.csv", index=False)

print("\nSaved: outputs/per_type_results.csv")
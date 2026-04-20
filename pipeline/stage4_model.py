import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


# -------------------------------
# Manual per-class split (12/4)
# -------------------------------
def stratified_manual_split(df, train_per_class=12):

    train_rows = []
    test_rows = []

    for label in df["label"].unique():

        class_df = df[df["label"] == label]

        # shuffle
        class_df = class_df.sample(frac=1, random_state=42)

        train = class_df.iloc[:train_per_class]
        test = class_df.iloc[train_per_class:]

        train_rows.append(train)
        test_rows.append(test)

    train_df = pd.concat(train_rows)
    test_df = pd.concat(test_rows)

    return train_df, test_df


# -------------------------------
# Main training function
# -------------------------------
def train_model(csv_path):

    os.makedirs("data/plots", exist_ok=True)

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        print("No labels found. Skipping ML.")
        return

    # -------------------------------
    # Split
    # -------------------------------
    train_df, test_df = stratified_manual_split(df)

    X_train = train_df.drop(columns=["label", "filename"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label", "filename"])
    y_test = test_df["label"]

    # -------------------------------
    # Scale features (important)
    # -------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------
    # Model
    # -------------------------------
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42
    )

    # -------------------------------
    # Cross-validation (on train set)
    # -------------------------------
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\nCV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # -------------------------------
    # Train
    # -------------------------------
    model.fit(X_train, y_train)

    # -------------------------------
    # Test
    # -------------------------------
    preds = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds))

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig("data/plots/confusion_matrix.png")
    plt.close()

    # -------------------------------
    # Feature importance
    # -------------------------------
    importances = model.feature_importances_
    feature_names = train_df.drop(columns=["label", "filename"]).columns

    idx = np.argsort(importances)[-15:]

    plt.figure(figsize=(8,6))
    plt.barh(range(len(idx)), importances[idx])
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.title("Top Feature Importance")
    plt.savefig("data/plots/feature_importance.png")
    plt.close()

    # -------------------------------
    # Save model + scaler
    # -------------------------------
    joblib.dump(model, "data/model.pkl")
    joblib.dump(scaler, "data/scaler.pkl")

    print("\nModel saved to data/model.pkl")
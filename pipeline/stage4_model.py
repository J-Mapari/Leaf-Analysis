import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def train_model(csv_path):

    # -------------------------------
    # Load dataset (THIS MUST BE features.csv)
    # -------------------------------
    df = pd.read_csv(csv_path)

    if "species" not in df.columns:
        raise ValueError("CSV must contain 'species' column")

    # -------------------------------
    # Features / Labels
    # -------------------------------
    X = df.drop(columns=["species", "filename"])
    y = df["species"]

    # -------------------------------
    # Train/Test split (STRATIFIED)
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

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
    # Cross-validation (robust metric)
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
    os.makedirs("data/plots", exist_ok=True)

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(xticks_rotation=45)

    scores = cross_val_score(model, X, y, cv=5)

    print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    plt.title("Confusion Matrix")
    plt.savefig("data/plots/confusion_matrix.png")
    plt.close()

    # -------------------------------
    # Feature Importance
    # -------------------------------
    importances = model.feature_importances_
    feature_names = X.columns

    idx = np.argsort(importances)[-15:]

    plt.figure(figsize=(8, 6))
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


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    train_model(r"C:\Users\jvm33\Leaf-Analysis\data\features.csv")
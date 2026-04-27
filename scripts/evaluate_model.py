import pandas as pd
from sklearn.metrics import accuracy_score
import joblib 

model = joblib.load("model.pkl")

test = pd.read_csv("outputs/test.csv")

X = test.drop(columns=["image", "class_id", "type"])
y = test["class_id"]

preds = model.predict(X)

for t in ["Scan", "Scan-like", "Photo"]:
    subset = test[test["type"]==t]
    if len(subset)==0:
        continue

    X_sub = subset.drop(columns=["image", "class_id", "type"])
    y_sub = subset["class_id"]

    acc = accuracy_score(y_sub, model.predict(X_sub))
    print(t, acc)

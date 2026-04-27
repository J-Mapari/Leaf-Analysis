import pandas as pd
df = pd.read_csv("outputs/train.csv")
print(df["type"].value_counts())
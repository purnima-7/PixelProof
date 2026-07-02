import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("../features.csv")

X = df.drop(columns=["image", "label"])

encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])

selector = SelectFromModel(
    RandomForestClassifier(
        n_estimators=700,
        random_state=42,
        n_jobs=-1
    ),
    threshold="0.75*mean"
)

selector.fit(X, y)

selected = X.columns[selector.get_support()]

print("\nSelected Features:\n")

for feature in selected:
    print(feature)

X_selected = X[selected]

new_df = pd.concat(
    [
        df["image"],
        X_selected,
        df["label"]
    ],
    axis=1
)

new_df.to_csv("../features_selected.csv", index=False)

print("\nSaved features_selected.csv")
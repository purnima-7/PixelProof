import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("../features.csv")

X = df.drop(columns=["image", "label"])

encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# -----------------------------
# Models
# -----------------------------

models = {

    "Random Forest":
    RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced"
    ),

    "SVM":
    Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            probability=True,
            C=5,
            gamma="scale",
            random_state=42
        ))
    ]),

    "XGBoost":
    XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

# -----------------------------
# Evaluate
# -----------------------------

results = {}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    results[name] = scores

    print("="*60)
    print(name)
    print("="*60)

    print("Fold Accuracy:")
    print(np.round(scores,4))

    print()

    print("Mean Accuracy : {:.4f}".format(scores.mean()))
    print("Std Dev       : {:.4f}".format(scores.std()))

    print()

# -----------------------------
# Train Best Model
# -----------------------------

best_name = max(results, key=lambda k: results[k].mean())

print("="*60)
print("Best Model :", best_name)
print("="*60)

best_model = models[best_name]

best_model.fit(X,y)

joblib.dump(best_model,"../models/best_model.pkl")

print("\nSaved model to models/best_model.pkl")
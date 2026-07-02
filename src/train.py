import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

df = pd.read_csv("../features.csv")

X = df.drop(columns=["image", "label"])
y = LabelEncoder().fit_transform(df["label"])

# -------------------------------------------------
# Cross Validation
# -------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# -------------------------------------------------
# Models
# -------------------------------------------------

models = {

    "Random Forest":
    Pipeline([
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]),

    "SVM":
    Pipeline([
        (
            "selector",
            SelectFromModel(
                RandomForestClassifier(
                    n_estimators=700,
                    random_state=42,
                    n_jobs=-1
                ),
                threshold="0.75*mean"
            )
        ),
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=5,
                gamma="scale",
                probability=True,
                random_state=42
            )
        )
    ]),

    "XGBoost":
    Pipeline([
        (
            "selector",
            SelectFromModel(
                RandomForestClassifier(
                    n_estimators=700,
                    random_state=42,
                    n_jobs=-1
                ),
                threshold="0.75*mean"
            )
        ),
        (
            "xgb",
            XGBClassifier(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42
            )
        )
    ])
}

# -------------------------------------------------
# Metrics
# -------------------------------------------------

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

results = {}

# -------------------------------------------------
# Evaluation
# -------------------------------------------------

for name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results[name] = np.mean(scores["test_accuracy"])

    print("=" * 65)
    print(name)
    print("=" * 65)

    print(f"Accuracy : {scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}")
    print(f"Precision: {scores['test_precision'].mean():.4f}")
    print(f"Recall   : {scores['test_recall'].mean():.4f}")
    print(f"F1 Score : {scores['test_f1'].mean():.4f}")
    print(f"ROC AUC  : {scores['test_roc_auc'].mean():.4f}")

    print()

# -------------------------------------------------
# Best Model
# -------------------------------------------------

best_name = "Random Forest"

print("=" * 65)
print("Best Model :", best_name)
print("=" * 65)

best_model = models[best_name]

best_model.fit(X, y)

joblib.dump(best_model, "../models/best_model.pkl")

print("\nModel saved to models/best_model.pkl")
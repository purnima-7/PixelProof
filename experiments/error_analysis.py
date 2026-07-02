import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------
# Load dataset
# ----------------------------

df = pd.read_csv("../features.csv")

image_names = df["image"]

X = df.drop(columns=["image", "label"])

encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])

# ----------------------------
# Model
# ----------------------------

model = Pipeline([
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
            gamma="scale"
        )
    )
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ----------------------------
# Predictions
# ----------------------------

pred = cross_val_predict(
    model,
    X,
    y,
    cv=cv
)

print(confusion_matrix(y, pred))
print(classification_report(y, pred))

# ----------------------------
# Misclassified Images
# ----------------------------

errors = df[y != pred].copy()

errors["True Label"] = encoder.inverse_transform(y[y != pred])
errors["Predicted"] = encoder.inverse_transform(pred[y != pred])

errors.to_csv("../output/misclassified.csv", index=False)

print(f"\nMisclassified images: {len(errors)}")
print(errors[["image", "True Label", "Predicted"]].head())
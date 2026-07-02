import pandas as pd
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("../features.csv")

X = df.drop(columns=["image","label"])

model = joblib.load("../models/best_model.pkl")

# Only Random Forest and XGBoost have feature_importances_
if hasattr(model, "feature_importances_"):
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(importance.head(20))

    plt.figure(figsize=(10,7))

    plt.barh(
        importance["Feature"][:20],
        importance["Importance"][:20]
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig("../output/feature_importance.png")

    plt.show()

else:
    print("Selected model does not support feature importance.")
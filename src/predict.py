import sys
import joblib
import pandas as pd

from feature_extraction import extract_features_from_image

model = joblib.load("../models/best_model.pkl")
image_path = sys.argv[1]

features = extract_features_from_image(image_path)

features.pop("image", None)
features.pop("label", None)

X = pd.DataFrame([features])

# Probability that the image is a screen
score = model.predict_proba(X)[0][1]
print(f"{score:.2f}")
import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "../dataset"

# LBP Parameters
RADIUS = 3
N_POINTS = 8 * RADIUS


# -----------------------------
# Feature Functions
# -----------------------------

def laplacian_variance(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size


def brightness(gray):
    return np.mean(gray)


def contrast(gray):
    return np.std(gray)


def entropy(gray):
    return shannon_entropy(gray)


def glare_percentage(gray):
    return np.sum(gray > 240) / gray.size


def fft_features(gray):

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    radius = min(h, w) * 0.15

    high_freq = magnitude[distance > radius]
    low_freq = magnitude[distance <= radius]

    return np.mean(high_freq), np.mean(low_freq)


def lbp_features(gray):

    lbp = local_binary_pattern(
        gray,
        N_POINTS,
        RADIUS,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, N_POINTS + 3),
        density=True
    )

    return hist


# -----------------------------
# Main
# -----------------------------

rows = []

for label in ["real", "screen"]:

    folder = os.path.join(DATASET_PATH, label)

    for file in tqdm(os.listdir(folder), desc=label):

        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hf, lf = fft_features(gray)

        features = {
            "image": file,
            "label": label,
            "laplacian_variance": laplacian_variance(gray),
            "edge_density": edge_density(gray),
            "brightness": brightness(gray),
            "contrast": contrast(gray),
            "entropy": entropy(gray),
            "glare_percentage": glare_percentage(gray),
            "fft_high": hf,
            "fft_low": lf,
        }

        lbp_hist = lbp_features(gray)

        for i, value in enumerate(lbp_hist):
            features[f"lbp_{i}"] = value

        rows.append(features)


df = pd.DataFrame(rows)

df.to_csv("../features.csv", index=False)

print()
print("=" * 50)
print(df.head())
print("=" * 50)
print()
print("Saved as features.csv")
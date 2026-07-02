import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.feature import (
    local_binary_pattern,
    graycomatrix,
    graycoprops
)
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

    fft_high = np.mean(high_freq)
    fft_low = np.mean(low_freq)

    peak_strength = np.max(magnitude) / (np.mean(magnitude) + 1e-8)

    return fft_high, fft_low, peak_strength

def lbp_features(gray):

    hist_all = []

    for radius in [1, 3]:

        n_points = 8 * radius

        lbp = local_binary_pattern(
            gray,
            n_points,
            radius,
            method="uniform"
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, n_points + 3),
            density=True
        )

        hist_all.extend(hist)

    return np.array(hist_all)

def glcm_features(gray):

    gray = (gray / 8).astype(np.uint8)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=32,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]

    return contrast, homogeneity, energy, correlation

def rgb_features(img):

    b, g, r = cv2.split(img)

    return (
        np.mean(r),
        np.mean(g),
        np.mean(b),

        np.std(r),
        np.std(g),
        np.std(b)
    )

def hsv_features(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    return (
        np.mean(h),
        np.mean(s),
        np.mean(v),

        np.std(h),
        np.std(s),
        np.std(v)
    )


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
        gray = cv2.resize(gray, (256, 256))
        img = cv2.resize(img,(256,256))

        hf, lf, peak = fft_features(gray)

        r_mean,\
        g_mean,\
        b_mean,\
        r_std,\
        g_std,\
        b_std = rgb_features(img)

        h_mean,\
        s_mean,\
        v_mean,\
        h_std,\
        s_std,\
        v_std = hsv_features(img)

        glcm_contrast,\
        glcm_homogeneity,\
        glcm_energy,\
        glcm_correlation = glcm_features(gray)

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
            "fft_peak_strength": peak,

            "glcm_contrast": glcm_contrast,
            "glcm_homogeneity": glcm_homogeneity,
            "glcm_energy": glcm_energy,
            "glcm_correlation": glcm_correlation,

            "r_mean": r_mean,
            "g_mean": g_mean,
            "b_mean": b_mean,

            "r_std": r_std,
            "g_std": g_std,
            "b_std": b_std,

            "h_mean": h_mean,
            "s_mean": s_mean,
            "v_mean": v_mean,

            "h_std": h_std,
            "s_std": s_std,
            "v_std": v_std,
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
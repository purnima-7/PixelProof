import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- CHANGE THIS ----------
IMAGE_PATH = r"dataset\screen\20260702_152727.jpg"
# -------------------------------


def fft_spectrum(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.abs(fshift)
    magnitude = np.log1p(magnitude)

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return magnitude

img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Image not found!")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge Detection
edges = cv2.Canny(gray, 100, 200)

# Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian_display = cv2.convertScaleAbs(laplacian)

# FFT
fft = fft_spectrum(gray)

# Statistics
lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
edge_density = np.sum(edges > 0) / edges.size

print(f"Laplacian Variance : {lap_var:.2f}")
print(f"Edge Density       : {edge_density:.4f}")

plt.figure(figsize=(15,8))

plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(edges, cmap='gray')
plt.title("Edges")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(laplacian_display, cmap='gray')
plt.title("Laplacian")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(fft, cmap='gray')
plt.title("FFT Spectrum")
plt.axis("off")

plt.tight_layout()
plt.show()
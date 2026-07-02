import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../features.csv")

features = [
    "laplacian_variance",
    "edge_density",
    "brightness",
    "contrast",
    "entropy",
    "glare_percentage",
    "fft_high",
    "fft_low"
]

for feature in features:

    plt.figure(figsize=(8,5))

    real = df[df["label"]=="real"][feature]
    screen = df[df["label"]=="screen"][feature]

    plt.hist(real, bins=20, alpha=0.6, label="Real")
    plt.hist(screen, bins=20, alpha=0.6, label="Screen")

    plt.title(feature)

    plt.legend()

    plt.tight_layout()

    plt.savefig(f"../output/{feature}.png")

    plt.close()

print("Done!")
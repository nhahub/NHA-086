# ========================================================
# plot_utils.py  -- safe Matplotlib utilities for Flask
# ========================================================

import matplotlib
matplotlib.use("Agg")     # Safe non-GUI backend for web servers

import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO


# ---------------------------------------------
# Convert Matplotlib figure → Base64 PNG string
# ---------------------------------------------
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)  # VERY IMPORTANT: prevents memory leaks
    return encoded


# ---------------------------------------------
# Plot confusion matrix
# ---------------------------------------------
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Add text labels
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    return fig_to_base64(fig)


# ---------------------------------------------
# Plot PCA 2D scatter (with fixed axis limits)
# ---------------------------------------------
def plot_pca_scatter(X2, clusters):
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        X2[:, 0], X2[:, 1],
        c=clusters, cmap="tab10", s=8, alpha=0.6
    )

    ax.set_title("KMeans Clusters (PCA Projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # User requested fixed x-axis range
    ax.set_xlim([-5000, 5000])

    # Legend
    legend = ax.legend(*scatter.legend_elements(),
                       title="Cluster",
                       bbox_to_anchor=(1.02, 1))
    ax.add_artist(legend)

    return fig_to_base64(fig)


# ---------------------------------------------
# Spectral Signature Plot
# ---------------------------------------------
def plot_spectral_signature(band_values):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(band_values)+1), band_values, marker="o")

    ax.set_title("Spectral Signature")
    ax.set_xlabel("Band Number")
    ax.set_ylabel("Reflectance (scaled)")

    ax.grid(True, linestyle="--", alpha=0.3)
    return fig_to_base64(fig)


# ---------------------------------------------
# Histogram for each band
# ---------------------------------------------
def plot_band_histogram(band_data, band_id):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(band_data, bins=60, alpha=0.7)
    ax.set_title(f"Histogram - Band {band_id}")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Count")

    return fig_to_base64(fig)

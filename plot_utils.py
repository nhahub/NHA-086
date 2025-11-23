# plot_utils.py
import io
import base64
import matplotlib
matplotlib.use("Agg")  # ensures no GUI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid")


def fig_to_base64(fig):
    """
    Convert a matplotlib figure to a base64 PNG string
    """
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"


def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix using seaborn heatmap
    cm: numpy array (n_classes x n_classes)
    class_names: list of labels
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    return fig_to_base64(fig)


def plot_pca_scatter(X2, labels):
    """
    Plot 2D PCA scatter
    X2: numpy array (n_samples, 2)
    labels: array-like cluster/predicted labels
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab10", s=8, alpha=0.6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Scatter Plot")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", bbox_to_anchor=(1.02, 1))
    ax.add_artist(legend1)
    return fig_to_base64(fig)


def plot_spectral_signature(mean_by_class):
    """
    Plot mean spectral signature per class
    mean_by_class: dict {class_name: dict or list of bands}
      - if dict: keys = band names or integers
      - if list/array: order is assumed
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for cls, bands in mean_by_class.items():
        # convert dict to list if needed
        if isinstance(bands, dict):
            band_values = [bands[k] for k in sorted(bands.keys())]
        else:
            band_values = list(bands)
        ax.plot(range(1, len(band_values) + 1), band_values, marker="o", label=str(cls))

    ax.set_xlabel("Band Index")
    ax.set_ylabel("Mean Reflectance / Value")
    ax.set_title("Spectral Signature per Class")
    ax.legend(fontsize="small", ncol=2)
    ax.grid(True)
    return fig_to_base64(fig)


def plot_band_histogram(samples, band_labels=None, bins=30):
    """
    Plot histograms for multiple bands
    samples: numpy array (n_samples, n_bands)
    band_labels: optional list of names per band
    """
    n_bands = samples.shape[1]
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(n_bands):
        label = band_labels[i] if band_labels is not None else f"B{i+1}"
        ax.hist(samples[:, i], bins=bins, alpha=0.4, label=label)
    ax.set_xlabel("Reflectance / Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Band Histograms")
    ax.legend(fontsize="small", ncol=3)
    ax.grid(True)
    return fig_to_base64(fig)

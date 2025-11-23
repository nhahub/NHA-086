# app.py
import os
import io
import base64
from collections import Counter

import numpy as np
import rasterio
import joblib
from flask import Flask, render_template, request, redirect, url_for

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

# import plotting utils (must exist)
from plot_utils import (
    fig_to_base64,
    plot_confusion_matrix,
    plot_pca_scatter,
    plot_spectral_signature,
    plot_band_histogram
)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# sampling params
SAMPLE_PER_IMAGE = int(os.getenv("SAMPLE_PER_IMAGE", 400))
SAMPLE_PER_IMAGE_CM = int(os.getenv("SAMPLE_PER_IMAGE_CM", 200))
PCA_SAMPLE = int(os.getenv("PCA_SAMPLE", 2000))

# models paths (put models in ./models or mount them)
MODEL_DIR = os.getenv("MODEL_DIR", "models")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_landtype.pkl")
XGB_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")   # or rf_land_classifier.pkl
RF_PATH = os.path.join(MODEL_DIR, "rf_land_classifier.pkl")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.pkl")
CLUSTER_MAP_PATH = os.path.join(MODEL_DIR, "cluster_map.pkl")  # optional mapping cluster->class_id

TEST_DATASET_PATH = os.getenv("TEST_DATASET_PATH", "test_dataset")  # for confusion, kmeans viz, spectra

# ---------------- LOAD MODELS ----------------
kmeans_model = None
classifier_model = None
label_map = {}
cluster_map = {}

# try RF first, then XGBoost style model
if os.path.exists(RF_PATH):
    classifier_model = joblib.load(RF_PATH)
elif os.path.exists(XGB_PATH):
    classifier_model = joblib.load(XGB_PATH)

if os.path.exists(KMEANS_PATH):
    kmeans_model = joblib.load(KMEANS_PATH)

if os.path.exists(LABEL_MAP_PATH):
    label_map = joblib.load(LABEL_MAP_PATH)
    # ensure int keys
    label_map = {int(k): v for k, v in label_map.items()}
id2label = {int(k): label_map[k] for k in label_map} if label_map else {}
label2id = {v: k for k, v in id2label.items()}

if os.path.exists(CLUSTER_MAP_PATH):
    cluster_map = joblib.load(CLUSTER_MAP_PATH)


# ---------------- HELPERS ----------------
def sample_pixels_from_tif(path, n):
    """Randomly sample n pixels (spectral vectors) from GeoTIFF without loading full image into memory."""
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        total = h * w
        nsel = min(n, total)
        idx = np.random.choice(total, nsel, replace=False)
        rows = idx // w
        cols = idx % w
        pixels = []
        for r, c in zip(rows, cols):
            pv = src.read(window=((r, r+1), (c, c+1)))[:, 0, 0]
            pixels.append(pv)
    return np.array(pixels, dtype=np.float32)


def create_rgb_base64(path):
    try:
        with rasterio.open(path) as src:
            if src.count < 4:
                return None
            r = src.read(4).astype(np.float32)
            g = src.read(3).astype(np.float32)
            b = src.read(2).astype(np.float32)
        rgb = np.stack([r, g, b], axis=-1)
        # clip/normalize
        p2 = np.nanpercentile(rgb, 2)
        p98 = np.nanpercentile(rgb, 98)
        rgb = (rgb - p2) / (p98 - p2 + 1e-10)
        rgb = np.clip(rgb, 0, 1)
        # create matplotlib figure and return base64 (use fig_to_base64)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(rgb)
        ax.axis("off")
        return fig_to_base64(fig)
    except Exception:
        return None


def create_ndvi_base64(path):
    try:
        with rasterio.open(path) as src:
            if src.count < 8:
                return None
            b8 = src.read(8).astype(np.float32)
            b4 = src.read(4).astype(np.float32)
        ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
        fig, ax = plt.subplots(figsize=(5, 5))
        cax = ax.imshow(ndvi, cmap="RdYlGn")
        ax.axis("off")
        fig.colorbar(cax, ax=ax, fraction=0.03)
        return fig_to_base64(fig)
    except Exception:
        return None


def overlay_band_histograms(samples):
    """samples: (N, bands) -> overlay histogram figure base64 using plot_band_histogram helper for each band? 
       We'll create an overlay figure here and return base64 via fig_to_base64 from plot_utils.
    """
    # We'll reuse plot_band_histogram for each band and assemble a combined figure,
    # but to keep consistent we create an overlay figure here and call fig_to_base64.
    import matplotlib.pyplot as plt
    bands = samples.shape[1]
    fig, ax = plt.subplots(figsize=(10, 4))
    for b in range(bands):
        ax.hist(samples[:, b], bins=40, alpha=0.35, label=f"B{b+1}")
    ax.legend(ncol=3, fontsize="small")
    ax.set_title("Sampled Band Histograms")
    return fig_to_base64(fig)


# ---------------- FLASK APP ----------------
from flask import Flask, render_template, request
import matplotlib.pyplot as plt  # local import after backend set in plot_utils

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    if request.method == "POST":
        f = request.files.get("tif_file")
        if not f:
            return render_template("index.html", error="No file uploaded")
        fname = f.filename
        savepath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        f.save(savepath)

        # sample pixels fast
        samples = sample_pixels_from_tif(savepath, n=SAMPLE_PER_IMAGE)

        # classifier predictions (majority vote)
        if classifier_model is not None:
            preds = classifier_model.predict(samples)
            top = Counter(preds).most_common(1)[0][0]
            result["classifier"] = {"id": int(top), "name": id2label.get(int(top), str(top))}
            # probabilities if available
            if hasattr(classifier_model, "predict_proba"):
                probs = classifier_model.predict_proba(samples).mean(axis=0)
                result["classifier"]["probs"] = [{"class": id2label.get(i, str(i)), "prob": float(probs[i])} for i in range(len(probs))]
        else:
            result["classifier"] = None

        # kmeans predictions if available
        if kmeans_model is not None:
            clusters = kmeans_model.predict(samples)
            if cluster_map:
                mapped = [cluster_map.get(int(c), None) for c in clusters]
                result["kmeans_mapped_counts"] = dict(Counter(mapped))
            else:
                result["kmeans_cluster_counts"] = dict(Counter(clusters))

        # visuals
        result["rgb"] = create_rgb_base64(savepath)
        result["ndvi"] = create_ndvi_base64(savepath)
        result["hist"] = overlay_band_histograms(samples)

        # PCA scatter colored by classifier preds or clusters
        if classifier_model is not None:
            pca_b64 = None
            try:
                pca = PCA(n_components=2)
                # reduce to PCA_SAMPLE points for plotting
                if samples.shape[0] > PCA_SAMPLE:
                    idx = np.random.choice(samples.shape[0], PCA_SAMPLE, replace=False)
                    sub = samples[idx]
                    sub_preds = classifier_model.predict(sub)
                else:
                    sub = samples
                    sub_preds = classifier_model.predict(sub)
                # create PCA 2D
                pca2 = pca.fit_transform(sub)
                pca_b64 = plot_pca_scatter(pca2, sub_preds)
            except Exception:
                pca_b64 = None
            result["pca_by_classifier"] = pca_b64
        elif kmeans_model is not None:
            try:
                pca = PCA(n_components=2)
                if samples.shape[0] > PCA_SAMPLE:
                    idx = np.random.choice(samples.shape[0], PCA_SAMPLE, replace=False)
                    sub = samples[idx]
                    sub_clusters = kmeans_model.predict(sub)
                else:
                    sub = samples
                    sub_clusters = kmeans_model.predict(sub)
                pca2 = pca.fit_transform(sub)
                pca_b64 = plot_pca_scatter(pca2, sub_clusters)
            except Exception:
                pca_b64 = None
            result["pca_by_kmeans"] = pca_b64

        # spectral signatures per predicted class (if classifier preds exist)
        if classifier_model is not None:
            sig = {}
            preds = classifier_model.predict(samples)
            for lbl in np.unique(preds):
                sig[id2label.get(int(lbl), str(lbl))] = samples[preds == lbl].mean(axis=0)
            result["spectral_signatures"] = plot_spectral_signature(sig)

    return render_template("index.html", result=result)


@app.route("/confusion_matrix")
def confusion_matrix_route():
    if classifier_model is None:
        return "No classifier model loaded.", 400
    test_root = TEST_DATASET_PATH
    if not os.path.isdir(test_root):
        return f"Set TEST_DATASET_PATH to a valid directory containing class subfolders.", 400

    y_true = []
    y_pred = []
    classes = sorted([d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))])
    for cls_name in classes:
        cls_dir = os.path.join(test_root, cls_name)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(".tif"):
                continue
            p = os.path.join(cls_dir, fname)
            samples = sample_pixels_from_tif(p, n=SAMPLE_PER_IMAGE_CM)
            preds = classifier_model.predict(samples)
            top = Counter(preds).most_common(1)[0][0]
            y_pred.append(int(top))
            true_id = label2id.get(cls_name)
            if true_id is None:
                continue
            y_true.append(int(true_id))

    if len(y_true) == 0:
        return "No labeled test images found or label_map mismatch.", 400

    labels_sorted = sorted(id2label.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    cm_b64 = plot_confusion_matrix(cm, [id2label[i] for i in labels_sorted])

    report = classification_report(y_true, y_pred, labels=labels_sorted, target_names=[id2label[i] for i in labels_sorted], output_dict=True)
    return render_template("confusion.html", cm_img=cm_b64, report=report)


@app.route("/kmeans_viz")
def kmeans_viz_route():
    if kmeans_model is None:
        return "KMeans model not loaded.", 400
    data_root = TEST_DATASET_PATH
    if not os.path.isdir(data_root):
        return f"Set TEST_DATASET_PATH to a valid folder.", 400

    pooled = []
    for cls in sorted(os.listdir(data_root)):
        cls_dir = os.path.join(data_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".tif")]
        for f in files[:4]:
            p = os.path.join(cls_dir, f)
            pix = sample_pixels_from_tif(p, n=int(SAMPLE_PER_IMAGE//2))
            pooled.append(pix)
    if not pooled:
        return "No files found in TEST_DATASET_PATH", 400

    Xpool = np.vstack(pooled)
    clusters = kmeans_model.predict(Xpool)
    # PCA reduction and call plot_pca_scatter (it expects X2 and clusters)
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(Xpool if Xpool.shape[0] <= PCA_SAMPLE else Xpool[np.random.choice(Xpool.shape[0], PCA_SAMPLE, replace=False)])
    # create subset clusters for plotted points:
    if Xpool.shape[0] > PCA_SAMPLE:
        subset_idx = np.random.choice(Xpool.shape[0], PCA_SAMPLE, replace=False)
        clustered_subset = clusters[subset_idx]
    else:
        clustered_subset = clusters
    kmeans_img = plot_pca_scatter(X2, clustered_subset)
    return render_template("kmeans.html", kmeans_img=kmeans_img)


@app.route("/spectral_signatures")
def spectral_signatures_route():
    root = TEST_DATASET_PATH
    if not os.path.isdir(root):
        return f"Set TEST_DATASET_PATH to a valid folder.", 400

    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    mean_by_class = {}
    for cls_name in classes:
        cls_dir = os.path.join(root, cls_name)
        collected = []
        for f in os.listdir(cls_dir)[:8]:
            if not f.lower().endswith(".tif"):
                continue
            p = os.path.join(cls_dir, f)
            pix = sample_pixels_from_tif(p, n=max(50, SAMPLE_PER_IMAGE//4))
            collected.append(np.mean(pix, axis=0))
        if collected:
            mean_by_class[cls_name] = np.mean(np.stack(collected, axis=0), axis=0)

    if not mean_by_class:
        return "No images found under TEST_DATASET_PATH", 400

    sig_b64 = plot_spectral_signature(mean_by_class)
    return render_template("spectra.html", sig_img=sig_b64)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

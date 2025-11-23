# app.py
# Flask web UI using XGBoost Booster for prediction (low-memory sampling + visualizations)
import os
import io
import base64
from collections import Counter

import numpy as np
import rasterio
import joblib
import xgboost as xgb

# ensure non-GUI matplotlib backend (plot_utils also sets it, but safe to set early)
import matplotlib
matplotlib.use("Agg")

from flask import Flask, render_template, request
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

# plotting utilities you created earlier (must implement fig_to_base64, plot_confusion_matrix, plot_pca_scatter, plot_spectral_signature, plot_band_histogram)
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

MODEL_DIR = os.getenv("MODEL_DIR", "models")
BOOSTER_PATH = os.path.join(MODEL_DIR, "xgb_booster.json")  # raw Booster file
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.pkl")   # int -> class name
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_landtype.pkl")  # optional

TEST_DATASET_PATH = os.getenv("TEST_DATASET_PATH", "test_dataset")

# sampling / plotting params
SAMPLE_PER_IMAGE = int(os.getenv("SAMPLE_PER_IMAGE", 400))
SAMPLE_PER_IMAGE_CM = int(os.getenv("SAMPLE_PER_IMAGE_CM", 200))
PCA_SAMPLE = int(os.getenv("PCA_SAMPLE", 2000))

# ---------------- LOAD MODELS ----------------
booster = None
kmeans_model = None
label_map = {}

if os.path.exists(BOOSTER_PATH):
    booster = xgb.Booster()
    booster.load_model(BOOSTER_PATH)
else:
    print(f"[WARN] Booster not found at {BOOSTER_PATH} — prediction disabled.")

if os.path.exists(KMEANS_PATH):
    try:
        kmeans_model = joblib.load(KMEANS_PATH)
    except Exception as e:
        print("Warning loading kmeans:", e)

if os.path.exists(LABEL_MAP_PATH):
    label_map = joblib.load(LABEL_MAP_PATH)
    # Option 1 confirmed: label_map is int -> name
    label_map = {int(k): v for k, v in label_map.items()}
else:
    print(f"[WARN] label_map not found at {LABEL_MAP_PATH} — labels may be missing.")

# convenience mappings
id2label = label_map
label2id = {v: k for k, v in id2label.items()}

# ---------------- FLASK APP ----------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------------- UTILS ----------------
def sample_pixels_from_tif(path, n):
    """
    Randomly sample n pixel spectra from GeoTIFF without loading full image.
    Returns array shape (nsel, bands).
    """
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        total = h * w
        nsel = min(n, total)
        idx = np.random.choice(total, nsel, replace=False)
        rows = idx // w
        cols = idx % w
        samples = []
        for r, c in zip(rows, cols):
            # read single pixel window
            pv = src.read(window=((r, r+1), (c, c+1)))[:, 0, 0]
            samples.append(pv)
    return np.array(samples, dtype=np.float32)


def create_rgb_base64(path):
    """True color B4,B3,B2 preview; returns base64 PNG string or None"""
    try:
        with rasterio.open(path) as src:
            if src.count < 4:
                return None
            r = src.read(4).astype(np.float32)
            g = src.read(3).astype(np.float32)
            b = src.read(2).astype(np.float32)
        rgb = np.stack([r, g, b], axis=-1)
        # robust percentile normalization
        p2 = np.nanpercentile(rgb, 2)
        p98 = np.nanpercentile(rgb, 98)
        rgb = (rgb - p2) / (p98 - p2 + 1e-10)
        rgb = np.clip(rgb, 0, 1)
        # create fig
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(rgb)
        ax.axis("off")
        return fig_to_base64(fig)
    except Exception as e:
        print("RGB preview error:", e)
        return None


def create_ndvi_base64(path):
    """NDVI preview using B8 and B4 if available."""
    try:
        with rasterio.open(path) as src:
            if src.count < 8:
                return None
            b8 = src.read(8).astype(np.float32)
            b4 = src.read(4).astype(np.float32)
        ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        cax = ax.imshow(ndvi, cmap="RdYlGn")
        ax.axis("off")
        fig.colorbar(cax, ax=ax, fraction=0.03)
        return fig_to_base64(fig)
    except Exception as e:
        print("NDVI preview error:", e)
        return None


# ---------------- PREDICTION (Booster) ----------------
def predict_booster_on_samples(samples):
    """
    samples: (n, bands)
    Returns:
      preds (ndarray shape (n,)) -> class indices
      probs (ndarray shape (n, num_classes)) -> per-class probabilities
    """
    if booster is None:
        return None, None
    dmat = xgb.DMatrix(samples)
    proba = booster.predict(dmat)  # (n, num_class) for multi:softprob
    # proba shape might be (n, num_class) or (n,) for binary; handle both
    proba = np.array(proba)
    if proba.ndim == 1:
        # binary case: booster returns single probability for class 1
        probs = np.vstack([1 - proba, proba]).T
    else:
        probs = proba
    preds = np.argmax(probs, axis=1)
    return preds, probs


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    """
    Upload one TIFF, run booster prediction on sampled pixels,
    show RGB/NDVI/histograms, PCA scatter and spectral signatures.
    """
    result = {}
    if request.method == "POST":
        f = request.files.get("tif_file")
        if not f:
            return render_template("index.html", error="No file uploaded", result=None)

        filename = f.filename
        savepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(savepath)

        # sample pixels
        samples = sample_pixels_from_tif(savepath, SAMPLE_PER_IMAGE)

        # booster predictions
        preds, probs = predict_booster_on_samples(samples)
        if preds is not None:
            # majority vote per image (from sampled pixels)
            top = int(Counter(preds).most_common(1)[0][0])
            name = id2label.get(top, str(top))
            # mean probability across sampled pixels
            mean_probs = probs.mean(axis=0) if probs is not None else None
            prob_list = [{"class": id2label.get(i, str(i)), "prob": float(mean_probs[i])} for i in range(len(mean_probs))] if mean_probs is not None else None
            result["booster"] = {"id": top, "name": name, "probs": prob_list}
        else:
            result["booster"] = None

        # kmeans if available
        if kmeans_model is not None:
            clusters = kmeans_model.predict(samples)
            result["kmeans_cluster_counts"] = dict(Counter(clusters))

        # visuals
        result["rgb"] = create_rgb_base64(savepath)
        result["ndvi"] = create_ndvi_base64(savepath)

        # band histogram (overlay all bands) using plot_band_histogram per band or combined
        try:
            # create combined overlay histogram manually and return base64 via fig_to_base64
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            # samples shape (n, bands)
            bands = samples.shape[1]
            for b in range(bands):
                ax.hist(samples[:, b], bins=30, alpha=0.35, label=f"B{b+1}")
            ax.legend(ncol=3, fontsize="small")
            ax.set_title("Sampled Band Histograms")
            hist_b64 = fig_to_base64(fig)
            result["hist"] = hist_b64
        except Exception as e:
            print("Histogram error:", e)
            result["hist"] = None

        # PCA scatter: color by booster prediction if available else by kmeans cluster
        try:
            if preds is not None:
                pca = PCA(n_components=2)
                if samples.shape[0] > PCA_SAMPLE:
                    idx = np.random.choice(samples.shape[0], PCA_SAMPLE, replace=False)
                    sub = samples[idx]
                    sub_preds = preds[idx]
                else:
                    sub = samples
                    sub_preds = preds
                X2 = pca.fit_transform(sub)
                # plot_pca_scatter expects X2 and clusters (labels)
                pca_b64 = plot_pca_scatter(X2, sub_preds)
                result["pca"] = pca_b64
            elif kmeans_model is not None:
                pca = PCA(n_components=2)
                if samples.shape[0] > PCA_SAMPLE:
                    idx = np.random.choice(samples.shape[0], PCA_SAMPLE, replace=False)
                    sub = samples[idx]
                    sub_clusters = kmeans_model.predict(sub)
                else:
                    sub = samples
                    sub_clusters = kmeans_model.predict(sub)
                X2 = pca.fit_transform(sub)
                pca_b64 = plot_pca_scatter(X2, sub_clusters)
                result["pca"] = pca_b64
            else:
                result["pca"] = None
        except Exception as e:
            print("PCA error:", e)
            result["pca"] = None

        # Spectral signatures per predicted class (mean spectra for each predicted label in sampled pixels)
        if preds is not None:
            mean_by_class = {}
            for lbl in np.unique(preds):
                name = id2label.get(int(lbl), str(lbl))
                mean_by_class[name] = samples[preds == lbl].mean(axis=0)
            try:
                sig_b64 = plot_spectral_signature(mean_by_class)
                result["spectral_signatures"] = sig_b64
            except Exception as e:
                print("Spectral signature plot error:", e)
                result["spectral_signatures"] = None

    return render_template("index.html", result=result)


@app.route("/confusion_matrix")
def confusion_matrix_route():
    """
    Compute confusion matrix on labeled TEST_DATASET_PATH.
    For each image, sample SAMPLE_PER_IMAGE_CM pixels, predict, majority-vote per image.
    """
    if booster is None:
        return "Booster model not loaded.", 400

    test_root = TEST_DATASET_PATH
    if not os.path.isdir(test_root):
        return f"Set TEST_DATASET_PATH to a valid folder (current: {TEST_DATASET_PATH})", 400

    y_true = []
    y_pred = []

    classes = sorted([d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))])
    for cls_name in classes:
        cls_dir = os.path.join(test_root, cls_name)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(".tif"):
                continue
            path = os.path.join(cls_dir, fname)
            samples = sample_pixels_from_tif(path, SAMPLE_PER_IMAGE_CM)
            preds, _ = predict_booster_on_samples(samples)
            if preds is None or len(preds) == 0:
                continue
            top = int(Counter(preds).most_common(1)[0][0])
            # predicted class id
            y_pred.append(top)
            # true id via label2id mapping (label_map is int->name so invert)
            true_id = None
            # label_map is int->name (Option 1), find key for cls_name
            true_id = None
            for k, v in id2label.items():
                if v == cls_name:
                    true_id = int(k)
                    break
            if true_id is None:
                # skip if mapping missing
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
    """
    Visualize KMeans clusters across a sample of images (PCA projection).
    """
    if kmeans_model is None:
        return "KMeans model not available.", 400

    data_root = TEST_DATASET_PATH
    if not os.path.isdir(data_root):
        return f"Set TEST_DATASET_PATH to a valid folder (current: {TEST_DATASET_PATH})", 400

    pooled = []
    for cls in sorted(os.listdir(data_root)):
        cls_dir = os.path.join(data_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".tif")]
        for f in files[:4]:
            p = os.path.join(cls_dir, f)
            pix = sample_pixels_from_tif(p, n=max(50, SAMPLE_PER_IMAGE // 2))
            pooled.append(pix)
    if not pooled:
        return "No images found in TEST_DATASET_PATH", 400

    Xpool = np.vstack(pooled)
    clusters = kmeans_model.predict(Xpool)

    # PCA reduce (sample for plotting)
    if Xpool.shape[0] > PCA_SAMPLE:
        subset_idx = np.random.choice(Xpool.shape[0], PCA_SAMPLE, replace=False)
        Xsub = Xpool[subset_idx]
        cluster_sub = clusters[subset_idx]
    else:
        Xsub = Xpool
        cluster_sub = clusters

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(Xsub)
    kmeans_img = plot_pca_scatter(X2, cluster_sub)
    return render_template("kmeans.html", kmeans_img=kmeans_img)


@app.route("/spectral_signatures")
def spectral_signatures_route():
    """
    Compute mean spectral signature per class (sample up to 8 images per class).
    """
    root = TEST_DATASET_PATH
    if not os.path.isdir(root):
        return f"Set TEST_DATASET_PATH to a valid folder (current: {TEST_DATASET_PATH})", 400

    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    mean_by_class = {}
    for cls_name in classes:
        cls_dir = os.path.join(root, cls_name)
        collected = []
        for f in os.listdir(cls_dir)[:8]:
            if not f.lower().endswith(".tif"):
                continue
            p = os.path.join(cls_dir, f)
            pix = sample_pixels_from_tif(p, n=max(50, SAMPLE_PER_IMAGE // 4))
            collected.append(np.mean(pix, axis=0))
        if collected:
            mean_by_class[cls_name] = np.mean(np.stack(collected, axis=0), axis=0)

    if not mean_by_class:
        return "No images found under TEST_DATASET_PATH", 400

    sig_b64 = plot_spectral_signature(mean_by_class)
    return render_template("spectra.html", sig_img=sig_b64)


# ---------------- RUN ----------------
if __name__ == "__main__":
    # debug False in production; host 0.0.0.0 to be reachable in container
    app.run(host="0.0.0.0", port=5000, debug=False)

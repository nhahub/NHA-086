# app.py
import os
import io
import base64
import numpy as np
import rasterio
import joblib
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns

# -------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Sampling / speed controls
SAMPLE_PER_IMAGE = 400   # number of pixels sampled per image for visualization/prediction
SAMPLE_FOR_CM = 200      # number of pixels/sample when building confusion matrix per image

# Models and data paths (change if needed)
RF_MODEL_PATH = "rf_land_classifier.pkl"
KMEANS_MODEL_PATH = "kmeans_landtype.pkl"
LABEL_MAP_PATH = "label_map.pkl"
TEST_DATASET_PATH = "test_dataset"  # path: test_dataset/ClassName/*.tif

# -------------- LOAD MODELS -------------
clf = joblib.load(RF_MODEL_PATH)
kmeans_bundle = None
try:
    kmeans = joblib.load(KMEANS_MODEL_PATH)
except Exception:
    kmeans = None
label_map = joblib.load(LABEL_MAP_PATH)  # expected dict int->class_name
id2label = {int(k): label_map[k] for k in label_map}
label2id = {v: int(k) for k, v in label_map.items()}

# -------------- FLASK -------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------- UTIL --------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{data}"

def sample_pixels_from_path(path, n=SAMPLE_PER_IMAGE):
    """Read a tif and sample n pixels (spectral vectors). Returns (n, bands)."""
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        total = h * w
        nsel = min(n, total)
        idx = np.random.choice(total, nsel, replace=False)
        rows = idx // w
        cols = idx % w
        pixels = []
        for r, c in zip(rows, cols):
            px = src.read(window=((r, r+1), (c, c+1)))[:, 0, 0]
            pixels.append(px)
    return np.array(pixels, dtype=np.float32)

def create_rgb_base64(path):
    """Create RGB preview (bands 4,3,2) if available"""
    try:
        with rasterio.open(path) as src:
            if src.count < 4:
                return None
            r = src.read(4).astype(np.float32)
            g = src.read(3).astype(np.float32)
            b = src.read(2).astype(np.float32)
        rgb = np.stack([r, g, b], axis=-1)
        # simple normalization
        rgb = (rgb - np.nanpercentile(rgb, 2)) / (np.nanpercentile(rgb, 98) - np.nanpercentile(rgb, 2) + 1e-10)
        rgb = np.clip(rgb, 0, 1)
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(rgb)
        ax.axis('off')
        return fig_to_base64(fig)
    except Exception:
        return None

def compute_ndvi_image_base64(path):
    """Compute NDVI and return plotted figure base64 (if B8 & B4 exist)"""
    try:
        with rasterio.open(path) as src:
            if src.count < 8:
                return None
            b8 = src.read(8).astype(np.float32)
            b4 = src.read(4).astype(np.float32)
        ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
        fig, ax = plt.subplots(figsize=(5,5))
        cax = ax.imshow(ndvi, cmap='RdYlGn')
        ax.axis('off')
        fig.colorbar(cax, ax=ax, fraction=0.03)
        return fig_to_base64(fig)
    except Exception:
        return None

def band_histogram_base64(pixels):
    """pixels: (N, bands)"""
    bands = pixels.shape[1]
    fig, ax = plt.subplots(figsize=(8,4))
    # Plot hist per band (overlay)
    for b in range(bands):
        ax.hist(pixels[:, b], bins=40, alpha=0.4, label=f"B{b+1}")
    ax.legend(ncol=3, fontsize='small')
    ax.set_title("Sampled Band Histograms")
    return fig_to_base64(fig)

def spectral_signature_plot_base64(df_mean_by_class):
    """df_mean_by_class: dict class->mean_vector (bands,)"""
    fig, ax = plt.subplots(figsize=(8,5))
    bands = None
    for cls, vec in df_mean_by_class.items():
        if bands is None:
            bands = np.arange(1, len(vec)+1)
        ax.plot(bands, vec, marker='o', label=cls)
    ax.set_xlabel("Band #")
    ax.set_ylabel("Mean Reflectance (sampled)")
    ax.set_title("Spectral Signatures (per class)")
    ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize='small')
    plt.tight_layout()
    return fig_to_base64(fig)

# -------------- ROUTES ------------------
@app.route("/", methods=['GET','POST'])
def index():
    result = None
    rgb_img = None
    ndvi_img = None
    hist_img = None
    if request.method == 'POST':
        f = request.files.get('tif_file')
        if f:
            fname = f.filename
            savepath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            f.save(savepath)

            # Fast sampled prediction
            pixels = sample_pixels_from_path(savepath, n=SAMPLE_PER_IMAGE)
            preds = clf.predict(pixels)
            top = np.bincount(preds).argmax()
            result = id2label.get(int(top), str(top))

            # visuals
            rgb_img = create_rgb_base64(savepath)
            ndvi_img = compute_ndvi_image_base64(savepath)
            hist_img = band_histogram_base64(pixels)

    return render_template("index.html", result=result, rgb=rgb_img, ndvi=ndvi_img, hist=hist_img)


@app.route("/predict_json", methods=['POST'])
def predict_json():
    """API returning prediction + probs + visuals as base64 JSON (for JS)"""
    f = request.files.get('file')
    if not f:
        return jsonify({"error":"no file"}), 400
    savepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(savepath)

    pixels = sample_pixels_from_path(savepath, n=SAMPLE_PER_IMAGE)
    preds = clf.predict(pixels)
    top = np.bincount(preds).argmax()
    label = id2label.get(int(top), str(top))

    # probabilities (mean prob across sampled pixels)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(pixels).mean(axis=0)
        prob_list = [{"class": id2label[i], "prob": float(probs[i])} for i in range(len(probs))]
    else:
        prob_list = []

    rgb_img = create_rgb_base64(savepath)
    ndvi_img = compute_ndvi_image_base64(savepath)
    hist_img = band_histogram_base64(pixels)

    return jsonify({"prediction": label, "probs": prob_list, "rgb": rgb_img, "ndvi": ndvi_img, "hist": hist_img})


@app.route("/confusion_matrix")
def confusion_matrix_page():
    """Compute confusion matrix over TEST_DATASET_PATH (samples per image)."""
    test_root = TEST_DATASET_PATH
    if not os.path.isdir(test_root):
        return "Set TEST_DATASET_PATH in app to a valid folder with class subfolders.", 400

    y_true = []
    y_pred = []
    classes = sorted([d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))])
    # class order must match label_map keys/names
    for cls_name in classes:
        cls_dir = os.path.join(test_root, cls_name)
        for file in os.listdir(cls_dir):
            if not file.lower().endswith(".tif"):
                continue
            path = os.path.join(cls_dir, file)
            pixels = sample_pixels_from_path(path, n=SAMPLE_FOR_CM)
            preds = clf.predict(pixels)
            # majority vote per image
            pred_label_id = np.bincount(preds).argmax()
            y_pred.append(pred_label_id)
            # true id
            true_id = label2id.get(cls_name, None)
            if true_id is None:
                # skip if label not in label_map
                continue
            y_true.append(true_id)

    # build confusion matrix; align with id2label order
    labels_sorted = [id2label[i] for i in sorted(id2label.keys())]
    cm = confusion_matrix(y_true, y_pred, labels=sorted(id2label.keys()))
    # plot
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_sorted, yticklabels=labels_sorted, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (sampled per-image)")
    cm_b64 = fig_to_base64(fig)
    return render_template("confusion.html", cm_img=cm_b64)


@app.route("/kmeans_viz")
def kmeans_viz_page():
    """Visualize KMeans clusters (requires kmeans model). We sample several images across classes
       and create a PCA 2D scatter colored by cluster id; if mapping exists we show mapped class color too.
    """
    if kmeans is None:
        return "KMeans model not available.", 400

    # build a sampled dataset of pixels across classes (small)
    sample_pool = []
    sample_labels = []
    data_root = TEST_DATASET_PATH
    if not os.path.isdir(data_root):
        return "Set TEST_DATASET_PATH to a valid folder for KMeans viz.", 400

    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    for cls_name in classes:
        cls_dir = os.path.join(data_root, cls_name)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(".tif")]
        # sample up to 5 images per class
        for f in files[:5]:
            p = os.path.join(cls_dir, f)
            pix = sample_pixels_from_path(p, n=int(SAMPLE_PER_IMAGE/2))
            sample_pool.append(pix)
            sample_labels += [cls_name] * pix.shape[0]

    X = np.vstack(sample_pool)
    sample_labels = np.array(sample_labels)

    # reduce to 2D for plotting with PCA
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    clusters = kmeans.predict(X)

    # plot scatter with cluster color and small alpha
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(X2[:,0], X2[:,1], c=clusters, cmap='tab10', s=8, alpha=0.6)
    ax.set_title("KMeans clusters (PCA projection)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_xlim(-5000, 5000)
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", bbox_to_anchor=(1.02,1))
    ax.add_artist(legend1)
    kmeans_b64 = fig_to_base64(fig)
    return render_template("kmeans.html", kmeans_img=kmeans_b64)


@app.route("/spectral_signatures")
def spectral_signatures_page():
    """Compute mean spectral signature per class (samples) and plot"""
    root = TEST_DATASET_PATH
    if not os.path.isdir(root):
        return "Set TEST_DATASET_PATH to a valid folder.", 400

    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    mean_by_class = {}
    for cls_name in classes:
        cls_dir = os.path.join(root, cls_name)
        collected = []
        for f in os.listdir(cls_dir)[:8]:          # sample up to 8 images per class
            if not f.lower().endswith(".tif"):
                continue
            p = os.path.join(cls_dir, f)
            pix = sample_pixels_from_path(p, n=int(SAMPLE_PER_IMAGE/4))
            collected.append(np.mean(pix, axis=0))
        if collected:
            mean_by_class[cls_name] = np.mean(np.stack(collected, axis=0), axis=0)

    sig_b64 = spectral_signature_plot_base64(mean_by_class)
    return render_template("spectra.html", sig_img=sig_b64)

# -------------- RUN ---------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

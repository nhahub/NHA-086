# app.py
import os
from flask import Flask, request, render_template_string, jsonify
import rasterio
import numpy as np
import joblib
import xgboost as xgb

MODEL_DIR = "models"
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_classifier.joblib")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_cluster_and_map.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
IMPUTER_PATH = os.path.join(MODEL_DIR, "imputer.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_land_classifier.joblib")
label_map_path = os.path.join(MODEL_DIR, "label_map.joblib")

# Load models
rf = joblib.load(RF_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)
#kmeans_bundle = joblib.load(KMEANS_PATH)
#kmeans = kmeans_bundle['kmeans']
#cluster_map = kmeans_bundle['mapping']
scaler = joblib.load(SCALER_PATH)
imputer = joblib.load(IMPUTER_PATH)
label_map = joblib.load(label_map_path)

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Minimal HTML template (upload form) — enhanced layout + placeholders for visualizations
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Land Type Classifier</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style> body { padding: 1.2rem; } .img-panel img { max-width:100%; height:auto; } </style>
</head>
<body>
<div class="container">
  <h1 class="mb-3">Land Type Classifier</h1>
  <div class="card mb-3">
    <div class="card-body">
      <form method="post" enctype="multipart/form-data" action="/predict" class="row g-3">
        <div class="col-md-6">
          <input class="form-control" type="file" name="file" accept=".tif,.tiff" required>
        </div>
        <div class="col-md-3">
          <select name="model" class="form-select">
            <option value="rf">RandomForest (supervised)</option>
            <option value="xgb_model">XGBoosting (supervised)</option>
          </select>
        </div>
        <div class="col-md-3">
          <button class="btn btn-primary w-100" type="submit">Upload & Predict</button>
        </div>
      </form>
    </div>
  </div>

  {% if result %}
  <div class="row">
    <div class="col-md-4">
      <div class="card mb-3">
        <div class="card-header">Prediction</div>
        <div class="card-body">
          <p><strong>File:</strong> {{ filename }}</p>
          <pre>{{ result | tojson(indent=2) }}</pre>
        </div>
      </div>
    </div>

    <div class="col-md-8">
      <div class="card mb-3">
        <div class="card-header">Visualizations</div>
        <div class="card-body">
          <div class="row">
            {% if img_rgb %}
            <div class="col-md-6 img-panel">
              <h6>RGB preview</h6>
              <img src="data:image/png;base64,{{ img_rgb }}" alt="rgb preview">
            </div>
            {% endif %}
            {% if img_ndvi %}
            <div class="col-md-6 img-panel">
              <h6>NDVI</h6>
              <img src="data:image/png;base64,{{ img_ndvi }}" alt="ndvi">
            </div>
            {% endif %}
            {% if img_ndwi %}
            <div class="col-md-6 img-panel">
              <h6>NDWI</h6>
              <img src="data:image/png;base64,{{ img_ndwi }}" alt="ndwi">
            </div>
            {% endif %}
            {% if img_ndbi %}
            <div class="col-md-6 img-panel">
              <h6>NDBI</h6>
              <img src="data:image/png;base64,{{ img_ndbi }}" alt="ndbi">
            </div>
            {% endif %}
          </div>

          <div class="row mt-3">
            {% if img_spec %}
            <div class="col-md-6 img-panel">
              <h6>Spectral signature (band means)</h6>
              <img src="data:image/png;base64,{{ img_spec }}" alt="spectral signature">
            </div>
            {% endif %}
            {% if img_hist %}
            <div class="col-md-6 img-panel">
              <h6>Band histograms (first bands)</h6>
              <img src="data:image/png;base64,{{ img_hist }}" alt="band histograms">
            </div>
            {% endif %}
          </div>

          <div class="row mt-3">
            {% if img_sample_hist %}
            <div class="col-md-6 img-panel">
              <h6>Sampled band histograms</h6>
              <img src="data:image/png;base64,{{ img_sample_hist }}" alt="sampled hist">
            </div>
            {% endif %}
            {% if img_pca %}
            <div class="col-md-6 img-panel">
              <h6>PCA (sampled pixels)</h6>
              <img src="data:image/png;base64,{{ img_pca }}" alt="pca sampled">
            </div>
            {% endif %}
          </div>

        </div>
      </div>
    </div>
  </div>
  {% endif %}
</div>
</body>
</html>
"""

def extract_features_from_tif_path(path, compute_indices=True):
    import numpy as np
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        band_means = arr.reshape(arr.shape[0], -1).mean(axis=1)
        feats = band_means.tolist()
        if compute_indices and arr.shape[0] >= 8:
            try:
                b8 = arr[7].reshape(-1).astype(np.float32)
                b4 = arr[3].reshape(-1).astype(np.float32)
                ndvi = (b8.mean() - b4.mean()) / (b8.mean() + b4.mean() + 1e-10)
            except:
                ndvi = 0.0
            try:
                b11 = arr[10].reshape(-1).astype(np.float32)
                ndbi = (b11.mean() - b8.mean()) / (b11.mean() + b8.mean() + 1e-10)
            except:
                ndbi = 0.0
            try:
                b3 = arr[2].reshape(-1).astype(np.float32)
                ndwi = (b3.mean() - b8.mean()) / (b3.mean() + b8.mean() + 1e-10)
            except:
                ndwi = 0.0
            feats += [ndvi, ndbi, ndwi]
    return np.array(feats, dtype=np.float32)

# --- new helper functions for visualization ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64
from sklearn.decomposition import PCA as SKPCA

def read_tif_array(path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    return arr

def fig_to_base64(fig, dpi=150):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return b64

def plot_spectral_signature_from_means(band_means):
    fig, ax = plt.subplots(figsize=(6,3.5))
    x = np.arange(1, len(band_means)+1)
    ax.plot(x, band_means, marker="o", linestyle="-", color="#1f77b4")
    ax.set_xlabel("Band")
    ax.set_ylabel("Mean value")
    ax.set_title("Spectral signature (band means)")
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)

def plot_band_histograms(arr, max_bands=4):
    bands = arr.shape[0]
    nb = min(bands, max_bands)
    fig, axes = plt.subplots(nb, 1, figsize=(6, 2.2*nb))
    if nb == 1:
        axes = [axes]
    for i in range(nb):
        data = arr[i].ravel()
        axes[i].hist(data, bins=60, color="#2ca02c", alpha=0.7)
        axes[i].set_ylabel(f"Band {i+1}")
        axes[i].set_xlim(np.percentile(data, 1), np.percentile(data, 99))
    fig.suptitle("Per-band histograms (trimmed view)")
    return fig_to_base64(fig)

# ---------------- HELPERS (add previews / sampling / PCA) ----------------
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
        rgb = np.dstack([r, g, b])
        p2 = np.nanpercentile(rgb, 2)
        p98 = np.nanpercentile(rgb, 98)
        rgb = (rgb - p2) / (p98 - p2 + 1e-10)
        rgb = np.clip(rgb, 0, 1)
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

def create_ndbi_base64(path):
    try:
        with rasterio.open(path) as src:
            if src.count < 8:
                return None
            b11 = src.read(11).astype(np.float32)
            b8 = src.read(8).astype(np.float32)
        ndbi = (b11 - b8) / (b11 + b8 + 1e-10)
        fig, ax = plt.subplots(figsize=(5, 5))
        cax = ax.imshow(ndbi, cmap="seismic")
        ax.axis("off")
        fig.colorbar(cax, ax=ax, fraction=0.03)
        return fig_to_base64(fig)
    except Exception:
        return None
    
def create_ndwi_base64(path):
    try:
        with rasterio.open(path) as src:
            if src.count < 8:
                return None
            b3 = src.read(3).astype(np.float32)
            b8 = src.read(8).astype(np.float32)
        ndbi = (b3 - b8) / (b3 + b8 + 1e-10)
        fig, ax = plt.subplots(figsize=(5, 5))
        cax = ax.imshow(ndbi, cmap="RdBu")
        ax.axis("off")
        fig.colorbar(cax, ax=ax, fraction=0.03)
        return fig_to_base64(fig)
    except Exception:
        return None
    
def overlay_band_histograms(samples):
    """samples: (N, bands) -> overlay histogram figure base64"""
    fig, ax = plt.subplots(figsize=(10, 6))
    bands = samples.shape[1]
    for b in range(bands):
        ax.hist(samples[:, b], bins=40, alpha=0.35, label=f"B{b+1}")
    ax.legend(ncol=3, fontsize="small")
    ax.set_title("Sampled Band Histograms")
    return fig_to_base64(fig)


def plot_pca_scatter_from_samples(samples, max_points=2000):
    """PCA 2D scatter for sampled pixels (returns base64 image)."""
    if samples is None or samples.size == 0:
        return None
    # subset to speed up
    n = samples.shape[0]
    idx = np.random.choice(n, min(n, max_points), replace=False)
    Xs = samples[idx].astype(np.float32)
    # simple standardization (per-band)
    Xs = (Xs - np.nanmean(Xs, axis=0)) / (np.nanstd(Xs, axis=0) + 1e-10)
    pca = SKPCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(np.nan_to_num(Xs))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(Xp[:, 0], Xp[:, 1], s=8, alpha=0.4, cmap="tab10")
    ax.set_title("PCA (pixels sampled) - 2D")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return fig_to_base64(fig)

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("file")
    model_choice = request.form.get("model", "rf")
    if uploaded_file is None:
        return "No file uploaded", 400

    save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
    uploaded_file.save(save_path)

    # extract features
    try:
        feats = extract_features_from_tif_path(save_path, compute_indices=True).reshape(1, -1)
    except Exception as e:
        return f"Error reading TIFF: {e}", 400

    # impute/scale consistently with training
    feats = imputer.transform(feats)
    feats_scaled = scaler.transform(feats)

    # build response
    result = {}
    try:
        if model_choice == "rf":
            pred = rf.predict(feats_scaled)[0]
            proba = rf.predict_proba(feats_scaled).max()
            result = {"method": "RandomForest", "predicted_class": label_map[pred], "confidence": round(float(proba),2)}
        else:
            
            dmat = xgb.DMatrix(feats_scaled)
    
            preds = xgb_model.predict(dmat)  # For multi-class, this returns probabilities
            pred = int(np.argmax(preds))
            proba = float(np.max(preds))  # Highest probability
            result = {"method": "XGBoosting","predicted_class": label_map[pred],"confidence": round(float(proba),2)}

    except Exception as e:
        return f"Model prediction error: {e}", 500

    # create visualizations from the uploaded file
    try:
        arr = read_tif_array(save_path)
        band_means = arr.reshape(arr.shape[0], -1).mean(axis=1)
        img_spec = plot_spectral_signature_from_means(band_means)
        img_hist = plot_band_histograms(arr, max_bands=4)

        # new previews: RGB, NDVI, sampled histograms, PCA on sampled pixels
        img_rgb = create_rgb_base64(save_path)
        img_ndvi = create_ndvi_base64(save_path)
        img_ndwi = create_ndwi_base64(save_path)
        img_ndbi = create_ndbi_base64(save_path)
        samples = sample_pixels_from_tif(save_path, n=2000)
        img_sample_hist = overlay_band_histograms(samples) if (samples is not None and samples.size>0) else None
        img_pca = plot_pca_scatter_from_samples(samples, max_points=2000) if (samples is not None and samples.size>0) else None

    except Exception as e:
        img_spec = None
        img_hist = None
        img_rgb = None
        img_ndvi = None
        img_sample_hist = None
        img_pca = None
        # keep going; show prediction even if plotting failed

    return render_template_string(HTML,
                                 result=result,
                                 img_spec=img_spec,
                                 img_hist=img_hist,
                                 img_rgb=img_rgb,
                                 img_ndvi=img_ndvi,
                                 img_ndwi=img_ndwi,
                                 img_ndbi=img_ndbi,
                                 img_sample_hist=img_sample_hist,
                                 img_pca=img_pca,
                                 filename=uploaded_file.filename)

if __name__ == "__main__":
    # keep use_reloader=False when debugging plotting issues
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

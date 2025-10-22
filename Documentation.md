# 🌍 Land Type Classification using EuroSAT Dataset

## Overview
This project uses the **EuroSAT Dataset**—based on Sentinel-2 imagery—to classify land types such as *Annual Crop, Forest, Urban, and Water Bodies* using deep learning in Python. The trained CNN model outputs predictions displayed on an interactive web dashboard.

---

## 🔧 Tech Stack
- **Language:** Python  
- **Libraries:** TensorFlow / Keras, NumPy, Pandas, OpenCV, Scikit-learn, Matplotlib  
- **Framework:** Dash (Plotly) or Flask  
- **Dataset:** [EuroSAT Dataset – Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)  
- **Storage:** Local directories (class-based) + optional SQLite metadata  

---

## 🗂 Dataset Structure
```
/dataset/
   ├── AnnualCrop/
   ├── Forest/
   ├── HerbaceousVegetation/
   ├── Highway/
   ├── Industrial/
   ├── Pasture/
   ├── PermanentCrop/
   ├── Residential/
   ├── River/
   └── SeaLake/
```
Each folder contains 64×64 RGB images (≈27,000 total).  

---

## 🚀 Workflow

1. **Data Preprocessing**
   - Load and normalize images (0–1 scale).  
   - Apply data augmentation (rotation, shift, zoom).  
   - Split into training (70%), validation (15%), and test (15%) sets.

2. **Model Training**
   - CNN or transfer learning (ResNet, MobileNet).  
   - Loss: categorical cross-entropy.  
   - Optimizer: Adam / RMSProp.  
   - Track validation accuracy and loss.

3. **Evaluation**
   - Compute accuracy, precision, recall, and F1-score.  
   - Generate confusion matrix and class-wise results.

4. **Visualization**
   - Plot accuracy/loss curves.  
   - Visualize predictions on a dashboard.  

---

## 👥 Stakeholders

| Role | Responsibility |
|------|----------------|
| Data Scientist | Model development and evaluation |
| Web Developer | Dashboard implementation |
| End User | Visualization and interpretation |
| Supervisor / Sponsor | Review and approval |

---

## 💾 Database Design
**Storage Type:** Local directories + optional SQLite metadata.

**Schema Example:**
```
TABLE Images (
  id INTEGER PRIMARY KEY,
  path TEXT,
  label TEXT
);
```

---

## 🎨 UI / UX Design

**Dashboard Features:**
- Upload or select sample image  
- Display predicted land type  
- Show confidence level and explanation  
- Plot confusion matrix and metrics  

**Color Legend**
| Class | Color |
|--------|--------|
| Sea/Lake | Blue |
| River | Cyan |
| Forest | Green |
| Pasture / Vegetation | Light Green |
| Industrial / Residential | Red |
| Highway | Gray |
| Annual Crop | Yellow |

---

## 📊 Example Results
| Metric | Value (Sample) |
|---------|----------------|
| Accuracy | 91.2% |
| Precision | 0.89 |
| Recall | 0.90 |
| F1-Score | 0.89 |

---

## 📈 Roadmap
- [ ] Add support for multispectral EuroSAT (13-band).  
- [ ] Implement Grad-CAM visual explanations.  
- [ ] Extend dashboard with geographic overlays.  
- [ ] Deploy online (Render / Heroku).

---


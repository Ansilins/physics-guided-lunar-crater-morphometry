# 🌕 Physics-Guided Lunar Crater Morphometry

A web-based application for automated detection and 3D morphometric analysis of lunar impact craters using **YOLOv8 deep learning**, **shadow geometry**, and **Digital Elevation Models (DEM)**. Deployed on Hugging Face Spaces via Docker.

---

## 🔭 Project Overview

This tool analyzes lunar surface images (GeoTIFF / PNG) to:
- **Detect craters** automatically using a trained YOLOv8 model
- **Measure crater morphology** (diameter, depth, volume) using two independent methods:
  - 📐 **Shadow-based method** — uses solar geometry and shadow length physics
  - 🗺️ **DEM-based method** — uses Digital Elevation Model elevation data
- **Compare both methods** side-by-side to validate accuracy
- **Visualize results** with interactive charts and annotated images

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 YOLOv8 Crater Detection | Custom-trained model (`best_final2.pt`) detects crater boundaries |
| 🌑 Shadow Segmentation | Separate YOLO model (`best200shadow.pt`) segments crater shadows |
| ☀️ Solar Geometry Engine | Computes sun elevation & azimuth using subsolar coordinates on the Moon |
| 📊 Morphometric Analysis | Calculates diameter, depth, and volume per crater |
| 🗺️ DEM Elevation Mapping | Reads GeoTIFF bands for elevation-based depth estimation |
| 📈 Comparison Charts | Bar charts comparing Shadow vs DEM results across all craters |
| 📥 CSV Export | Download full results as a structured CSV file |

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, Gunicorn
- **AI/ML:** Ultralytics YOLOv8
- **Computer Vision:** OpenCV, GDAL, Rasterio
- **Data & Viz:** NumPy, Pandas, Matplotlib, Seaborn
- **Geo:** Geographiclib (Moon geodesic calculations)
- **Deployment:** Docker, Hugging Face Spaces

---

## 📁 Project Structure

```
├── app.py                  # Main Flask application
├── dockerfile              # Docker container config (Hugging Face Spaces)
├── requirements.txt        # Python dependencies
├── models/
│   ├── best_final2.pt      # YOLOv8 crater detection model
│   └── best200shadow.pt    # YOLOv8 shadow segmentation model
├── templates/
│   ├── input.html          # Upload & detection UI
│   ├── results.html        # Results & charts page
│   └── about.html          # About page
└── static/
    └── output/             # Generated result images & CSV (auto-created)
```

---

## 🚀 Running Locally

### Prerequisites
- Python 3.11 (recommended for GDAL compatibility on Windows)
- GDAL installed (see Windows note below)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Ansilins/physics-guided-lunar-crater-morphometry.git
cd physics-guided-lunar-crater-morphometry

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Open your browser at: `http://127.0.0.1:5000`

> **Windows + GDAL:** GDAL requires a special install on Windows.
> Download the wheel from [cgohlke's geospatial wheels](https://github.com/cgohlke/geospatial-wheels/releases)
> and install with: `pip install GDAL-3.x.x-cp311-win_amd64.whl`

---

## 🐳 Running with Docker

```bash
docker build -t lunar-crater-app .
docker run -p 7860:7860 lunar-crater-app
```

Open: `http://localhost:7860`

---

## 🧪 How to Use

1. **Upload** a lunar image — supports GeoTIFF (`.tif`) or PNG
2. **Enter solar parameters** — subsolar latitude, longitude, and observer coordinates
3. **Run Detection** — the app detects craters and measures morphometry via both methods
4. **View Results** — compare Shadow vs DEM depth/diameter/volume on the results page
5. **Download CSV** — export the full crater dataset

---

## 📊 Output Example

- Annotated crater detection image
- Shadow segmentation overlay
- DEM elevation colormap
- Bar charts: Diameter, Depth, Volume (Shadow vs DEM)
- Combined accuracy score across all detected craters

---

## 🔬 Methodology

### Shadow-Based Depth Estimation
Uses the relationship between shadow length and sun elevation angle:

```
depth = shadow_length × tan(sun_elevation_angle)
```

Solar geometry is computed on a spherical Moon model (radius = 1,737,400 m) using geodesic calculations.

### DEM-Based Depth Estimation
Reads the elevation band from GeoTIFF input, identifies the crater rim and floor elevations, and computes depth as the difference.

Both methods detect the same craters (matched by crater ID) and their results are compared to compute an overall accuracy score.

---

## 📦 Model Info

| Model | Purpose | Format |
|---|---|---|
| `best_final2.pt` | Crater boundary detection | YOLOv8 `.pt` |
| `best200shadow.pt` | Shadow segmentation (instance masks) | YOLOv8 segmentation `.pt` |

> ⚠️ Model files are not included in this repo due to size. Place them in the `models/` directory before running.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🙋 Author

**Ansilins** — built as part of research into physics-guided deep learning for planetary science.

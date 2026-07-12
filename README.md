# 🌊 Ankara Urban Flood Risk: GeoAI-Based Spatial Analysis
### and Social Vulnerability Assessment

**Mustafa Berk Doğru**  
Ankara University · Faculty of Language, History and Geography · Department of Geography  
📧 [Contact](#contact) · 🏆 Esri Türkiye Young Scholars 2026

---

## 📌 Project Summary

This study performs machine-learning-based urban flood risk mapping in ArcGIS Pro, using **215 real citizen flood report points** from the Ankara Metropolitan Municipality (2020–2022) as the dependent variable.

It is **Turkey's first GeoAI-based urban pluvial flood risk system**.

---

## 🎯 Key Findings

| Metric | Value |
|--------|-------|
| Model | Forest-based Classification v5 |
| Spatial CV (LODO) MCC | **0.830** |
| AUC-ROC | **0.973** |
| Recall | **0.920** · F1: 0.869 |
| Independent Validation (Akyurt, n=76) | Recall: **0.934** · Precision: **1.000** |
| Population at Risk (2023) | **666,009 people** (22.1%) |
| RCP 8.5 Projection (2035) | **841,248 people** |

---

## 🗺️ Study Area

**5 Central Districts:** Keçiören · Çankaya · Altındağ · Yenimahalle · Akyurt

---

## 🔬 Methodology

### Model
- **ArcGIS Forest-based Classification v5**
- 4 variables: Elevation · Slope · Flow Accumulation · TWI
- 702 training points (292 flood points + 410 physically-criteria-based negative samples)
- 8,017 prediction points

### Original Contributions
1. **LODO Spatial CV** — Leave-One-District-Out spatial cross-validation (rare in the literature)
2. **ABB Pluvial Flood Report Data** — real urban infrastructure data, not DSİ/AFAD river flood data
3. **SHAP Explainability** — Elevation (0.325) > Distance to Streams (0.158) — independently validated against Gül (2025)
4. **Space-Time Cube + Emerging Hot Spot** — applied to temporal flood report data
5. **SVI** — first integration of a social vulnerability index in flood studies in Turkey

### ArcGIS Tools Used
- Forest-based Classification
- Optimized Hot Spot Analysis
- Space-Time Cube · Emerging Hot Spot Analysis
- Zonal Statistics As Table
- IDW Interpolation · Kernel Density
- Spatial Join · Extract Multi Values to Points

---

## 📊 Data Sources

| Data | Source |
|------|--------|
| Flood Report Points (2020–2022) | Ankara Metropolitan Municipality |
| Digital Elevation Model (12.5m) | ALOS PALSAR |
| Satellite Imagery (NDVI/NDBI) | Sentinel-2 (ESA Copernicus) |
| Population Data | TÜİK (Turkish Statistical Institute) 2022 |
| Precipitation Stations | MGM (Turkish State Meteorological Service) |
| Climate Projections | IPCC AR6 Turkey |
| Basemap | Esri Living Atlas · OpenStreetMap |

---

## 🛠️ Technical Requirements

```
ArcGIS Pro 3.x
  ├── Spatial Analyst Extension
  ├── 3D Analyst Extension
  └── Advanced License

Python 3.x (within ArcGIS Pro environment)
  ├── arcpy
  ├── numpy
  ├── pandas
  ├── matplotlib
  ├── scikit-learn
  └── shap
```

---

## 📁 File Structure

```
ankara-flood-risk-geoai/
│
├── ankara_flood_risk_FINAL_v2.py   # Main analysis code (all sections)
└── README.md                        # This file
```

### Code Structure (ankara_flood_risk_FINAL_v2.py)

| Section | Content |
|---------|---------|
| 0 | Setup and paths |
| 1 | CSV → Point layer |
| 2 | DEM Mosaic |
| 3 | Derived layers (Slope, Flow, TWI, EUC_DIST) |
| 4 | Point merging and raster value extraction |
| 5 | Negative sampling strategy |
| 6 | Training data creation |
| 7 | Data enrichment (Population + Drainage) |
| 8 | Prediction grid + Risk map |
| 9 | Forest-based Classification v5 |
| 10 | LODO Spatial CV |
| 11 | Independent validation |
| 12 | SHAP Analysis |
| 13 | Hotspot + Space-Time Cube + Emerging Hot Spot |
| 14 | Social Vulnerability Index (SVI) |
| 15 | Zonal Statistics |
| 16 | Map symbology |
| 17 | IDW Precipitation Interpolation |
| 18 | Kernel Density |
| 19 | Population Risk Analysis |
| 20 | Climate Scenario (RCP 4.5 / RCP 8.5) |

---

## 🚀 Usage

```python
# Run in ArcGIS Pro Notebook
# Each function is commented out — uncomment on first run

# Example usage:
# merge_dem()                 # First run only
# calculate_dem_derivatives() # First run only
# train_forest_model()        # Model training
# predict_risk(pred_pts)      # Prediction
# lodo_spatial_cv()           # Validation
# shap_analysis()             # Explainability
```

---

## 📫 Contact

**Mustafa Berk Doğru**  
Ankara University, Department of Geography  

🔗 GitHub: [github.com/mustafaberkdogru](https://github.com/mustafaberkdogru)

---

## 📄 License

This project was produced for academic purposes.  
May be used with attribution.

---

*Esri Türkiye Young Scholars 2026 · ArcGIS Pro · GeoAI · Urban Flood Risk*

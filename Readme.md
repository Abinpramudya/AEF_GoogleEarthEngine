
# Vegetation Fraction Prediction Pipeline

Predicts per-cell vegetation fraction (0-1) at 10m resolution using
Google Annual Earth Foundation (AEF) embeddings and Sentinel-2 imagery.
Trained on hand-labeled field data, applied to unlabeled fields in the
same region.

Developed at [Lab Name] as part of [Your Name]'s master's thesis, [Year].

---

## Requirements

- Google Earth Engine account with access to:
  - `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
  - `COPERNICUS/S2_SR_HARMONIZED`
  - Your labeled raster asset (veg_fraction_10m_v2 or equivalent)
- GEE project with sufficient compute quota
- Labeled field asset: a raster where each pixel value = vegetation
  fraction (0.0 to 1.0), 10m resolution, EPSG:32632

---

## How to Run

**Step 1 — Preprocessing**

Open `gee/1_preprocessing.js` in the GEE code editor.

This script loads the labeled raster, AEF embeddings, and Sentinel-2
composite. It builds the feature stack, computes spectral indices,
assigns fold IDs to each sample, and prints sample statistics to the
console. Run this first and verify the following in the console before
continuing:

- Total samples is 135 (or your expected count)
- veg_fraction min/max/mean look reasonable (expected mean ~0.80)
- No null/missing band errors in the console

**Step 2 — Training**

Open `gee/2_training.js` in the GEE code editor.

This script trains the Random Forest model on the preprocessed samples
and runs 5-fold cross-validation. Check the following in the console
before continuing:

- Per-fold RMSE values are consistent (no single fold wildly different)
- CV RMSE is in the expected range (~0.077)
- CV MAE is lower than CV RMSE (confirms no catastrophic outliers)
- Feature importance is printed — BSI and NBR should rank highest

**Step 3 — Prediction and Export**

Open `gee/3_testing.js` in the GEE code editor.

This script takes the trained model and applies it to a new unlabeled
field. Before running, update the field boundary coordinates at the
top of the script with your target field polygon. After running:

- Inspect the prediction layer on the map — it should show spatial
  variation matching the satellite texture
- Inspect the residual layer if a ground truth label is available
  for the target field
- Run the export task from the Tasks tab to save the GeoTIFF to
  Google Drive

---

## Repository Structure


---

## Model Summary

| Property            | Value                                      |
|---------------------|--------------------------------------------|
| Algorithm           | Random Forest regression                   |
| Number of trees     | 150                                        |
| Bag fraction        | 0.8                                        |
| Variables per split | 4 (sqrt of 15 input features)              |
| Input features      | 15 bands (see Feature Set below)           |
| Training samples    | 135 labeled pixels                         |
| Validation          | 5-fold cross-validation                    |
| CV RMSE             | 0.0774                                     |
| CV MAE              | 0.0612                                     |

---

## Feature Set (Top 15 Bands)

Selected by feature importance from an initial 46-band model.

**Sentinel-2 indices (4):**

| Band | Description                       | Notes                                      |
|------|-----------------------------------|--------------------------------------------|
| BSI  | Bare Soil Index                   | Top feature — separates bare soil from veg |
| NBR  | Normalized Burn Ratio             | Sensitive to canopy density and structure  |
| SAVI | Soil-Adjusted Vegetation Index    | Corrects bare soil background effect       |
| NDWI | Normalized Difference Water Index | Distinguishes dry vs healthy vegetation    |

**Sentinel-2 raw bands (4):**

| Band | Wavelength | Description                            |
|------|------------|----------------------------------------|
| B8   | NIR        | Near-infrared — core vegetation signal |
| B3   | Green      | Green reflectance                      |
| B4   | Red        | Red reflectance                        |
| B2   | Blue       | Blue reflectance                       |

**AEF year-band combinations (7):**

| Band     | Year |
|----------|------|
| A41_2023 | 2023 |
| A02_2025 | 2025 |
| A17_2023 | 2023 |
| A37_2023 | 2023 |
| A59_2023 | 2023 |
| A59_2024 | 2024 |
| A44_2023 | 2023 |

Note: NDVI ranked poorly (1.1% importance) due to saturation in
high-vegetation areas. BSI and NBR are more discriminating for
semi-arid shrubland where distinguishing bare soil from vegetation
is the primary challenge.

---

## Transfer to New Fields

The model generalizes based on spectral and structural similarity
to the training field. Summary:

| Field type                                          | Expected result        |
|-----------------------------------------------------|------------------------|
| Same region, same shrubland type                    | Good (RMSE ~0.08-0.10) |
| Same vegetation, different spatial structure        | Partial failure        |
| Different vegetation type (forest, crops, bare rock)| Full failure           |

For new field types, collect at least 20 labeled cells and retrain.
Retraining cost in GEE is approximately 10-30 seconds.

---

## Known Limitations

- Model weights are not persistent in GEE. The model retrains from
  samples on every script run. To save weights externally, export
  training samples as CSV and retrain in Python using scikit-learn.
- 135 samples is the practical ceiling for this approach. Estimated
  RMSE with additional data: ~0.060 at 300 samples, ~0.050 at 500.
- The S2 composite covers May 2024 - September 2025 (growing season).
  Predictions outside this temporal window may degrade.
- AEF coverage should be verified for new regions before applying
  the pipeline outside Corsica.

---

## RMSE Progression

| Stage                             | CV RMSE |
|-----------------------------------|---------|
| RF, 10 AEF bands, 2025 only       | 0.0989  |
| RF, 30 AEF bands, 2023-2025       | 0.0989  |
| GBT, 10 AEF bands, 2025 only      | 0.0978  |
| RF, 46 bands, AEF + S2 combined   | 0.0771  |
| RF, top 15 bands (final model)    | 0.0774  |

---

## Data Collection Protocol

When labeling new fields:

- Label at the same 10m grid resolution as the prediction output
- Each cell value = fraction of cell covered by vegetation
  (0.0 = fully bare, 1.0 = fully vegetated)
- Prioritize labeling cells across the full range of values (0.0-1.0),
  not only high-vegetation cells. The current training set is biased
  toward high values (mean 0.80) which degrades performance on sparse
  or bare cells
- Use high-resolution imagery (Google Maps, Pleiades) as reference
  when labeling mixed boundary cells
- Minimum per new field type: 20 cells
- Recommended for reliable transfer: 50+ cells

---



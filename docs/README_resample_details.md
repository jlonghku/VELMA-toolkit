# Resampling Technical Details

This document provides detailed descriptions of the resampling strategies used in the **VELMA Resampling Toolkit**.

---

## DEM Resampling

- **Hydrologic Conditioning**: Pit filling, depression breaching, and flat resolution are applied.  
- **Flow Metrics**: Flow direction and accumulation computed using [PySheds](https://github.com/mdbartos/pysheds).  
- **Weighted Downscaling**:  
  - Each resampled cell is calculated as the weighted average of source elevations.  
  - Weights = flow accumulation.  
  - Zero accumulation → fallback to simple mean.  
- **Outlet Adjustment**: To preserve consistency, outlet elevation is reset to the local minimum of its neighborhood.  
- **Export**: Resampled DEM is post-processed to correct flats and exported in ASCII format.

---

## Categorical Raster Resampling

- Applies to **land cover** and **soil type** rasters.  
- **Modes**:  
  - **Majority mode** (default, `Resampling.mode` from Rasterio).  
  - **Weighted mode** (custom, user-defined weights).  

Example weight dictionary:

```python
{
  'coverSpeciesIndexMapFileName': {24: 3},
  'soilParametersIndexMapFileName': {17: 2}
}
```

This biases class **24** (land cover) and **17** (soil type) during resampling.  
If no weights are given, standard majority mode is used.

---

## Continuous Raster Resampling

- Applies to continuous variables such as **NH₄, biomass**.  
- Resampled using **average pooling** (`Resampling.average`).  
- Preserves magnitude and spatial patterns.

---

## Weather Station Grid Mapping (CSV)

- CSV files containing station coordinates are resampled by **integer division** of grid indices by the downscale factor.  
- Ensures correct alignment to the new coarser grid.

---

## Grid-Based Initialization Data (CSV)

- Applies to `initializeHistoricalData` and `initializeSpecificCells`.  
- Resampling rules:  
  - Convert original indices to downscaled indices.  
  - For each new index:  
    - **Historical time series** → merged and sorted.  
    - **Initial scalar values** → averaged.  
  - If fractional values exist, they are scaled accordingly.  

---

## Outlet Handling

- To avoid topology mismatches:  
  - All `initialReachOutlets` entries are **cleared** during resampling.  
  - Outlets can be regenerated later via downstream hydrologic tools.  

---

## Statistical Comparison and Visualization

- Option: `plot_hist=True`.  
- Generates:  
  - **Bar plots** comparing class distributions before/after resampling.  
  - **CSV tables** of category percentages.  
- Provides both **visual** and **quantitative** validation.  

---

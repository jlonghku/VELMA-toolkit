# User Manual: DEM and Input Data Resampling Toolkit

This toolkit provides utilities for resampling Digital Elevation Models (DEM) and associated ecohydrological model input data (e.g., ASC rasters, XML configuration files, CSV inputs). It is designed to support workflows such as **VELMA** model preprocessing, enabling downscaling/upscaling of inputs with consistent catchment hydrology.

---

## Features

- **DEM Resampling with Flow Accumulation**  
  Resample DEMs using accumulation-weighted selection to preserve hydrologic structure.

- **Weighted Mode Resampling**  
  Supports categorical rasters (e.g., land cover, soil maps) with optional class weighting.

- **XML-based Batch Resampling**  
  Parses and updates XML configuration files, automatically resampling referenced ASC/CSV files.

- **Distribution Comparison Plots**  
  Compare class distributions between original and resampled maps.

- **CSV Index Adjustments**  
  Handles weather station files, historical disturbance data, modification schedules, and initialization values.

- **Visualization Support**  
  Generate DEM and distribution comparison plots for validation.

---
## Technical Details

For a full explanation of DEM, categorical raster, continuous raster, and CSV resampling strategies,  
see the [Resampling Technical Details](README_resample_details.md).

---

## Installation Requirements

### Python Dependencies
- `numpy`
- `pandas`
- `matplotlib`
- `rasterio`
- `pysheds`
- `pyproj`

Install via pip:
```bash
pip install numpy pandas matplotlib rasterio pysheds pyproj
```

---

## Functions

### 1. `resample_dem_with_acc`
**Description:**  
Resample a DEM with flow accumulation weighting to preserve hydrologic realism.

**Arguments:**
- `input_asc (str)` – Input DEM (ASC format).  
- `resample_asc (str)` – Output resampled DEM (ASC).  
- `outx, outy (int)` – Grid dimensions.  
- `crs (str)` – Coordinate reference system (default: EPSG:4326).  
- `downscale_factor (int)` – Factor for resampling (e.g., 2 = half resolution).  
- `plot_dem (bool)` – Plot catchments and accumulation map.  
- `output_dirs (dict)` – Output directories for plots.  

**Output:**  
Saves resampled DEM and returns original/resampled catchments.

---

### 2. `plot_distribution_comparison`
**Description:**  
Compare value distributions between raw and resampled categorical rasters.

**Arguments:**
- `raw, data (ndarray)` – Original and resampled arrays.  
- `masks (tuple)` – Optional masks for comparison.  
- `output_dirs (dict)` – Output directory for plots/CSV.  
- `title (str)` – Plot title.  

---

### 3. `resample_with_weighted_mode`
**Description:**  
Resample categorical rasters using weighted mode (e.g., to bias rare land-cover classes).

**Arguments:**
- `data (ndarray)` – Input array.  
- `downscale_factor (int)` – Scaling factor.  
- `weight_map (dict)` – Optional weights for categories.  

**Output:**  
Resampled categorical array.

---

### 4. `resample_xml`
**Description:**  
Central function for resampling all ASC/CSV references in an XML configuration file.

**Arguments:**
- `xml_path (str)` – Path to XML file.  
- `output_folder (str)` – Folder to save outputs (`asc`, `csv`, `xmls`, `png`).  
- `downscale_factor (int)` – Resampling factor.  
- `crs (str)` – Coordinate system.  
- `plot_dem (bool)` – Plot DEM and catchments.  
- `overwrite (bool)` – Overwrite existing files.  
- `plot_hist (bool)` – Generate distribution comparison plots.  
- `weights (dict)` – Optional weights for categorical rasters.  
- `change_disturbance_fraction (bool)` – Adjust harvest/disturbance fractions.  
- `num_processors (int)` – Number of processors for catchment subdivision.
- `num_subbasins (int)` – Number of subbasins for catchment subdivision.
- `plot_subdivide (bool)` – If True, plot the subdivided catchments.

**Behavior:**
- Updates DEM, land cover, soil, and filter maps.  
- Resamples weather station and initialization CSVs.  
- Removes `initialReachOutlets` (can be regenerated separately).  
- Updates grid dimensions (`outx`, `outy`).  

---

## Example Usage

```python
if __name__ == "__main__": 
    weights = {
        'coverSpeciesIndexMapFileName': {24: 3},  # Weight land cover class 24
        'soilParametersIndexMapFileName': {17: 2} # Weight soil class 17
    }
    xml_file = 'Big_Beef/XML/1.xml'
    resample_xml(
        xml_file,
        'resampled',
        downscale_factor=5,
        num_processors=8, 
        num_subbasins=50,
        plot_dem=True,
        overwrite=True,
        plot_hist=True,
        weights=weights,
        change_disturbance_fraction=False       
    )
```

---

## Output Structure

After running `resample_xml`, outputs are organized into:

```
<base_path>/<output_folder>/
    ├── asc/     # Resampled ASC files
    ├── csv/     # Resampled CSV files
    ├── xmls/    # Updated XML files
    ├── png/     # DEM & distribution plots
```

---

## Notes & Warnings

- Large downscale factors may push catchments near DEM edges, causing **Index -1 errors** in VELMA.  
- Check and adjust `ReachMap` manually if flow paths are broken.  
- For categorical rasters, provide **weights** to prevent minority class loss.  
- Use `plot_hist=True` to verify class proportions before/after resampling.

---

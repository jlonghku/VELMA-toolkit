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

### 3. `resample_with_weights`

**Description**  
Resample rasters using either average or mode.  
Automatically switches to weighted average or weighted mode if `acc` (cell weights) or `class_weight_map` (category weights) is provided.

**Arguments**
- `src_or_data (DatasetReader or ndarray)` – Raster source or array.  
- `band (int)` – Band index if using a raster source.  
- `downscale_factor (int)` – Scaling factor.  
- `method (str)` – `"average"` for continuous data or `"mode"` for categorical data.  
- `acc (ndarray, optional)` – Cell-level weights for weighted average or mode.  
- `class_weight_map (dict, optional)` – Category-level weights for weighted mode.  
- `nodata (float, optional)` – No-data value to ignore.

**Output**  
Downscaled array using plain or weighted average/mode depending on inputs.

---

### 4. `resample_xml`

**Description**  
Central routine to downscale all ASC/CSV references in a VELMA-style XML, update paths/shape, and write new XML/rasters.

**Arguments**  
- `xml_path (str)` – Path to input XML.  
- `output_folder (str)` – Output subfolder (`asc/`, `csv/`, `xmls/`, `png/`).  
- `downscale_factor (int)` – Resampling factor (>1 to downscale).  
- `crs (str)` – Target CRS (e.g., `"EPSG:26910"`).  
- `plot_dem (bool)` – Plot DEM and derived products.  
- `overwrite (bool)` – Overwrite existing outputs.  
- `plot_hist (bool)` – Plot category distributions before/after.  
- `weights (dict)` – Optional class weights for categorical rasters (`{elem.tag: {class: weight}}`).  
- `change_disturbance_fraction (bool)` – Scale disturbance/harvest fractions by cell-area change.  
- `num_processors (int)` – Processes for catchment subdivision.  
- `num_subbasins (int)` – Target number of subbasins.  
- `plot_subdivide (bool)` – Plot subdivided catchments.  
- `method (str)` – `"hydro-aware"` or `"hydro-aware-all"`; if `-all`, passes `acc` to weighted resampling.

**Behavior**  
- Resamples DEM, generates masks/`acc`, and subdivides catchments; writes resampled DEM (`asc/`) and figures (`png/`).  
- Resamples categorical rasters with **mode** (optionally class-weighted), continuous rasters with **average** (optionally cell-weighted if `hydro-aware-all`).  
- Resamples weather station and initialization CSVs; rewrites indices under the coarser grid.  
- Updates grid dims (`outx`, `outy`, `cellX`, `cellY`), output data roots, and all file paths to resampled versions.  
- Updates `initialReachOutlets` with new outlets derived from the subdivided DEM.  
- Writes two XMLs: alongside the input and under `xmls/` in the output tree.  
- Warns if the new domain approaches DEM edges (may break flow paths).

**Output**  
- Path to the resampled XML.  
- Side effects: new `asc/`, `csv/`, `xmls/`, `png/` files under `output_folder`.
 

---

## Example Usage

```python
if __name__ == "__main__":
    # optional class weights for categorical maps
    weights = {
        'coverSpeciesIndexMapFileName': {24: 3},   # boost land-cover class 24
        'soilParametersIndexMapFileName': {17: 2}  # boost soil class 17
    }

    xml_file = 'Big_Beef/XML/1.xml'
    resample_xml(
        xml_file,
        'resampled',
        downscale_factor=5,
        num_processors=8,
        num_subbasins=50,
        plot_dem=True,
        plot_subdivide=True,
        overwrite=True,
        plot_hist=True,
        weights=weights,
        change_disturbance_fraction=False,
        method='hydro-aware-all'   # use cell-weighted resampling for all rasters
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

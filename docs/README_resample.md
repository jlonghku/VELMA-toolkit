# VELMA / Ecohydro Input Resampling Toolkit
This toolkit provides utilities for resampling Digital Elevation Models (DEM) and associated ecohydrological model input data (e.g., ASC rasters, XML configuration files, CSV inputs). It is designed to support workflows such as **VELMA** model preprocessing, enabling coarsening of inputs with consistent catchment hydrology and ecological functions.

---

## Technical Details

For a full explanation of DEM, categorical raster, continuous raster, and CSV resampling strategies, including distribution matching and Hellinger-based stopping criteria,  
see the [Resampling Technical Details](README_resample_details.md).

---

## Key features

- **DEM resampling**  
  Downscale DEM by blocks (factor = `downscale_factor`) or by Whitebox
  resampling tools; auto pit/depression fill; add 1-cell NODATA rim to stop
  edge leakage; snap outlet to local max accumulation.  
  Method controlled by:
  - `dem_method='hydro-aware'` → flow-accumulation weighted block mean;
  - `dem_method='mean'` → plain block mean;
  - `dem_method='nearest'` → Whitebox nearest-neighbour resampling;
  - `dem_method='bilinear'` → Whitebox bilinear resampling;
  - `dem_method='burn-streams'` → bilinear resampling plus stream burning;
  - `dem_method='burn-breach'` → least-cost breach plus stream burning.  
  `cubic` and `feature-preserving` are not supported in the modified version.  
  Output DEM is written under `<base>/<output_folder>/asc/` and XML will be updated.

- **Flexible categorical raster resampling**  
  For `coverSpeciesIndexMapFileName`, `soilParametersIndexMapFileName`,
  `filterMapFullName` etc., you can choose:
  - `class_method='majority'` → plain block mode, optionally with user-given weights
    per class (e.g. to protect rare land cover);
  - `class_method='hydro-aware'` → mode but weighted by DEM accumulation;
  - `class_method='auto-weight'` → iterate to match original class percentages,
    using both an absolute percentage tolerance (`tol`) and an optional
    Hellinger distance threshold (`hellinger_tol`);
  - `class_method='auto-reassign'` → reassign a small number of blocks to match
    the original global class distribution, with early stopping when the
    Hellinger distance falls below `hellinger_tol`.  
  `class_method='mode'` is accepted as a backward-compatible alias for
  `majority`, but `majority` is the default and recommended name.
  Hydro-aware categorical mode falls back to unweighted majority when all
  accumulation weights in a block are zero.
  These options help avoid losing small-area but hydrologically important classes.

- **Continuous raster resampling with masks**  
  For all other ASC rasters (biomass, NH₄, forcing grids, etc.), use
  `avg_method`:
  - `avg_method='mean'` → simple mean;
  - `avg_method='hydro-aware'` → accumulation-weighted average;
  - `avg_method='landcover-aware'` → average weighted by land-cover agree mask;
  - `avg_method='soil-aware'` → average weighted by soil agree mask.  
  This lets you “project” continuous variables to the class pattern after
  coarsening.

- **XML-wide resampling**  
  Central API:  
  ```python
  resample_xml(
      xml_path,
      output_folder="resampled",
      downscale_factor=5,
      crs="EPSG:26910",
      plot_dem=True,
      overwrite=True,
      plot_hist=True,
      dem_method="hydro-aware",
      class_method="majority",
      avg_method="mean",
      num_processors=8,
      num_subbasins=1,
      plot_subdivide=False,
  )
  ```  
  It will:
  1. parse `<inputDataLocationRootName>` + `<inputDataLocationDirName>` to get
     the base path;
  2. resample DEM first and compute `acc`;
  3. resample all other ASC accordingly;
  4. resample CSVs that store grid indices;
  5. update grid size fields (`outx`, `outy`, `cellX`, `cellY`);
  6. clear & regenerate `initialReachOutlets` from subdivided catchments;
  7. write **two** updated XMLs:
     - beside original: `xxx_resampled_<factor>_<dem>-<class>-<avg>.xml`
     - under `<base>/<output_folder>/xmls/`  
     so you can track different runs.

- **Start-state folder resampling**  
  If XML contains a folder-style start state such as  
  `<setStartStateSpatialDataLocationName>init_state/2020</setStartStateSpatialDataLocationName>`  
  and that folder really exists under the base path, the toolkit will walk
  through it, resample every `.asc` inside **with the same strategy** and copy
  non-ASC files. The XML field will be rewritten to the new folder under
  `.../asc/<oldname>_resampled_...`.  
  This is important for long VELMA projects that store multiple initial layers
  in a directory.

- **Catchment subdivision (optional)**  
  After DEM resampling, you can call `subdivide_catchments(...)` to produce
  sub-basins for parallel processing or for checking connectivity.
  This is controlled by `num_processors`, `num_subbasins` and `plot_subdivide`.

- **Output redirection for model results**  
  `<initializeOutputDataLocationRoot>` will be suffixed with
  `/<downscale>_<dem>-<class>-<avg>` so that you can run multiple coarsened
  projects side by side.

---

## Supported input types

1. **DEM (ASC)** – hydrologically conditioned, downscaled, rimmed.  
2. **Categorical ASC** – land cover / soil / filter maps with multiple strategies.  
3. **Continuous ASC** – biomass, nitrogen, disturbance maps, etc.  
4. **CSV with grid indices** –  
   - `weatherLocationsDataFileName` → (col // factor, row // factor), path normalized for Windows;  
   - `initializeHistoricalData` → re-index + merge by new index;  
   - `modificationsDataFileName` → re-index + pick majority record per (t1, t2, new_idx);  
   - `initializeSpecificCells` → re-index + average values in the same new cell.  
   Flat-index CSV remapping uses `ceil(original_col_count / downscale_factor)`
   for the new column count, matching the raster output shape when the source
   width is not evenly divisible by the downscale factor.  
   This part is what lets you keep VELMA’s “flat index” inputs after coarsening.

---

## CLI / Example

```python
if __name__ == "__main__":
    label = "Big_Beef"
    xml_file = f"{label}/XML/1.xml"

    resample_xml(
        xml_file,
        "resampled",
        downscale_factor=2,
        crs="EPSG:26910",
        plot_dem=True,
        plot_subdivide=True,
        overwrite=True,
        plot_hist=True,
        dem_method="hydro-aware",
        class_method="majority",
        avg_method="mean",
        num_processors=8,
        num_subbasins=4,
    )
```

---

## Notes / Pitfalls

- Very large `downscale_factor` can push the watershed too close to DEM edge,
  which may break the routed river network and VELMA can report `'Index -1'`
  or similar errors. Check and reduce the factor if this happens.  
- Make sure paths in XML are **relative to** `<inputDataLocationRootName>/<inputDataLocationDirName>`,
  and that folder-style inputs actually exist, otherwise `os.path.isdir(...)`
  on Windows will fail.  
- For categorical maps with important small patches, prefer
  `class_method='majority'` with a weight map or `class_method='auto-reassign'`.  
- `auto-weight` keeps the best Hellinger-distance result seen across iterations
  so oscillating weight updates do not return a worse final map.  
- `auto-reassign` converts target percentages to integer coarse-cell counts
  using a largest-remainder rule, so small fractional improvements are not
  dropped by flooring. It can consider any ranked class present in a block, not
  only ranks 2-4.  
- When using `auto-weight` or `auto-reassign`, `hellinger_tol` provides an early
  stopping threshold. For `auto-weight`, `tol` is also used as a maximum class
  percentage error stopping threshold.
- Use `plot_hist=True` to visually check class percentages before/after.

---

## Installation

```bash
pip install numpy pandas matplotlib rasterio pysheds pyproj whitebox-workflows
```

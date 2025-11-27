# Resampling Technical Details

This document provides detailed descriptions of the resampling strategies used in the **VELMA Resampling Toolkit**.

---

## 1. DEM resampling

**Goal**: keep drainage structure and outlet position when coarsening DEM.

1. **Pre-conditioning**  
   - load original DEM with PySheds;  
   - pit filling → depression filling → flat resolution;  
   - compute flow direction + flow accumulation (`acc0`).  
   This step provides the accumulation array used by later steps.

2. **Block downscaling**  
   - new shape = `ceil(rows/f) × ceil(cols/f)`  
   - for each source block:  
     - if `dem_method='hydro-aware'` and block accumulation sum > 0 → weighted
       mean by `acc0`;  
     - else → simple mean.  
   - outlet cell is converted from `(outx, outy)` to `(outx//f, outy//f)`.  
   - then we ensure the outlet is the local minimum in 3×3, otherwise lower it
     slightly (`-0.01`).  
   Rationale: coarse DEM must still drain to the target outlet.

3. **Rim and re-routing**  
   - a 1-cell NODATA rim is added around the DEM to stop leakage across tile
     boundary;  
   - flow direction & accumulation are recomputed on the rimmed raster;  
   - outlet is snapped to the local max accumulation in a 3×3 window.  
   The final catchment is extracted on this recomputed network.

4. **Export**  
   DEM is written as ASCII with updated affine (scaled by factor). Plot is
   optional and saved under `.../png/` when `plot_dem=True`.  
   The function returns:  
   - original column count (for CSV index remapping);  
   - original vs. new catchment masks;  
   - original accumulation array.  
   These are later reused by XML-level functions.

---

## 2. Categorical raster resampling

Applies to land cover, soil, filter maps and other “index” rasters
specified in the XML. Current code supports **four** strategies:

1. **`class_method='majority'`**  
   - base mode: per-block majority;  
   - if user passes `weights={tag: {class_id: weight, ...}}`, the per-class weight
     is multiplied in voting;  
   - ties are randomly broken.  
   Use this if you know which classes should be protected.

2. **`class_method='hydro-aware'`**  
   - same as majority, but each pixel is also multiplied by the DEM accumulation
     of that pixel;  
   - idea: pixels that actually receive more upstream flow should dominate the
     coarsened class.  
   Works well for riparian or channel-adjacent classes.

3. **`class_method='auto-weight'`**  
   - run multiple iterations;  
   - each iteration compares the global class percentage of the resampled map
     with the original map (before coarsening);  
   - update class weights to shrink the difference, clipped to a reasonable
     range;  
   - stopping criteria combine:
     - a maximum allowed absolute percentage error (`tol`), and  
     - an optional Hellinger distance threshold (`hellinger_tol`) between the
       original and resampled class distributions.  
   Use this when you don’t know which class to weight, you just want to keep
   the overall histogram close to the original without over-tuning.

4. **`class_method='auto-reassign'`**  
   - first do a plain per-block assignment but record top-k candidates in each
     block;  
   - compute the global histogram of this initial assignment and compare it with
     the original histogram; if the Hellinger distance is already below
     `hellinger_tol` (when provided), no reassignment is performed;  
   - otherwise, look at the global histogram and selectively reassign blocks
     that have the target class as 2nd/3rd candidate;  
   - reassign only a small number to hit the target percentages, again without
     forcing an exact match that would break spatial coherence.  
   Use this when class distribution is critical (e.g. scenario modeling), but
   you still want a soft stopping rule based on distribution similarity.

All categorical methods can output an **agree mask**, which is later used by
continuous rasters (`landcover-aware` / `soil-aware` average).  
If `plot_hist=True`, the toolkit will generate a stacked bar chart + CSV
comparing original vs. resampled class percentages for each categorical map.
This is especially recommended when using `class_method='auto-weight'` or
`class_method='auto-reassign'` with `hellinger_tol`.

### Hellinger distance (qualitative)

The Hellinger distance is used as a bounded measure of similarity between two
discrete class distributions (original vs. resampled).  
- values close to 0 → very similar histograms;  
- larger values → more discrepancy.  
Using `hellinger_tol` as a stopping criterion avoids chasing tiny percentage
differences when the overall distribution is already close enough.

---

## 3. Continuous raster resampling

For all other ASC files (not DEM, not the listed categorical tags), the code
calls `resample_with_weights(..., method="average")` and chooses the averaging
strategy by `avg_method`:

- `avg_method='mean'`  
  → simple mean of valid pixels in the block.

- `avg_method='hydro-aware'`  
  → weighted average with DEM accumulation (needs `acc` from DEM stage).

- `avg_method='landcover-aware'`  
  → weighted average with the land-cover agree mask: pixels that belong to the
  chosen land-cover of that block get larger weights.

- `avg_method='soil-aware'`  
  → same idea, but using soil agree mask.  
  This is useful when you have per-landcover biomass / soilwater / nitrogen
  pools that must stay locked to the chosen discrete class after coarsening.

---

## 4. CSV / table resampling

Several VELMA inputs use **flattened grid index** (`row * ncol + col`). After
coarsening, both row and `ncol` change → the index must be recomputed.

1. **`initializeHistoricalData`**  
   - read each line → old_index → (row, col);  
   - compute new (row, col) by integer division by `downscale_factor`;  
   - recompute `new_index = new_row * new_colmax + new_col`;  
   - if multiple lines become the same `new_index`, merge their date/value pairs
     and sort;  
   - write back to CSV.  
   Optional: if `change_disturbance_fraction=True`, related harvest fractions
   in the same XML branch will be scaled by area (`/factor/factor`).

2. **`modificationsDataFileName`**  
   - similar re-indexing, but grouped by `(time1, time2, new_index)`;  
   - if there are multiple rows for the same key, pick the most frequent one.

3. **`initializeSpecificCells`**  
   - re-index;  
   - if multiple values fall into the same new cell, take mean.

4. **`weatherLocationsDataFileName`**  
   - x and y columns are integer-divided by `downscale_factor`;  
   - file path column is normalized (`os.path.normpath`) so it works on Windows
     too.  
   This prevents the Windows-style `ntpath.join` issue when XML gives a folder
   instead of file.

---

## 5. XML updates

During one pass over the XML tree:

- raster/CSV paths are rewritten to **relative paths** under
  `<base>/<output_folder>/(asc|csv|xmls)` so the project stays portable;  
- numeric grid fields are scaled down;  
- `initializeOutputDataLocationRoot` is suffixed;  
- `initialReachOutlets` is replaced by the output from `subdivide_catchments`
  (or cleared if subdivision not available).  

Two XMLs are written: one next to the original XML, one in the output folder,
both with the resampling suffix.

---

## 6. Visualization

If `plot_dem=True`, a DEM + catchment + capped accumulation map is saved.  
If `plot_hist=True`, for every categorical map a “before vs after” percentage
plot and CSV are saved. This is recommended whenever you use
`class_method='auto-*'`, especially with a non-default `hellinger_tol`, so you
can visually confirm that the class histogram has converged to a reasonable
match without overcorrection.

---

## 7. Known limitations

- Very large factors (≥10) can break river topology → check the resampled
  `ReachMap` manually.  
- Directory-style start states must actually exist under the base path.  
- Auto-reassign works best when the number of blocks is not too small.  
- Hellinger-based stopping (`hellinger_tol`) is a global histogram measure; it
  does not guarantee local (per-catchment or per-subregion) distribution
  matching, so local diagnostics may still be needed for sensitive applications.

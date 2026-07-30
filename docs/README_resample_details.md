# Resampling Technical Details

This document provides detailed descriptions of the resampling strategies used in the **VELMA Resampling Toolkit**.

---

## 0. Notation and implementation scope

Let \(G_f\) and \(G_c\) denote the fine and coarse grids. For a coarse cell
\(C\in G_c\), let \(F(C)\) be the set of valid fine cells mapped to \(C\).
For fine cell \(i\):

- \(x_i\) is a continuous value;
- \(c_i\in K\) is a categorical class;
- \(\alpha_i\) is cell area;
- \(a_i\) is flow accumulation (upstream contributing area or, on a
  unit-area raster, the accumulated cell count).

The Supporting Information defines flow accumulation recursively on the
directed flow network as

$$
a_i=\alpha_i+\sum_{j\in U(i)}a_j,
$$

where \(U(i)\) contains cells draining directly into \(i\). With equal cell
areas normalized to one, this becomes
\(a_i=1+\sum_{j\in U(i)}a_j\).

The formulas below separate the **current implementation** from useful
mathematical extensions. The code operates on regular rasters, uses equal-area
cell counts, and uses \(g(a)=a\) for hydrological weights. In the loop-based
low-level paths, valid cells are controlled by the `nodata` argument: floating
NaNs are masked automatically, but a numeric NODATA sentinel must be passed
explicitly. `resample_xml` currently does not forward `src.nodata` to those
paths, so convert sentinel values to NaN or call `resample_with_weights`
directly with `nodata=...` when this matters. Area-weighted formulas and
\(g(a)=a^\gamma\) are generalizations for heterogeneous grids; they are not
currently exposed by `resample_xml`.

---

## 1. DEM resampling

**Goal**: keep drainage structure and outlet position when coarsening DEM.

1. **Pre-conditioning**  
   - load original DEM with PySheds;  
   - pit filling → depression filling → flat resolution;  
   - compute flow direction + flow accumulation (`acc0`).  
   This step provides the accumulation array used by later steps. In the
   current regular-grid implementation, `acc0` is an accumulated upstream cell
   count and therefore represents contributing area up to a constant cell-area
   factor.

2. **DEM method selection**  
   The modified code supports six DEM methods:
   - `hydro-aware`: flow-accumulation weighted block mean;
   - `mean`: plain block mean;
   - `nearest`: Whitebox nearest-neighbour resampling;
   - `bilinear`: Whitebox bilinear resampling;
   - `burn-streams`: bilinear resampling followed by stream burning;
   - `burn-breach`: least-cost breaching followed by stream burning.  

3. **Block downscaling methods (`hydro-aware`, `mean`)**  
   - new shape = `ceil(rows/f) × ceil(cols/f)`  
   - for each source block:  
     - if `dem_method='hydro-aware'` and block accumulation sum > 0 → weighted
       mean by `acc0`,
       $$
       \widetilde{z}_C=
       \frac{\sum_{i\in F(C)}a_i z_i}
            {\sum_{i\in F(C)}a_i};
       $$
     - else → simple mean.  
   - outlet cell is converted from `(outx, outy)` to `(outx//f, outy//f)`.  
   - then we ensure the outlet is the local minimum in 3×3, otherwise lower it
     slightly (`-0.01`).  
   Rationale: coarse DEM must still drain to the target outlet.

4. **Whitebox methods (`nearest`, `bilinear`, `burn-streams`, `burn-breach`)**  
   - Whitebox resamples the DEM to `original_cell_size * downscale_factor`;  
   - `nearest` uses nearest-neighbour resampling;  
   - `bilinear`, `burn-streams`, and `burn-breach` use bilinear resampling;  
   - `burn-breach` first applies least-cost breaching before stream extraction;  
   - `burn-streams` and `burn-breach` derive streams from D8 flow accumulation
     and burn them into the resampled DEM.  
   Burn parameters depend on `downscale_factor`, with larger factors using
   larger stream thresholds and smaller burn gradients.

5. **Rim and re-routing**  
   - a 1-cell NODATA rim is added around the DEM to stop leakage across tile
     boundary;  
   - flow direction & accumulation are recomputed on the rimmed raster;  
   - outlet is snapped to the local max accumulation in a 3×3 window.  
   The final catchment is extracted on this recomputed network.

6. **Export**  
   DEM is written as ASCII with updated affine (scaled by factor). Plot is
   optional and saved under `.../png/` when `plot_dem=True`.  
   The function returns:  
   - original column count (for CSV index remapping);  
   - original vs. new catchment masks;  
   - original accumulation array.  
   These are later reused by XML-level functions.

The Supporting Information writes the normalized hydrological weight more
generally as

$$
\omega_i^{(h)}=
\frac{g(a_i)}{\sum_{j\in F(C)}g(a_j)},\qquad
\widetilde{z}_C=\sum_{i\in F(C)}\omega_i^{(h)}z_i,
$$

with monotonic non-decreasing \(g\), for example \(g(a)=a^\gamma\),
\(\gamma>0\). The current code corresponds to \(\gamma=1\); there is no
`gamma` argument at present.

---

## 2. Categorical raster resampling

Applies to land cover, soil, filter maps and other “index” rasters
specified in the XML. Current code supports **four** strategies:

Categorical labels are never numerically averaged. All local assignments can
be written as

$$
\widetilde{c}_C=\arg\max_{k\in K}S_C(k),\qquad
S_C(k)=\sum_{i\in F(C)}q_i\,\mathbf{1}(c_i=k),
$$

where the method determines vote weight \(q_i\).

1. **`class_method='majority'`**  
   - base mode: per-block majority,
     \(S_C(k)=N_C(k)=\sum_i\mathbf{1}(c_i=k)\);
   - if user passes `weights={tag: {class_id: weight, ...}}`, the per-class weight
     is multiplied in voting;  
   - ties are randomly broken; with the low-level default `seed=0`, results are
     reproducible.
   Use this if you know which classes should be protected.

   `class_method='mode'` is accepted as a backward-compatible alias for
   `majority`, but `majority` is the default and recommended name.

2. **`class_method='hydro-aware'`**  
   - same as majority, but each pixel is also multiplied by the DEM accumulation
     of that pixel:
     $$
     S_C^{(h)}(k)=
     \sum_{i\in F(C)}
     \frac{a_i}{\sum_{j\in F(C)}a_j}\,
     \mathbf{1}(c_i=k);
     $$
   - if all accumulation weights in a block are zero, non-finite, or otherwise
     unusable, the method falls back to unweighted majority for that block;  
   - idea: pixels that actually receive more upstream flow should dominate the
     coarsened class.  
   Works well for riparian or channel-adjacent classes.

3. **`class_method='auto-weight'`**  
   - compute local class proportions
     \(p_C(k)=N_C(k)/|F(C)|\), then assign by
     $$
     \widetilde{c}_C=
     \arg\max_k\left[w_kp_C(k)\right];
     $$
   - run multiple iterations, updating the positive global class weights
     \(w_k\);
   - each iteration compares the global class percentage of the resampled map
     with the original map (before coarsening);  
   - update class weights to shrink the difference, clipped to a reasonable
     range;  
   - keep the best map seen across iterations using Hellinger distance first and
     maximum class-percentage error second, so an oscillating sequence does not
     return a worse final iteration;  
   - stopping criteria combine:
     - a maximum allowed absolute percentage error (`tol`), and  
     - an optional Hellinger distance threshold (`hellinger_tol`) between the
       original and resampled class distributions.  
   Use this when you don’t know which class to weight, you just want to keep
   the overall histogram close to the original without over-tuning.

4. **`class_method='auto-reassign'`**  
   - first do a plain per-block assignment and record all candidate classes in
     each block, ordered by within-block frequency;  
   - compute the global histogram of this initial assignment and compare it with
     the original histogram; if the Hellinger distance is already below
     `hellinger_tol` (when provided), no reassignment is performed;  
   - convert target percentages into integer coarse-cell counts with a
     largest-remainder rule, so small maps do not lose useful fractional target
     counts through flooring;  
   - otherwise, look at the global histogram and selectively reassign surplus
     classes to underrepresented classes;  
   - a target class can be considered at any rank within a block, not only ranks
     2-4; candidates are preferred by lower rank, stronger within-block support,
     and larger surplus in the class being replaced.  
   Use this when class distribution is critical (e.g. scenario modeling), but
   you still want a soft stopping rule based on distribution similarity.

All categorical methods can output an **agree mask**, which is later used by
continuous rasters (`landcover-aware` / `soil-aware` average).  
If `plot_hist=True`, the toolkit will generate a stacked bar chart + CSV
comparing original vs. resampled class percentages for each categorical map.
This is especially recommended when using `class_method='auto-weight'` or
`class_method='auto-reassign'` with `hellinger_tol`.

### Watershed-scale distributions and Hellinger distance

For equal-area cells, the fine- and coarse-grid class distributions are

$$
p_{\mathrm{orig}}(k)=
\frac{1}{|G_f|}\sum_{i\in G_f}\mathbf{1}(c_i=k),\qquad
p_{\mathrm{coarse}}(k)=
\frac{1}{|G_c|}\sum_{C\in G_c}\mathbf{1}(\widetilde{c}_C=k).
$$

For heterogeneous cells, the Supporting Information replaces the cell counts
with area weights \(\alpha_i\) and
\(\beta_C=\sum_{i\in F(C)}\alpha_i\). This area-weighted distribution is a
mathematical extension and is not used by the current raster implementation.

The bounded Hellinger distance is

$$
H(p,q)=\frac{1}{\sqrt{2}}
\left[\sum_{k\in K}\left(\sqrt{p(k)}-\sqrt{q(k)}\right)^2\right]^{1/2},
\qquad 0\le H\le 1.
$$

The toolkit evaluates
\(H(p_{\mathrm{coarse}},p_{\mathrm{orig}})\). It is a global diagnostic or
convergence criterion, not a separate coarsening method:

- `auto-weight` stops when `H <= hellinger_tol` or the maximum absolute
  class-percentage error is `<= tol`, and returns the best iteration ranked by
  Hellinger distance and then maximum class error;
- `auto-reassign` checks `H <= hellinger_tol` after the initial majority
  assignment. If not satisfied, it converts the target distribution to integer
  coarse-cell counts and performs ranked surplus-to-deficit reassignments.

`resample_with_weights` exposes `hellinger_tol`, `tol`, `max_iter`, `clip`, and
`seed`. Their current defaults are `1e-3`, `1e-3`, `5`, `(0.25, 4.0)`, and
`0`, respectively; both tolerances are expressed as proportions, so `1e-3`
equals 0.1 percentage point. The high-level `resample_xml` API currently uses
these low-level defaults rather than forwarding them as arguments.

---

## 3. Continuous raster resampling

For all other ASC files (not DEM, not the listed categorical tags), the code
calls `resample_with_weights(..., method="average")` and chooses the averaging
strategy by `avg_method`:

$$
\widetilde{x}_C=
\frac{\sum_{i\in F(C)}q_i x_i}
     {\sum_{i\in F(C)}q_i}.
$$

- `avg_method='mean'`  
  → \(q_i=1\): simple mean of valid pixels in the block.

- `avg_method='hydro-aware'`  
  → \(q_i=a_i\): weighted average with DEM accumulation (needs `acc` from DEM
  stage).

- `avg_method='landcover-aware'`  
  → \(q_i=\mathbf{1}(l_i=\widehat{l}_C)\): average only fine cells that agree
  with the selected coarse land-cover class.

- `avg_method='soil-aware'`  
  → \(q_i=\mathbf{1}(s_i=\widehat{s}_C)\): same operation using the selected
  coarse soil class.
  This is useful when you have per-landcover biomass / soilwater / nitrogen
  pools that must stay locked to the chosen discrete class after coarsening.

For all weighted strategies, a zero weight sum falls back to the unweighted
mean of valid cells. The low-level function additionally implements
`average_strategy='acc_mask'`, with
\(q_i=a_i\,\mathbf{1}(\text{class agrees})\), but this combined strategy is not
currently selectable through the high-level `avg_method`.

For heterogeneous grids, the corresponding mean is
\(\widetilde{x}_C=\sum_i\alpha_i x_i/\sum_i\alpha_i\), and the class-aware
forms apply the same area factor within the agreeing subset. These are
generalized formulations from the Supporting Information, not current runtime
options.

---

## 4. CSV / table resampling

Several VELMA inputs use **flattened grid index** (`row * ncol + col`). After
coarsening, both row and `ncol` change → the index must be recomputed.
The modified code uses `new_colmax = ceil(original_col_count / downscale_factor)`
for flat-index remapping, matching the raster output width when the source
column count is not evenly divisible by the downscale factor.

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

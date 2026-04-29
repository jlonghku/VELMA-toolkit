import os
import rasterio
from rasterio.enums import Resampling
import xml.etree.ElementTree as ET
import pandas as pd
from pysheds.grid import Grid
import matplotlib.pyplot as plt
import numpy as np
if not hasattr(np, "in1d"):
    np.in1d = np.isin
from pysheds.sview import Raster, ViewFinder
from pyproj import Proj
from collections import defaultdict, Counter
import csv,math
from subdivide import subdivide_catchments
from whitebox_workflows import WbEnvironment

def resample_dem(input_asc, resample_asc, outx=None, outy=None, crs="EPSG:4326",
                 downscale_factor=2, plot_dem=False, output_dirs=None, method='hydro-aware'):
    """
    Downscale DEM using one of six methods:
        1. 'hydro-aware'       : accumulation-weighted block aggregation
        2. 'mean'              : plain block mean aggregation
        3. 'nearest'           : Whitebox nearest-neighbour resampling
        4. 'bilinear'          : Whitebox bilinear resampling
        5. 'burn-streams'      : stream burning
        6. 'burn-breach'       : constrained breach + stream burning

    Returns
    -------
    cols : int
    [catch_raw, catch_new] : list
    acc_orig : array-like
    """

    def _convert_tif_to_asc(src_tif, dst_asc):
        with rasterio.open(src_tif) as src:
            profile = src.profile.copy()
            profile.update(driver="AAIGrid")
            data = src.read(1)
            os.makedirs(os.path.dirname(os.path.abspath(dst_asc)), exist_ok=True)
            with rasterio.open(dst_asc, "w", **profile) as dst:
                dst.write(data, 1)

    def _get_burn_params(scale_factor):
        if scale_factor <= 2:
            return 800.0, 2.0, 2
        elif scale_factor == 3:
            return 1200.0, 1.5, 1
        elif scale_factor == 4:
            return 2000.0, 1.0, 1
        else:
            return 3000.0, 0.5, 1

    def _route_and_delineate(grid_obj, dem_raster, outx_ds, outy_ds):
        nodata_val = dem_raster.nodata if dem_raster.nodata is not None else -9999.0

        pit_filled = grid_obj.fill_pits(dem_raster)
        flooded = grid_obj.fill_depressions(pit_filled)
        dem_resolved = grid_obj.resolve_flats(flooded)

        arr = dem_resolved.copy()
        arr[[0, -1], :] = nodata_val
        arr[:, [0, -1]] = nodata_val
        rim_raster = Raster(arr, viewfinder=dem_resolved.viewfinder)
        rim_raster.nodata = nodata_val

        fdir = grid_obj.flowdir(rim_raster)
        acc_ds_local = grid_obj.accumulation(fdir)

        win = 1
        r0, r1 = max(outy_ds - win, 0), min(outy_ds + win + 1, acc_ds_local.shape[0])
        c0, c1 = max(outx_ds - win, 0), min(outx_ds + win + 1, acc_ds_local.shape[1])
        sub = acc_ds_local[r0:r1, c0:c1]
        dy, dx = np.unravel_index(np.nanargmax(sub), sub.shape)
        snap_y, snap_x = r0 + dy, c0 + dx

        catch_new_local = grid_obj.catchment(x=snap_x, y=snap_y, fdir=fdir, xytype='index')
        return rim_raster, acc_ds_local, catch_new_local, nodata_val

    # ------------------------------------------------------------------
    # 1. Original DEM preprocessing
    # ------------------------------------------------------------------
    grid = Grid.from_ascii(input_asc, crs=Proj(crs))
    orig_dem = grid.read_ascii(input_asc, crs=Proj(crs))

    pit_filled_dem = grid.fill_pits(orig_dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    dem = grid.resolve_flats(flooded_dem)

    fdir0 = grid.flowdir(dem)
    catch_raw = grid.catchment(x=outx, y=outy, fdir=fdir0, xytype='index')
    acc0 = grid.accumulation(fdir0)
    acc_orig = acc0.copy()

    rows, cols = dem.shape
    f = downscale_factor
    new_outx, new_outy = outx // f, outy // f

    acc_ds = None
    catch_new = None

    # ------------------------------------------------------------------
    # 2. Block aggregation methods
    # ------------------------------------------------------------------
    if method in ['hydro-aware', 'mean']:
        new_rows, new_cols = math.ceil(rows / f), math.ceil(cols / f)
        corrected_dem = np.zeros((new_rows, new_cols), dtype=float)

        for i in range(new_rows):
            for j in range(new_cols):
                r0, r1 = i * f, min((i + 1) * f, rows)
                c0, c1 = j * f, min((j + 1) * f, cols)

                block_dem = dem[r0:r1, c0:c1]
                if method == 'hydro-aware':
                    block_acc = acc0[r0:r1, c0:c1]
                    total_acc = float(np.sum(block_acc))
                    corrected_dem[i, j] = (
                        float(np.sum(block_dem * block_acc) / total_acc)
                        if total_acc > 0 else float(np.mean(block_dem))
                    )
                else:
                    corrected_dem[i, j] = float(np.mean(block_dem))

        # Outlet adjustment
        i0, i1 = max(new_outy - 1, 0), min(new_outy + 1, new_rows - 1)
        j0, j1 = max(new_outx - 1, 0), min(new_outx + 1, new_cols - 1)
        neigh_min = float(np.min(corrected_dem[i0:i1 + 1, j0:j1 + 1]))
        corrected_dem[new_outy, new_outx] = min(corrected_dem[new_outy, new_outx], neigh_min - 0.01)

        nodata_val = dem.nodata if dem.nodata is not None else -9999.0
        corrected_dem = np.nan_to_num(corrected_dem, nan=nodata_val)

        viewfinder = ViewFinder(
            affine=dem.affine * dem.affine.scale(f, f),
            shape=corrected_dem.shape,
            crs=dem.crs,
            nodata=nodata_val
        )
        corrected_dem_raster = Raster(corrected_dem, viewfinder=viewfinder)
        newgrid = Grid.from_raster(corrected_dem_raster)

        rim_raster, acc_ds, catch_new, nodata_val = _route_and_delineate(
            newgrid, corrected_dem_raster, new_outx, new_outy
        )
        newgrid.to_ascii(rim_raster, resample_asc, nodata=nodata_val)

    # ------------------------------------------------------------------
    # 3. Whitebox-based methods
    # ------------------------------------------------------------------
    elif method in ['nearest', 'bilinear', 'burn-streams', 'burn-breach']:
        workdir = os.path.dirname(os.path.abspath(resample_asc)) or os.getcwd()
        os.makedirs(workdir, exist_ok=True)

        wbe = WbEnvironment()
        wbe.working_directory = workdir
        wbe.verbose = False
        wbe.max_procs = -1

        dem_wb = wbe.read_raster(input_asc)
        target_cellsize = dem_wb.configs.resolution_x * f

        wb_method_map = {
            'nearest': 'nn',
            'bilinear': 'bilinear',
            'burn-streams': 'bilinear',
            'burn-breach': 'bilinear'
        }

        dem_resampled = wbe.resample(
            input_rasters=[dem_wb],
            cell_size=target_cellsize,
            method=wb_method_map[method]
        )

        tmp_resample_tif = os.path.join(workdir, "_tmp_resample.tif")
        tmp_burned_tif = os.path.join(workdir, "_tmp_burned.tif")
        wbe.write_raster(dem_resampled, tmp_resample_tif, compress=True)

        final_tif = tmp_resample_tif

        if method in ['burn-streams', 'burn-breach']:
            stream_threshold, decrement_value, gradient_distance = _get_burn_params(f)

            if method == 'burn-breach':
                dem_hydro = wbe.breach_depressions_least_cost(
                    dem_wb, max_cost=2.0, max_dist=20, minimize_dist=True
                )
            else:
                dem_hydro = dem_wb

            d8_pointer = wbe.d8_pointer(dem_hydro)
            flow_accum = wbe.d8_flow_accum(dem_hydro, out_type="cells")
            streams_raster = wbe.extract_streams(
                flow_accumulation=flow_accum,
                threshold=stream_threshold
            )
            streams_vector = wbe.raster_streams_to_vector(
                streams=streams_raster,
                d8_pointer=d8_pointer
            )

            dem_burned = wbe.burn_streams(
                dem=dem_resampled,
                streams=streams_vector,
                decrement_value=decrement_value,
                gradient_distance=gradient_distance
            )
            wbe.write_raster(dem_burned, tmp_burned_tif, compress=True)
            final_tif = tmp_burned_tif

        _convert_tif_to_asc(final_tif, resample_asc)

        newgrid = Grid.from_ascii(resample_asc, crs=Proj(crs))
        corrected_dem_raster = newgrid.read_ascii(resample_asc, crs=Proj(crs))
        rim_raster, acc_ds, catch_new, nodata_val = _route_and_delineate(
            newgrid, corrected_dem_raster, new_outx, new_outy
        )

        # overwrite output with rim-applied version for consistency
        newgrid.to_ascii(rim_raster, resample_asc, nodata=nodata_val)

        for tmpf in [tmp_resample_tif, tmp_burned_tif]:
            if os.path.exists(tmpf):
                try:
                    os.remove(tmpf)
                except OSError:
                    pass

    else:
        raise ValueError(
            "method must be one of ['hydro-aware', 'mean', 'nearest', 'bilinear', "
            "'burn-streams', 'burn-breach']"
        )

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    if plot_dem and acc_ds is not None and catch_new is not None:
        acc_clip = np.where(acc_ds > 500, 500, acc_ds)
        plt.figure(figsize=(10, 8))
        plt.imshow(catch_new, cmap='Blues', interpolation='nearest')
        plt.imshow(acc_clip, cmap='binary', interpolation='nearest', alpha=0.7)
        plt.colorbar(label='Accumulation (cells)')
        plt.title(f"Resampled DEM, factor={downscale_factor}, method={method}")

        if output_dirs and 'png' in output_dirs:
            os.makedirs(output_dirs['png'], exist_ok=True)
            plt.savefig(
                os.path.join(output_dirs['png'], f'Resampled_DEM_{downscale_factor}_{method}.png'),
                dpi=300, bbox_inches='tight'
            )
        plt.show()

    print(f"Resampled DEM saved to: {resample_asc}")
    return cols, [catch_raw, catch_new], acc_orig


def plot_distribution_comparison(raw, data, masks=None, output_dirs=None, title='Category Distribution Comparison'):
    def get_pct(arr):
        vals, counts = np.unique(arr[~np.isnan(arr)], return_counts=True)
        pct = counts / counts.sum() * 100
        return vals.astype(int), pct

    if masks:
        raw, data = np.where(masks[0], raw, np.nan), np.where(masks[1], data, np.nan)

    rv, rp = get_pct(raw)
    dv, dp = get_pct(data)
    allv = np.union1d(rv, dv)
    ra = np.array([rp[np.where(rv == v)[0][0]] if v in rv else 0 for v in allv])
    da = np.array([dp[np.where(dv == v)[0][0]] if v in dv else 0 for v in allv])

    x = np.arange(len(allv))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, ra, 0.6, label='Original', color='skyblue')
    ax.bar(x, da, 0.6, bottom=ra, label='Resampled', color='salmon')
    for i in range(len(x)):
        if ra[i] > 0: ax.text(x[i], ra[i]/2, f'{ra[i]:.1f}%', ha='center', va='center', fontsize=8)
        if da[i] > 0: ax.text(x[i], ra[i] + da[i]/2, f'{da[i]:.1f}%', ha='center', va='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(allv, rotation=45)
    ax.set_ylabel('Percentage (%)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if output_dirs:
        plt.savefig(os.path.join(output_dirs['png'], title + '.png'), dpi=300)
        with open(os.path.join(output_dirs['png'], title + '.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Value', 'Original (%)', 'Resampled (%)'])
            writer.writerows(zip(allv, ra.round(3), da.round(3)))
    plt.show()


def resample_with_weights(src_or_data,
                          band=1,
                          downscale_factor=2,
                          method="average",
                          acc=None,
                          class_weight=None,
                          nodata=None,
                          auto_weight=False,
                          max_iter=5,
                          tol=1e-3,
                          clip=(0.25, 4.0),
                          eps=1e-9,
                          seed=0,
                          auto_reassign=False,
                          # --- updated params ---
                          average_strategy="mean",       # 'mean' | 'mask' | 'acc' | 'acc_mask'
                          avg_mask=None,
                          hellinger_tol=1e-3):

    rng = np.random.default_rng(seed)

    # read
    if isinstance(src_or_data, np.ndarray):
        data = np.asarray(src_or_data)
        rows, cols = data.shape
        out_rows, out_cols = math.ceil(rows/downscale_factor), math.ceil(cols/downscale_factor)
        src = None
    else:
        src = src_or_data
        rows, cols = src.height, src.width
        out_rows, out_cols = math.ceil(rows/downscale_factor), math.ceil(cols/downscale_factor)
        data = src.read(band)

    # masks
    if acc is not None and acc.shape != data.shape:
        raise ValueError("acc shape must match data.")
    if avg_mask is not None and avg_mask.shape != data.shape:
        raise ValueError("avg_mask shape must match data.")
    if nodata is None and np.issubdtype(data.dtype, np.floating):
        nodata_mask = np.isnan(data)
    else:
        nodata_mask = (data == nodata) if nodata is not None else np.zeros_like(data, bool)
    out_fill = nodata if nodata is not None else 0

    # helpers
    def hist_pct_cells(arr):
        a = arr[~nodata_mask].astype(np.int64, copy=False)
        if a.size == 0: return {}
        k, c = np.unique(a, return_counts=True)
        return {int(kk): float(cc/c.sum()) for kk, cc in zip(k, c)}
    
    def hellinger_distance(p_dict, q_dict):
        """Compute Hellinger distance between two discrete distributions."""
        classes = set(p_dict) | set(q_dict)
        if not classes:
            return 0.0
        classes = sorted(classes)
        p = np.array([p_dict.get(k, 0.0) for k in classes], dtype=float)
        q = np.array([q_dict.get(k, 0.0) for k in classes], dtype=float)
        # defensive normalization
        p = np.clip(p, 0.0, 1.0)
        q = np.clip(q, 0.0, 1.0)
        sp = p.sum()
        sq = q.sum()
        if sp > 0:
            p /= sp
        if sq > 0:
            q /= sq
        return float(np.sqrt(((np.sqrt(p) - np.sqrt(q)) ** 2).sum()) / np.sqrt(2.0))

    def clean_weights(w):
        w = np.asarray(w, dtype=float)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        return np.where(w > 0.0, w, 0.0)

    def choose_unweighted_mode(v):
        u, inv = np.unique(v, return_inverse=True)
        cnt = np.bincount(inv)
        cand = np.flatnonzero(cnt == cnt.max())
        return u[rng.choice(cand)].astype(data.dtype, copy=False)

    def build_agree_mask(assign, valid_blocks=None):
        agree_mask = np.zeros_like(data, dtype=np.uint8)
        for i in range(out_rows):
            r0, r1 = i*downscale_factor, min((i+1)*downscale_factor, rows)
            for j in range(out_cols):
                if valid_blocks is not None and not valid_blocks[i, j]:
                    continue
                c0, c1 = j*downscale_factor, min((j+1)*downscale_factor, cols)
                m = ~nodata_mask[r0:r1, c0:c1]
                if not np.any(m):
                    continue
                chosen = assign[i, j]
                blk = data[r0:r1, c0:c1]
                am = (blk == chosen).astype(np.uint8)
                am[~m] = 0
                agree_mask[r0:r1, c0:c1] = am
        return agree_mask

    def desired_counts_from_pct(target_pct, classes, n_cells):
        """Integer coarse-cell target counts using largest remainder; ties favor rare classes."""
        classes = sorted(int(k) for k in classes)
        raw = {int(k): float(target_pct.get(k, 0.0) * n_cells) for k in classes}
        desired = {k: int(np.floor(v)) for k, v in raw.items()}
        remaining = int(n_cells - sum(desired.values()))

        if remaining > 0:
            order = sorted(
                classes,
                key=lambda k: (-(raw[k] - desired[k]), target_pct.get(k, 0.0), k)
            )
            for k in order[:remaining]:
                desired[k] += 1
        elif remaining < 0:
            order = sorted(
                classes,
                key=lambda k: (raw[k] - desired[k], -target_pct.get(k, 0.0), k)
            )
            for k in order[:abs(remaining)]:
                desired[k] = max(0, desired[k] - 1)
        return desired

    def block_mode(w_map=None):
        """Return per-block mode (weighted if provided) and a full-size agree mask."""
        out = np.full((out_rows, out_cols), out_fill, dtype=data.dtype)
        valid = np.zeros((out_rows, out_cols), dtype=bool)
        agree_mask = np.zeros_like(data, dtype=np.uint8)

        for i in range(out_rows):
            r0, r1 = i*downscale_factor, min((i+1)*downscale_factor, rows)
            for j in range(out_cols):
                c0, c1 = j*downscale_factor, min((j+1)*downscale_factor, cols)
                m = ~nodata_mask[r0:r1, c0:c1]
                if not np.any(m):
                    continue
                valid[i, j] = True
                v = data[r0:r1, c0:c1][m].astype(float, copy=False)

                if w_map is None and acc is None:
                    chosen = choose_unweighted_mode(v)
                else:
                    w = np.ones_like(v)
                    if acc is not None:
                        w *= acc[r0:r1, c0:c1][m].astype(float, copy=False)
                    if w_map is not None:
                        w *= np.vectorize(lambda x: w_map.get(int(x), 1.0), otypes=[float])(v)
                    w = clean_weights(w)
                    if float(w.sum()) <= eps:
                        chosen = choose_unweighted_mode(v)
                    else:
                        u, inv = np.unique(v, return_inverse=True)
                        ws = np.bincount(inv, weights=w, minlength=len(u))
                        if not np.any(ws > eps):
                            chosen = choose_unweighted_mode(v)
                        else:
                            cand = np.flatnonzero(ws == ws.max())
                            chosen = u[rng.choice(cand)].astype(data.dtype, copy=False)

                out[i, j] = chosen
                blk = data[r0:r1, c0:c1]
                am = (blk == chosen).astype(np.uint8)
                am[~m] = 0
                agree_mask[r0:r1, c0:c1] = am

        return out, valid, agree_mask

    # average
    if method == "average":
        strat = average_strategy.lower()
        if strat == "mean" and (src is not None) and (Resampling is not None):
            return src.read(band, out_shape=(out_rows, out_cols), resampling=Resampling.average)

        out = np.zeros((out_rows, out_cols), dtype=float)
        for i in range(out_rows):
            r0, r1 = i*downscale_factor, min((i+1)*downscale_factor, rows)
            for j in range(out_cols):
                c0, c1 = j*downscale_factor, min((j+1)*downscale_factor, cols)
                m = ~nodata_mask[r0:r1, c0:c1]
                if not np.any(m):
                    continue
                vals = data[r0:r1, c0:c1][m].astype(float, copy=False)

                if strat == "mean":
                    out[i, j] = float(vals.mean())
                elif strat == "mask":
                    if avg_mask is None: raise ValueError("average_strategy='mask' requires avg_mask.")
                    w = avg_mask[r0:r1, c0:c1][m].astype(float, copy=False)
                    out[i, j] = float(vals.mean()) if w.sum() == 0 else float(np.average(vals, weights=w))
                elif strat == "acc":
                    if acc is None: raise ValueError("average_strategy='acc' requires acc.")
                    w = acc[r0:r1, c0:c1][m].astype(float, copy=False)
                    out[i, j] = float(vals.mean()) if w.sum() == 0 else float(np.average(vals, weights=w))
                elif strat == "acc_mask":
                    if acc is None or avg_mask is None:
                        raise ValueError("average_strategy='acc_mask' requires both acc and avg_mask.")
                    w = (acc[r0:r1, c0:c1] * avg_mask[r0:r1, c0:c1])[m].astype(float, copy=False)
                    out[i, j] = float(vals.mean()) if w.sum() == 0 else float(np.average(vals, weights=w))
                else:
                    raise ValueError("average_strategy must be 'mean', 'mask', 'acc', or 'acc_mask'.")

        return out

    # mode: always return (out, mask)
    if method != "mode":
        raise ValueError("method must be 'average' or 'mode'")

    # class weights
    if class_weight:
        out, _, agree = block_mode(w_map=dict(class_weight))
        return out, agree

    # auto-balance
    if auto_weight:
        target_pct = hist_pct_cells(data)
        from collections import defaultdict as _dd
        weights = _dd(lambda: 1.0)
        out, valid, agree = None, None, None
        best_out, best_agree = None, None
        best_score = (float("inf"), float("inf"))
        for _it in range(max_iter):
            out, valid, agree = block_mode(w_map=dict(weights))
            a = out[valid].astype(np.int64, copy=False)
            if a.size == 0: break
            k, c = np.unique(a, return_counts=True)
            cur_pct = {int(kk): float(cc/c.sum()) for kk, cc in zip(k, c)}
            classes = set(target_pct) | set(cur_pct)
            if not classes: break

            hell = hellinger_distance(target_pct, cur_pct)
            max_err = max(abs(target_pct.get(k,0.0) - cur_pct.get(k,0.0)) for k in classes)
            score = (hell, max_err)
            if score < best_score:
                best_score = score
                best_out = out.copy()
                best_agree = agree.copy()

            if hellinger_tol is not None:
                if hell <= hellinger_tol:
                    break

            if max_err <= tol: break
            up = {}
            for kk in classes:
                ps, po = target_pct.get(kk,0.0), cur_pct.get(kk,0.0)
                w = (ps+eps)/(po+eps)
                if clip: w = float(np.clip(w, clip[0], clip[1]))
                up[int(kk)] = w
            mu = np.mean(list(up.values())) if up else 1.0
            for kk, ww in up.items():
                weights[kk] *= ww/(mu if mu>0 else 1.0)
        if best_out is None:
            best_out, _, best_agree = block_mode(w_map=dict(weights))
        return best_out, best_agree

    # auto reassign
    if auto_reassign:
        target_pct = hist_pct_cells(data)
        blk_info = []
        assign = np.full((out_rows, out_cols), out_fill, dtype=data.dtype)
        valid = np.zeros((out_rows, out_cols), dtype=bool)

        for i in range(out_rows):
            r0, r1 = i*downscale_factor, min((i+1)*downscale_factor, rows)
            for j in range(out_cols):
                c0, c1 = j*downscale_factor, min((j+1)*downscale_factor, cols)
                m = ~nodata_mask[r0:r1, c0:c1]
                if not np.any(m):
                    continue
                valid[i, j] = True
                v = data[r0:r1, c0:c1][m].astype(np.int64, copy=False)
                u, cnt = np.unique(v, return_counts=True)
                ords = np.lexsort((u, -cnt))
                u, cnt = u[ords], cnt[ords]
                assign[i, j] = u[0]
                blk_info.append({"id": (i, j), "classes": u, "counts": cnt, "size": int(cnt.sum())})

        a = assign[valid].astype(np.int64, copy=False)
        N = max(a.size, 1)
        k, c = np.unique(a, return_counts=True)
        cur_pct = {int(kk): float(cc/N) for kk, cc in zip(k, c)}

        # Early stop if Hellinger distance already small
        if hellinger_tol is not None:
            hell = hellinger_distance(target_pct, cur_pct)
            if hell <= hellinger_tol:
                return assign, build_agree_mask(assign, valid)

        def current_counts():
            aa = assign[valid].astype(np.int64, copy=False)
            if aa.size==0: return {}
            kk, cc = np.unique(aa, return_counts=True)
            return {int(kkk): int(ccc) for kkk, ccc in zip(kk, cc)}

        classes = set(target_pct) | set(cur_pct)
        desired = desired_counts_from_pct(target_pct, classes, N)
        cur_counts = current_counts()
        counts = {int(k): cur_counts.get(int(k), 0) for k in classes}

        for cls in sorted(classes, key=lambda k: (desired.get(k, 0) - counts.get(k, 0), target_pct.get(k, 0.0)), reverse=True):
            while counts.get(cls, 0) < desired.get(cls, 0):
                surplus = {k for k, v in counts.items() if v > desired.get(k, 0)}
                if not surplus:
                    break

                cand = []
                for b in blk_info:
                    i, j = b["id"]
                    cur = int(assign[i, j])
                    if cur not in surplus or cur == cls:
                        continue
                    matches = np.flatnonzero(b["classes"] == cls)
                    if matches.size == 0:
                        continue
                    rk = int(matches[0])
                    if rk == 0:
                        continue
                    score = float(b["counts"][rk] / b["size"])
                    surplus_count = counts.get(cur, 0) - desired.get(cur, 0)
                    cand.append((rk, score, surplus_count, i, j, cur))

                if not cand:
                    break

                cand.sort(key=lambda x: (x[0], -x[1], -x[2], x[3]*out_cols + x[4]))
                rk, score, surplus_count, ii, jj, old_cls = cand[0]
                assign[ii, jj] = cls
                counts[old_cls] = counts.get(old_cls, 0) - 1
                counts[cls] = counts.get(cls, 0) + 1

        return assign, build_agree_mask(assign, valid)

    # plain mode
    out, _, agree = block_mode(w_map=None)
    return out, agree


def resample_xml(xml_path, output_folder, downscale_factor=2, crs="EPSG:26910", plot_dem=False, overwrite=True,plot_hist=False,weights=None,change_disturbance_fraction=False, num_processors=8, num_subbasins=1, plot_subdivide=False, dem_method='hydro-aware', class_method='majority',avg_method='mean'):
    """
    Resample data in an XML file, including DEM and CSV files.

    Parameters:
    - xml_path (str): Path to input XML file.
    - output_folder (str): Folder for resampled outputs.
    - downscale_factor (int): Scaling factor for resampling.
    - crs (str): Coordinate Reference System.
    - plot_dem (bool): If True, plot the DEM.
    - overwrite (bool): If True, overwrite existing files.
    - plot_hist (bool): If True, plot histograms for categorical data.
    - weights (dict): Weights for categorical data resampling.
    - change_disturbance_fraction (bool): If True, adjust disturbance fractions.
    - num_processors (int): Number of processors for catchment subdivision.
    - num_subbasins (int): Number of subbasins for catchment subdivision.
    - plot_subdivide (bool): If True, plot the subdivided catchments.
    """
    if downscale_factor==1:
        return xml_path
    xml_output_path = xml_path.replace('.xml', f'_resampled_{downscale_factor}_{dem_method}-{class_method}-{avg_method}.xml')
    if os.path.exists(xml_output_path) and not overwrite:
        print(f"Output XML already exists: {xml_output_path}")
        return xml_output_path
    tree = ET.parse(xml_path)
    root = tree.getroot()
    parent_map = {child: parent for parent in root.iter() for child in parent}

    # Parse base paths
    root_name = next((elem.text for elem in root.iter() if 'inputDataLocationRootName' in elem.tag), '')
    dir_name = next((elem.text for elem in root.iter() if 'inputDataLocationDirName' in elem.tag), '')
    outx = next((int(elem.text) for elem in root.iter() if 'outx' in elem.tag), '')
    outy = next((int(elem.text) for elem in root.iter() if 'outy' in elem.tag), '')
    base_path = os.path.join(root_name, dir_name)
    colmax=None
    hist_data = []
    landcover_mask = None
    soil_mask = None
    
    # Create output directories
    subfolders = ['xmls', 'asc', 'csv', 'png']
    output_dirs = {sub: os.path.join(base_path, output_folder, sub) for sub in subfolders}
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    for elem in root.iter():
        if elem.text and elem.text.endswith('.asc'):
            input_asc = elem.text if os.path.isabs(elem.text) else os.path.join(base_path, elem.text)
            output_asc = os.path.join(output_dirs['asc'], elem.text.split('/')[-1].replace('.asc', f'_resampled_{downscale_factor}_{dem_method}-{class_method}-{avg_method}.asc'))
            if elem.tag.endswith('input_dem'):
                colmax, masks, acc=resample_dem(input_asc, output_asc, outx= outx, outy=outy, downscale_factor=downscale_factor,plot_dem=plot_dem,output_dirs=output_dirs, method=dem_method)
                outlets=subdivide_catchments(output_asc, outx//downscale_factor, outy//downscale_factor, num_processors, num_subbasins, method='equal', crs=crs, is_plot=plot_subdivide,save_dir=output_dirs['png'])
                elem.text = os.path.relpath(output_asc, base_path)
                print(f"Resampled ASC saved: {output_asc}")

    for elem in root.iter():
        if elem.text and elem.text.endswith('.asc'):
            input_asc = elem.text if os.path.isabs(elem.text) else os.path.join(base_path, elem.text)
            output_asc = os.path.join(output_dirs['asc'], elem.text.split('/')[-1].replace('.asc', f'_resampled_{downscale_factor}_{dem_method}-{class_method}-{avg_method}.asc'))             
            
            if not elem.tag.endswith('input_dem') and elem.tag.endswith(('coverSpeciesIndexMapFileName','soilParametersIndexMapFileName','filterMapFullName')):
                with rasterio.open(input_asc) as src:                                                
                    if class_method == 'auto-weight':
                        data, mode_mask = resample_with_weights(src, band=1, downscale_factor=downscale_factor, method='mode', auto_weight=True)
                    elif class_method == 'auto-reassign':
                        data, mode_mask = resample_with_weights(src, band=1, downscale_factor=downscale_factor, method='mode', auto_reassign=True)
                    elif class_method == 'hydro-aware':
                        data, mode_mask = resample_with_weights(src, band=1, downscale_factor=downscale_factor, method='mode', acc=acc)
                    elif class_method in ('majority', 'mode'):
                        data, mode_mask = resample_with_weights(src, band=1, downscale_factor=downscale_factor, method='mode', class_weight=weights.get(elem.tag) if weights else None)
                    else:
                        raise ValueError("class_method must be 'majority', 'auto-weight', 'auto-reassign', or 'hydro-aware'")
                    if elem.tag.endswith('coverSpeciesIndexMapFileName'):
                        landcover_mask=mode_mask
                    elif elem.tag.endswith('soilParametersIndexMapFileName'):
                        soil_mask=mode_mask
                    raw= src.read(1)
                    hist_data.append((elem.tag, raw, data))
                    transform = src.transform * src.transform.scale(downscale_factor, downscale_factor)
                    profile = src.profile
                    profile.update(driver='AAIGrid', height=data.shape[0], width=data.shape[1], transform=transform, crs=crs or src.crs)
                    with rasterio.open(output_asc, 'w', **profile) as dst:
                        dst.write(data, 1)
                elem.text = os.path.relpath(output_asc, base_path)
                print(f"Resampled ASC saved: {output_asc}")

    for elem in root.iter():
        if elem.text and elem.text.endswith('.asc'):
            input_asc = elem.text if os.path.isabs(elem.text) else os.path.join(base_path, elem.text)
            output_asc = os.path.join(output_dirs['asc'], elem.text.split('/')[-1].replace('.asc', f'_resampled_{downscale_factor}_{dem_method}-{class_method}-{avg_method}.asc'))             
            
            if not elem.tag.endswith('input_dem') and not elem.tag.endswith(('coverSpeciesIndexMapFileName','soilParametersIndexMapFileName','filterMapFullName')):
                with rasterio.open(input_asc) as src: 
                    if avg_method =='mean':                                                 
                        data = resample_with_weights(src, band=1, downscale_factor=downscale_factor, method='average')
                    elif avg_method =='hydro-aware':
                        data = resample_with_weights(src, band=1, downscale_factor=downscale_factor, method='average', acc=acc, average_strategy='acc')
                    elif avg_method == 'landcover-aware':
                        data = resample_with_weights(src, band=1, downscale_factor=downscale_factor, method='average', avg_mask=landcover_mask, average_strategy='mask')
                    elif avg_method == 'soil-aware':
                        data = resample_with_weights(src, band=1, downscale_factor=downscale_factor, method='average', avg_mask=soil_mask, average_strategy='mask')
                    else:
                        raise ValueError(f"Unknown avg_method: {avg_method}")
                    transform = src.transform * src.transform.scale(downscale_factor, downscale_factor)
                    profile = src.profile
                    profile.update(driver='AAIGrid', height=data.shape[0], width=data.shape[1], transform=transform, crs=crs or src.crs)
                    with rasterio.open(output_asc, 'w', **profile) as dst:
                        dst.write(data, 1)

                elem.text = os.path.relpath(output_asc, base_path)
                print(f"Resampled ASC saved: {output_asc}")

        elif elem.tag.endswith('setStartStateSpatialDataLocationName') and elem.text and os.path.isdir(os.path.join(base_path, elem.text.strip())):
            input_dir = elem.text if os.path.isabs(elem.text) else os.path.join(base_path, elem.text)
            output_dir = os.path.join(
                output_dirs['asc'],
                os.path.basename(input_dir) + f'_resampled_{downscale_factor}_{dem_method}-{class_method}-{avg_method}'
            )
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            for root_dir, _, files in os.walk(input_dir):
                for f in files:
                    in_path = os.path.join(root_dir, f)
                    rel = os.path.relpath(in_path, input_dir)
                    out_path = os.path.join(output_dir, rel)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    if f.lower().endswith('.asc'):
                        with rasterio.open(in_path) as src:
                            if avg_method == 'mean':
                                data = resample_with_weights(src, band=1, downscale_factor=downscale_factor, method='average')
                            elif avg_method == 'hydro-aware':
                                data = resample_with_weights(src, band=1, downscale_factor=downscale_factor,
                                                            method='average', acc=acc, average_strategy='acc')
                            elif avg_method == 'landcover-aware':
                                data = resample_with_weights(src, band=1, downscale_factor=downscale_factor,
                                                            method='average', avg_mask=landcover_mask, average_strategy='mask')
                            elif avg_method == 'soil-aware':
                                data = resample_with_weights(src, band=1, downscale_factor=downscale_factor,
                                                            method='average', avg_mask=soil_mask, average_strategy='mask')
                            else:
                                raise ValueError(f"Unknown avg_method: {avg_method}")

                            transform = src.transform * src.transform.scale(downscale_factor, downscale_factor)
                            profile = src.profile
                            profile.update(driver='AAIGrid',
                                        height=data.shape[0],
                                        width=data.shape[1],
                                        transform=transform,
                                        crs=crs or src.crs)
                            with rasterio.open(out_path, 'w', **profile) as dst:
                                dst.write(data, 1)
                        print(f"Resampled ASC saved: {out_path}")
                    else:
                        with open(in_path, 'rb') as rf, open(out_path, 'wb') as wf:
                            wf.write(rf.read())
                        print(f"Copied (non-ASC): {out_path}")

            elem.text = os.path.relpath(output_dir, base_path)
            print(f"Resampled folder saved: {output_dir}")


            
        elif 'weatherLocationsDataFileName' in elem.tag and elem.text.endswith('.csv'):
            input_csv = elem.text if os.path.isabs(elem.text) else os.path.join(base_path, elem.text)
            output_csv = os.path.join(output_dirs['csv'], elem.text.split('/')[-1].replace('.csv', f'_resampled_{downscale_factor}_{dem_method}-{class_method}-{avg_method}.csv'))
            df = pd.read_csv(input_csv, header=None)
            df.iloc[:, 0] = (df.iloc[:, 0] // downscale_factor).astype(int)
            df.iloc[:, 1] = (df.iloc[:, 1] // downscale_factor).astype(int)
            df.iloc[:, 3] = df.iloc[:, 3].apply(lambda path: os.path.normpath(path))
            df.to_csv(output_csv, header=None, index=False)
            elem.text = os.path.relpath(output_csv, base_path)
            print(f"Resampled CSV saved: {output_csv}")

        elif elem.tag.endswith(('outx', 'outy', 'cellX', 'cellY')):
            original_value = int(elem.text)
            elem.text = str(original_value // downscale_factor)
            print(f"Updated {elem.tag}: {original_value} -> {elem.text}")
            
        elif elem.tag.endswith('initializeHistoricalData'):
            input_file=elem.text if os.path.isabs(elem.text) else os.path.join(base_path, elem.text)            
            output_file = os.path.join(output_dirs['csv'], elem.text.split('/')[-1].rsplit('.', 1)[0] + f'_resampled_{downscale_factor}_{dem_method}-{class_method}-{avg_method}.csv')
            data_by_index = defaultdict(list)
            delimiter = ',' if input_file.endswith('.csv') else ' '
            with open(input_file, "r") as f_in:
                for line in f_in:                
                    parts = line.strip().split(delimiter)                    
                    if not parts:
                        continue
                    try:
                        old_index = int(parts[0])
                        row = old_index // colmax
                        col = old_index % colmax
                        new_row = row // downscale_factor
                        new_colmax = math.ceil(colmax / downscale_factor)
                        new_col = col // downscale_factor
                        new_index = new_row * new_colmax + new_col
                        date_pairs = [(int(parts[i]), int(parts[i+1])) for i in range(1, len(parts)-1, 2)]
                        data_by_index[new_index].extend(date_pairs)
                    except ValueError:
                        continue

            with open(output_file, "w", newline="") as f_out:
                writer = csv.writer(f_out)
                for index in sorted(data_by_index.keys()):
                    sorted_pairs = sorted(data_by_index[index], key=lambda x: (x[0], x[1]))
                    flat_list = [index] + [item for pair in sorted_pairs for item in pair]
                    writer.writerow(flat_list)

            elem.text = os.path.relpath(output_file, base_path)
            if change_disturbance_fraction:
                parent = parent_map.get(elem)
                if parent is not None:
                    for child in parent:
                        if child is not elem and child.tag.endswith("harvestFraction"):
                            try:
                                value = float(child.text)
                                child.text = str(value / downscale_factor/ downscale_factor)
                            except (TypeError, ValueError):
                                pass
            print(f"Resampled CSV saved: {output_file}")  

        elif elem.tag.endswith('modificationsDataFileName'):
            input_file = elem.text if os.path.isabs(elem.text) else os.path.join(base_path, elem.text)
            output_file = os.path.join(output_dirs['csv'], elem.text.split('/')[-1].rsplit('.', 1)[0] + f'_resampled_{downscale_factor}_{dem_method}-{class_method}-{avg_method}.csv')
            data_by_key = defaultdict(list)
            delimiter = ',' if input_file.endswith('.csv') else ' '
        
            with open(input_file, "r") as f_in:
                for line in f_in:
                    parts = line.strip().split(delimiter)
                    if len(parts) < 4:
                        continue
                    try:
                        time1 = int(parts[0])
                        time2 = int(parts[1])
                        old_index = int(parts[2])
                        row = old_index // colmax
                        col = old_index % colmax
                        new_row = row // downscale_factor
                        new_colmax = math.ceil(colmax / downscale_factor)
                        new_col = col // downscale_factor
                        new_index = new_row * new_colmax + new_col
                        key = (time1, time2, new_index)
                        values = tuple(parts[3:])  
                        data_by_key[key].append(values)
                    except ValueError:
                        continue
        
            with open(output_file, "w", newline="") as f_out:
                writer = csv.writer(f_out, delimiter=delimiter)
                for key in sorted(data_by_key.keys()):
                    counter = Counter(data_by_key[key])
                    most_common_values, _ = counter.most_common(1)[0]
                    time1, time2, index = key
                    row = [time1, time2, index] + list(most_common_values)
                    writer.writerow(row)
        
            elem.text = os.path.relpath(output_file, base_path)
            print(f"Resampled CSV saved: {output_file}")   

        elif elem.tag.endswith("initializeSpecificCells"):
            input_file = elem.text if os.path.isabs(elem.text) else os.path.join(base_path, elem.text)
            output_file = os.path.join(output_dirs['csv'], elem.text.split('/')[-1].rsplit('.', 1)[0] + f'_resampled_{downscale_factor}_{dem_method}-{class_method}-{avg_method}.csv')

            data_by_index = defaultdict(list)
            delimiter = ',' if input_file.endswith('.csv') else ' '

            with open(input_file, "r") as f_in:
                for line in f_in:
                    parts = line.strip().split(delimiter)
                    if len(parts) < 2:
                        continue
                    try:
                        old_index = int(parts[0])
                        row = old_index // colmax
                        col = old_index % colmax
                        new_row = row // downscale_factor
                        new_colmax = math.ceil(colmax / downscale_factor)
                        new_col = col // downscale_factor
                        new_index = new_row * new_colmax + new_col

                        value = float(parts[1])
                        data_by_index[new_index].append(value)
                    except ValueError:
                        continue

            with open(output_file, "w", newline="") as f_out:
                writer = csv.writer(f_out)
                for index in sorted(data_by_index.keys()):
                    avg_value = sum(data_by_index[index]) / len(data_by_index[index])
                    writer.writerow([index, round(avg_value, 6)])

            elem.text = os.path.relpath(output_file, base_path)        
            print(f"Resampled CSV saved: {output_file}") 
        elif elem.tag.endswith('initializeOutputDataLocationRoot'):
            elem.text = elem.text+f'/{downscale_factor}_{dem_method}-{class_method}-{avg_method}'
            print(f"Updated {elem.tag}: {elem.text}")


    for elem in root.iter():
        if elem.tag.endswith('initialReachOutlets'):
            elem.text = outlets or ''
            print(f"Updated {elem.tag}: {elem.text}")
    
    if plot_hist:
        for key, raw, data in hist_data:
            plot_distribution_comparison(raw, data, masks, output_dirs=output_dirs, title=f"Distribution Comparison for {key}_resampled_{downscale_factor}_{dem_method}-{class_method}-{avg_method}")
    output_path = os.path.join(output_dirs['xmls'], os.path.basename(xml_path).replace('.xml', f'_resampled_{downscale_factor}_{dem_method}-{class_method}-{avg_method}.xml'))
    
    tree.write(xml_output_path, encoding='utf-8', xml_declaration=True)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Updated XML saved to {xml_output_path}")
    print(f"Updated XML saved to {output_path}")
    print("Warning: Resampling may place the watershed too close to the DEM edge, which can break flow paths and cause 'Index -1' errors in VELMA.")
    print("Check the ReachMap and modify resampled DEM, or reduce the downscale factor.")
    return xml_output_path

import sys
# Example usage
if __name__ == "__main__": 
   
    label='Big_Beef'
    downscale_factor=4
    dem_method= 'burn-streams'
    class_method= 'auto-reassign'
    avg_method= 'mean'

    xml_file = f'{label}/XML/orig.xml'
    print(f"Processing {xml_file}")
    resample_xml(xml_file, 'resampled', downscale_factor=downscale_factor, plot_dem=True, plot_subdivide=True, overwrite=True, plot_hist=True, dem_method=dem_method, class_method=class_method, avg_method=avg_method)

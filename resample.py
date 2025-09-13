import os
import rasterio
from rasterio.enums import Resampling
import xml.etree.ElementTree as ET
import pandas as pd
from pysheds.grid import Grid
import matplotlib.pyplot as plt
import numpy as np
from pysheds.sview import Raster, ViewFinder
from pyproj import Proj
from collections import defaultdict, Counter
import csv,math
from subdivide import subdivide_catchments

def resample_dem(input_asc, resample_asc, outx=None, outy=None, crs="EPSG:4326", downscale_factor=2,plot_dem=False,output_dirs=None,method='hydro-aware'):
    """
    Resample a DEM using accumulation-based selection.
    
    Parameters:
    - input_asc (str): Path to input ASC file.
    - resample_asc (str): Path to output ASC file.
    - outx (int, optional): Number of columns in the resampled DEM. Calculated automatically if not provided.
    - outy (int, optional): Number of rows in the resampled DEM. Calculated automatically if not provided.
    - crs (str): Coordinate Reference System (default: "EPSG:4326").
    - downscale_factor (int): Scaling factor for downsampling (e.g., 2 means 2x coarser resolution).
    - plot_dem (bool): If True, plots the original and resampled DEM for visual comparison.
    """
    grid = Grid.from_ascii(input_asc, crs=Proj(crs))  # Load grid
    orig_dem = grid.read_ascii(input_asc, crs=Proj(crs))  # Read DEM
    
    # Fill pits in the DEM
    pit_filled_dem = grid.fill_pits(orig_dem)
    # Fill depressions in the DEM
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    # Resolve flat areas in the DEM
    dem = grid.resolve_flats(flooded_dem)

    fdir = grid.flowdir(dem)  # Calculate flow direction
    catch_raw = grid.catchment(x=outx, y=outy, fdir=fdir, xytype='index')  # Get catchment area
    acc = grid.accumulation(fdir)  # Calculate accumulation
    cols, rows = acc.shape
    # Initialize resampled DEM
    rows, cols = dem.shape
    new_rows, new_cols = math.ceil(rows / downscale_factor), math.ceil(cols / downscale_factor)
    corrected_dem = np.zeros((new_rows, new_cols))

    # Process blocks for resampling
    f = downscale_factor
    for i in range(new_rows):
        for j in range(new_cols):
            r0, r1 = i * f, min((i + 1) * f, rows)
            c0, c1 = j * f, min((j + 1) * f, cols)

            block_dem = dem[r0:r1, c0:c1]
            block_acc = acc[r0:r1, c0:c1]

            total_acc = np.sum(block_acc)
            if method == 'hydro-aware' and total_acc > 0:
                corrected_dem[i, j] = np.sum(block_dem * block_acc) / total_acc
            else:
                corrected_dem[i, j] = np.mean(block_dem)

    new_outx, new_outy = outx// downscale_factor, outy// downscale_factor 
    min_value = np.min(corrected_dem[max(new_outy-1, 0):min(new_outy+1, new_rows-1)+1, 
                                 max(new_outx-1, 0):min(new_outx+1, new_cols-1)+1])
    corrected_dem[new_outy, new_outx] = min_value if corrected_dem[new_outy, new_outx] == min_value else min_value - 0.01
    corrected_dem=np.nan_to_num(corrected_dem, nan=dem.nodata)         
    new_viewfinder = ViewFinder(
        affine=dem.affine * dem.affine.scale(downscale_factor, downscale_factor),  # Update affine transform
        shape=corrected_dem.shape,
        crs=dem.crs,
        nodata=dem.nodata
    )

    corrected_dem_raster = Raster(corrected_dem, viewfinder=new_viewfinder)  # Create raster  
    newgrid = Grid.from_raster(corrected_dem_raster)  # Load into grid
    # Fill pits in the DEM
    pit_filled_dem = newgrid.fill_pits(corrected_dem_raster)
    # Fill depressions in the DEM
    flooded_dem = newgrid.fill_depressions(pit_filled_dem)
    # Resolve flat areas in the DEM
    corrected_dem_raster = newgrid.resolve_flats(flooded_dem)
    
    newgrid.to_ascii(corrected_dem_raster, resample_asc,nodata=newgrid.nodata)  # Save resampled DEM
    fdir = newgrid.flowdir(corrected_dem_raster)
    catch_new = newgrid.catchment(x=new_outx, y=new_outy, fdir=fdir, xytype='index')
    if plot_dem:  
        # corrected_dem_raster[catch == 0] = newgrid.nodata    
        acc = newgrid.accumulation(fdir)

        plt.figure(figsize=(10, 8))
        plt.imshow(catch_new, cmap='Blues', interpolation='nearest')
        plt.imshow(np.where(acc > 500, 500, acc), cmap='binary', interpolation='nearest', alpha=0.7)
        plt.colorbar(label='Catchment Area')
        plt.savefig(os.path.join(output_dirs['png'], f'Resampled_{downscale_factor}_catchment_area.png'), dpi=300)
        plt.show()
    print(f"Resampled DEM saved to: {resample_asc}")
    return cols, [catch_raw, catch_new]

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


def resample_with_weighted_mode(data, downscale_factor, weight_map=None):
    weight_map = weight_map or {}
    rows, cols = data.shape
    new_rows = math.ceil(rows / downscale_factor)
    new_cols = math.ceil(cols / downscale_factor)
    result = np.zeros((new_rows, new_cols), dtype=data.dtype)
    for i in range(new_rows):
        for j in range(new_cols):
            r0, r1 = i * downscale_factor, min((i + 1) * downscale_factor, rows)
            c0, c1 = j * downscale_factor, min((j + 1) * downscale_factor, cols)
            block = data[r0:r1, c0:c1].flatten()
            counts = Counter(int(v) for v in block if not np.isnan(v))
            for k in counts:
                counts[k] *= weight_map.get(k, 1)
            result[i, j] = counts.most_common(1)[0][0] if counts else 0
    return result

def resample_xml(xml_path, output_folder, downscale_factor=2, crs="EPSG:26910", plot_dem=False, overwrite=True,plot_hist=False,weights=None,change_disturbance_fraction=False, num_processors=8, num_subbasins=50, plot_subdivide=False):
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
    xml_output_path = xml_path.replace('.xml', f'_resampled_{downscale_factor}.xml')
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
    
    # Create output directories
    subfolders = ['xmls', 'asc', 'csv', 'png']
    output_dirs = {sub: os.path.join(base_path, output_folder, sub) for sub in subfolders}
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    for elem in root.iter():
        if elem.text and elem.text.endswith('.asc'):
            input_asc = elem.text if os.path.isabs(elem.text) else os.path.join(base_path, elem.text)
            output_asc = os.path.join(output_dirs['asc'], elem.text.split('/')[-1].replace('.asc', f'_resampled_{downscale_factor}.asc'))
            if elem.tag.endswith('input_dem'):
                colmax, masks=resample_dem(input_asc, output_asc, outx= outx, outy=outy, downscale_factor=downscale_factor,plot_dem=plot_dem,output_dirs=output_dirs)
                outlets=subdivide_catchments(input_asc, outx, outy, num_processors, num_subbasins, method='layer', crs=crs, is_plot=plot_subdivide,save_dir=output_dirs['png'])
                 
            else:
                with rasterio.open(input_asc) as src:                   
                    if elem.tag.endswith('coverSpeciesIndexMapFileName') or elem.tag.endswith('soilParametersIndexMapFileName') or elem.tag.endswith('filterMapFullName'):
                        if weights is not None and elem.tag in weights:
                            raw_data = src.read(1)
                            weight_map = weights[elem.tag]
                            data = resample_with_weighted_mode(raw_data, downscale_factor, weight_map=weight_map)                       
                        else:
                            data =src.read(1,  out_shape=(math.ceil(src.height / downscale_factor), math.ceil(src.width / downscale_factor)), resampling=Resampling.mode)
                    else:
                        data = src.read(1,  out_shape=(math.ceil(src.height / downscale_factor), math.ceil(src.width / downscale_factor)), resampling=Resampling.average)
                    transform = src.transform * src.transform.scale(downscale_factor, downscale_factor)
                    profile = src.profile
                    profile.update(driver='AAIGrid', height=data.shape[0], width=data.shape[1], transform=transform, crs=crs or src.crs)
                    with rasterio.open(output_asc, 'w', **profile) as dst:
                        dst.write(data, 1)
                    if elem.tag.endswith(('coverSpeciesIndexMapFileName', 'soilParametersIndexMapFileName')):
                        raw= src.read(1)
                        hist_data.append((elem.tag, raw, data))

            elem.text = os.path.relpath(output_asc, base_path)
            print(f"Resampled ASC saved: {output_asc}")

        elif 'weatherLocationsDataFileName' in elem.tag and elem.text.endswith('.csv'):
            input_csv = elem.text if os.path.isabs(elem.text) else os.path.join(base_path, elem.text)
            output_csv = os.path.join(output_dirs['csv'], elem.text.split('/')[-1].replace('.csv', f'_resampled_{downscale_factor}.csv'))
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
            output_file = os.path.join(output_dirs['csv'], elem.text.split('/')[-1].rsplit('.', 1)[0] + f'_resampled_{downscale_factor}.csv')
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
                        new_colmax = colmax // downscale_factor
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
            output_file = os.path.join(output_dirs['csv'], elem.text.split('/')[-1].rsplit('.', 1)[0] + f'_resampled_{downscale_factor}.csv')
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
                        new_colmax = colmax // downscale_factor
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
            output_file = os.path.join(output_dirs['csv'], elem.text.split('/')[-1].rsplit('.', 1)[0] + f'_resampled_{downscale_factor}.csv')

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
                        new_colmax = colmax // downscale_factor
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

    for elem in root.iter():
        if elem.tag.endswith('initialReachOutlets'):
            elem.text = outlets or ''
            print(f"Updated {elem.tag}: {elem.text}")
    
    if plot_hist:
        for key, raw, data in hist_data:
            plot_distribution_comparison(raw, data, masks, output_dirs=output_dirs, title=f"Distribution Comparison for {key}_resampled_{downscale_factor}")
    output_path = os.path.join(output_dirs['xmls'], os.path.basename(xml_path).replace('.xml', f'_resampled_{downscale_factor}.xml'))
    
    tree.write(xml_output_path, encoding='utf-8', xml_declaration=True)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Updated XML saved to {xml_output_path}")
    print(f"Updated XML saved to {output_path}")
    print("Warning: Resampling may place the watershed too close to the DEM edge, which can break flow paths and cause 'Index -1' errors in VELMA.")
    print("Check the ReachMap and modify resampled DEM, or reduce the downscale factor.")
    return xml_output_path

# Example usage
if __name__ == "__main__": 
    labels = ['Cedar', 'Deschutes', 'Duckabush', 'Elwha', 'Nisqually', 'Nooksack', 'Puyallup','Samish', 'Skagit', 'Skokomish', 'Snohomish']
    labels =[ 'Big_Beef']
    weights = {
    'coverSpeciesIndexMapFileName': {24: 3},
    'soilParametersIndexMapFileName': {17: 2}
    }
    for label in labels:
        xml_file = f'{label}/XML/1.xml'
        print(f"Processing {xml_file}")
        resample_xml(xml_file, 'resampled', downscale_factor=5, num_processors=8, num_subbasins=50, plot_dem=True, plot_subdivide=True, overwrite=True, plot_hist=True, weights=weights, change_disturbance_fraction=False)

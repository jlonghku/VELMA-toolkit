# Toolkit for VELMA

This repository provides a collection of tools for **eco-hydrological modeling workflows**, including case downscaleing, watershed subdivision, and parameter optimization.

---

## Tools Overview

### 1. Watershed Subdivision for Parallel Running (`subdivide_catchments`)
Divide large watersheds into subbasins and simulate task scheduling.  
- Methods: **equal**, **branch**, **layer**.  
- Task execution simulation and dependency resolution.  
- Visualization: subbasin maps, Gantt charts, and utilization plots.  

📖 [Read full manual](docs/README_subdivide.md)

---

### 2. Case Downscaling for Rapid Running (`resample_xml`)
Resample Digital Elevation Models (DEM) and associated ecohydrological input data (ASC/CSV/XML).

- **DEM methods:** `hydro-aware` (default) uses flow-accumulation-weighted
  block aggregation to emphasize dominant drainage paths; `mean`, `nearest`,
  `bilinear`, `burn-streams`, and `burn-breach` are also available.
- **Categorical rasters:** use `majority`, accumulation-weighted
  `hydro-aware`, or the distribution-matching `auto-weight` and
  `auto-reassign` methods for land cover, soils, and filter maps.
- **Continuous rasters:** use a block mean, a Hydro-aware accumulation-weighted
  mean, or land-cover/soil-aware averaging.
- Batch XML resampling with visualization support.

📖 [Read full manual](docs/README_resample.md) ·
[Technical formulation](docs/README_resample_details.md)

---

### 3. VELMA Parameter Optimization (`optimize_velma`)
Optimize parameters of the VELMA model with **Genetic Algorithm (GA)** and **Latin Hypercube Sampling (LHS)**.  
- Local or HPC (SLURM) batch runs.  
- Fitness metrics: **NSE**, **KGE**, with optional soft constraints.  
- XML parameter editing, DEM resampling, and logging/plotting.  

📖 [Read full manual](docs/README_optimize.md)

---

## Quick Start

### Installation
```bash
pip install numpy pandas matplotlib scipy deap scikit-optimize rasterio pysheds pyproj
```

Also required:
- **Java Application** (for VELMA)
- **SLURM** (if running on HPC)

---

### Example Workflow

```python
# --- Subdivide Watershed ---
from subdivide import subdivide_catchments
subdivide_catchments("dem.asc", col=257, row=32, num_processors=8, num_subbasins=100)

# --- Resample DEM ---
from resample import resample_xml
resample_xml("case.xml", "resampled", downscale_factor=5)

# --- Optimize VELMA Parameters ---
from optimize import VelmaModel, optimize_velma
model = VelmaModel(model_config)  # define config
best_param, best_score, best_var = optimize_velma("lhs", model, ngen=2, pop_size=8)
```

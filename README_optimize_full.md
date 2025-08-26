# User Manual: VELMA Parameter Optimization Toolkit

This tool provides a Python-based framework to **optimize parameters of the VELMA model** using either **Genetic Algorithm (GA)** or **Latin Hypercube Sampling (LHS)**.  
It supports both **local multiprocessing** and **HPC job arrays (SLURM)**, integrates **data resampling**, **soft constraints**, and **fitness evaluation** with NSE/KGE metrics.

---

## Features

- **VELMA Model Wrapper**
  - Automates XML parameter modifications.
  - Supports direct value replacement or scaling of parameters.
  - Runs simulations locally or via SLURM job arrays.

- **Data Preprocessing**
  - Outlier removal using Z-score.
  - Flexible computation of derived variables (`compute_var`).
  - Support for observational and simulation data alignment.

- **Fitness Evaluation**
  - Supports **NSE** (Nash–Sutcliffe Efficiency) and **KGE** (Kling–Gupta Efficiency).
  - Allows weighting of multiple calibration variables.
  - Soft constraints on state variables (e.g., humus pools, growth bounds).

- **Optimization Algorithms**
  - **Genetic Algorithm (GA)** with DEAP.
  - **Latin Hypercube Sampling (LHS)** with adaptive parameter range shrinking.

- **Resampling Support**
  - Optional DEM and XML input resampling using the `resample_xml` function.
  - Downscaling of input datasets for faster parameter testing.

- **Visualization & Logging**
  - Automatic plotting of observed vs. simulated variables.
  - Detailed logging of GA/LHS iterations, parameter ranges, and best solutions.

---

## Installation Requirements

### Python Dependencies
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `deap`
- `scikit-optimize`
- `concurrent.futures`
- `resample` (local module)

Install via pip:
```bash
pip install numpy pandas matplotlib scipy deap scikit-optimize
```

You also need:
- **Java Runtime** (to run VELMA `.jar` file).
- **SLURM** (if using HPC job submission).

---

## Configure Model Parameters

The `model_config` dictionary defines all settings required for optimization.  
Below is a description of each key:

- **`label`** *(str)*  
  A short name for the watershed or case study (used for log/output folder names).

- **`exe_file`** *(str)*  
  Path to the VELMA executable JAR file (e.g., `Velma.jar`).

- **`input_file`** *(str)*  
  Path to the base XML input configuration file for VELMA.

- **`obs_file`** *(str)*  
  Path to the observational dataset (CSV). The first column must be a valid date or year/day format.

- **`cali_var_names`** *(list of str)*  
  List of model output variables used for calibration (e.g., runoff, nitrate loss).

- **`weights`** *(list of float)*  
  Weighting factors applied to calibration variables when computing the final fitness score.

- **`soft_constraints`** *(dict)*  
  Optional additional constraints for specific state variables.  
  Format:  
  ```python
  "Variable_Name": [min_val, max_val, min_growth, max_growth, min_pct_growth, max_pct_growth]
  ```
  Example:  
  `Humus_Pool(gC/m2)_Delineated_Average: [10000, 25000, -200, 500, -0.1, 0.1]`  
  - Enforces bounds on value and growth of humus pool carbon.

- **`extra_var_names`** *(list of str)*  
  Variables included in output for reference but not used in calibration (e.g., flow observations).

- **`model_out_names`** *(list of str)*  
  Variables to extract from VELMA outputs (these must exist in `DailyResults.csv`).

- **`compute_var`** *(callable)*  
  Custom function to compute derived calibration variables from observations and simulations (e.g., nitrate loss = NO₃ concentration × runoff).

- **`metric`** *(str)*  
  Fitness metric:  
  - `"nse"` → Nash–Sutcliffe Efficiency.  
  - `"kge"` → Kling–Gupta Efficiency.

- **`param_ranges`** *(dict)*  
  Parameter search space for optimization.  
  Format: `{ "param_name": (min_value, max_value) }`.  
  Example:  
  ```python
  {
      "ks": (100, 400),
      "no3LossFraction": (0.1, 60)
  }
  ```

- **`param_type`** *(str)*  
  How parameters are updated in XML:  
  - `"value"` → direct replacement.  
  - `"scale"` → multiply the original value by a scaling factor.

- **`fixed_param`** *(dict)*  
  Parameters fixed at constant values (not optimized).  
  Example: `{'syear': 2010, 'eyear': 2010}` fixes simulation period.

- **`mem`** *(int)*  
  Java memory allocation in gigabytes (passed to `-Xmx`).

- **`downscale`** *(bool)*  
  If `True`, input DEM and XML files will be resampled before optimization.

- **`downscale_factor`** *(int or list)*  
  - If an integer: DEM is resampled by this factor.  
  - If a list: multiple downscale factors are tested, and the best configuration is selected.

- **`downscale_var_names`** *(list of str)*  
  Calibration variables used specifically during downscale evaluation.

- **`downscale_obs_file`** *(str)*  
  Observation file used for downscale evaluation.

- **`downscale_weights`** *(list of float)*  
  Variable weights used for downscale evaluation.

---

## Example Workflow

### Step 1: Configure Model
```python
label = 'Big_Beef'
input_file = f'{label}/XML/1.xml'

param_ranges = {
    "ks": (100, 400),
    "no3LossFraction": (0.1, 60)
}

model_config = {
    'label': label,
    'exe_file': 'Velma.jar',
    'input_file': input_file,
    'obs_file': f'{label}/obs.csv',
    'cali_var_names': [
        'Runoff_All(mm/day)_Delineated_Average',
        'NO3_Loss(gN/day/m2)_Delineated_Average'
    ],
    'weights': [1, 1],
    'soft_constraints': {
        "Humus_Pool(gC/m2)_Delineated_Average": [10000, 25000, -200, 500, -0.1, 0.1],
    },
    'extra_var_names': ['Result_Value_Flow'],
    'model_out_names': [
        'Runoff_All(mm/day)_Delineated_Average',
        'NO3_Loss(gN/day/m2)_Delineated_Average',
        'Humus_Pool(gC/m2)_Delineated_Average',
    ],
    'compute_var': compute_var,
    'metric': 'nse',
    'param_ranges': param_ranges,
    'param_type': 'scale',
    'fixed_param': {'syear': 2010, 'eyear': 2010},
    'mem': 2,
    'downscale': True,
    'downscale_factor': 8,
    'downscale_var_names': ['Runoff_All(mm/day)_Delineated_Average'],
    'downscale_obs_file': 'Big_Beef/obs.csv',
    'downscale_weights': [1.0]
}
model = VelmaModel(model_config)
```

### Step 2: Run Optimization
```python
best_param, best_score, best_var = optimize_velma(
    method='lhs',   # or 'ga'
    model=model,
    ngen=2,         # number of iterations/generations
    pop_size=8,     # sample/population size
    nproc=8,        # number of processors
    use_HPC=None,   # set to 'compute' for HPC
    show_plot=False
)
```

---

## Output Structure

```
opt_log_<label>/
    ga_optimization.log
    lhs_optimization.log
    slurm_job_array.sh
    input_files.txt

opt_output_<label>/
    ga_best_params.xml
    ga_best_result.csv
    ga_best_result_*.png
    lhs_best_params.xml
    lhs_best_result.csv
    lhs_best_result_*.png
```

---

## Notes & Recommendations

- Ensure **VELMA.jar** and XML files are accessible.  
- For HPC runs, edit `slurm_job_array.sh` with the correct account/partition/module.  
- Use **weights** to balance calibration variables.  
- Start with **LHS** for global exploration, then refine with **GA**.  
- Use `soft_constraints` to prevent unrealistic parameter solutions.  
- Check plots (`.png`) to validate fit quality.  

---

# Quick Start Guide: VELMA Parameter Optimization Toolkit

This is a simplified guide for running parameter optimization with VELMA. For the detailed manual, see [detailed manual](README_optimize_details.md).

---

## 1. Installation

```bash
pip install numpy pandas matplotlib scipy deap scikit-optimize
```

Requirements:
- **Java Runtime** (for VELMA `.jar` file)
- **SLURM** (if running on HPC)

---

## 2. Minimal Model Configuration

```python
from your_module import VelmaModel, optimize, compute_var

label = 'Big_Beef'
input_file = f'{label}/XML/1.xml'

param_ranges = {
    "ks": (100, 400),
    "no3LossFraction": (0.1, 60)
}

model_config = {
    'label': label,                     # watershed/case name
    'exe_file': 'Velma.jar',            # VELMA JAR file
    'input_file': input_file,           # base XML input
    'obs_file': f'{label}/obs.csv',     # observation CSV
    'cali_var_names': [                 # calibration variables
        'Runoff_All(mm/day)_Delineated_Average',
        'NO3_Loss(gN/day/m2)_Delineated_Average'
    ],
    'weights': [1, 1],                  # variable weights
    'metric': 'nse',                    # NSE or KGE
    'param_ranges': param_ranges,       # parameter search space
    'param_type': 'scale',              # update type
    'fixed_param': {'syear': 2010, 'eyear': 2010},  # fixed simulation years
    'mem': 2                            # memory in GB
}
model = VelmaModel(model_config)
```

---

## 3. Run Optimization

### LHS Example
```python
best_param, best_score, best_var = optimize_velma(
    method='lhs',     # or 'ga'
    model=model,
    ngen=2,           # number of iterations
    pop_size=8,       # samples per iteration
    nproc=8,          # local processors
    use_HPC=None,     # set to 'compute' for HPC
    show_plot=False
)
```

---

## 4. Output Structure

```
opt_log_<label>/
    lhs_optimization.log
    ga_optimization.log

opt_output_<label>/
    lhs_best_params.xml
    lhs_best_result.csv
    lhs_best_result_*.png
    ga_best_params.xml
    ga_best_result.csv
    ga_best_result_*.png
```

---

## 5. Tips

- Start with **LHS** for exploration.  
- Use **GA** for fine-tuning.  
- Adjust `weights` if some variables are more important.  
- Use `soft_constraints` only if needed to prevent unrealistic states.  

---

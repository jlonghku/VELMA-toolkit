import os, sys, uuid, shutil, subprocess, re, pprint
import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET
from deap import base, creator, tools, algorithms
from concurrent.futures import ProcessPoolExecutor
from skopt.sampler import Lhs
from skopt.space import Real
import matplotlib.pyplot as plt
from scipy.stats import zscore
from resample import resample_xml

class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.__stdout__
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        self.log = open(logfile, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def remove_outliers(obs, z_thresh=2.0, var_list=None):
    target_vars = var_list or obs.columns
    cleaned = {}

    for col in target_vars:
        if col not in obs.columns:
            continue
        series = obs[col].dropna()
        if len(series) > 2:
            z = zscore(series, nan_policy='omit')
            mask = np.abs(z) < z_thresh
            
            cleaned[col] = series[mask].reindex(obs.index)
        else:
            cleaned[col] = obs[col]

    return pd.DataFrame(cleaned, index=obs.index)


def compute_fit(obs, sim, var_names, metric, weights=None, extra_vars=None, soft_constraints=None,penalty_coeff=1000):
    errors, merged_list = [], []

    for var in var_names:
        
        has_obs = var in obs.columns and var in sim.columns
        has_sim = var in sim.columns
        error = 0.0

        merged = pd.DataFrame()
        if has_sim:
            merged[f'{var}'] = sim[var]

        # === compute NSE or KGE ===
        if has_obs:
            o, o_raw, s = obs[var], obs.get(f'{var}_raw', obs[var]), sim[var]
            idx_raw = o_raw.dropna().index.intersection(s.dropna().index)
            idx_clean = o.dropna().index.intersection(s.dropna().index)

            merged[f'{var}_obs'] = o.loc[idx_raw]
            merged[f'{var}_raw'] = o_raw.loc[idx_raw]
            merged[f'{var}'] = s.loc[idx_raw]


            o_valid, s_valid = o.loc[idx_clean], s.loc[idx_clean]
            if len(o_valid) >= 2 and len(s_valid) >= 2:
                if metric == 'kge':
                    r = np.corrcoef(s_valid, o_valid)[0, 1]
                    beta = s_valid.mean() / o_valid.mean()
                    gamma = s_valid.std() / o_valid.std()
                    error += 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
                elif metric == 'nse':
                    denom = np.sum((o_valid - o_valid.mean()) ** 2)
                    if denom > 0:
                        error += 1 - np.sum((s_valid - o_valid) ** 2) / denom

        # === Soft Constraint Penalty ===
        if soft_constraints and var in soft_constraints and has_sim:
            series = sim[var].dropna()
            if len(series) > 1:
                vals = soft_constraints[var]
                min_val, max_val, min_growth, max_growth = vals[:4]
                pct_growth = vals[4:6] if len(vals) >= 6 else None

                range_val = max(max_val - min_val, 1e-6)
                mean_val = series.mean() if series.mean() > 1e-6 else 1e-6
                error -= (penalty_coeff * (min_val - series.min()) / range_val/mean_val) ** 2 if series.min() < min_val else 0
                error -= (penalty_coeff * (series.max() - max_val) / range_val/mean_val) ** 2 if series.max() > max_val else 0

                dt = (series.index[-1] - series.index[0]).days
                if dt > 0:
                    avg_growth = (series.iloc[-1] - series.iloc[0]) / dt * 365.25
                    growth_range = max(max_growth - min_growth, 1e-6)
                    error -= (penalty_coeff * (min_growth - avg_growth) / growth_range) ** 2 if avg_growth < min_growth else 0
                    error -= (penalty_coeff * (avg_growth - max_growth) / growth_range) ** 2 if avg_growth > max_growth else 0

                    if pct_growth:
                        init_val = series.iloc[0]
                        if abs(init_val) > 1e-6:
                            pct = avg_growth / init_val
                            min_pct, max_pct = pct_growth
                            pct_range = max(max_pct - min_pct, 1e-6)
                            error -= (penalty_coeff * (min_pct - pct) / pct_range) ** 2 if pct < min_pct else 0
                            error -= (penalty_coeff * (pct - max_pct) / pct_range) ** 2 if pct > max_pct else 0
        errors.append(error)
        merged_list.append(merged)

    merged_all = pd.concat(merged_list, axis=1).dropna(how='all') if merged_list else pd.DataFrame()

    if extra_vars:
        extra_cols = []
        for var in extra_vars:
            if var in obs.columns:
                extra_cols.append(obs[[var]].rename(columns={var: f'{var}_obs'}))
            if var in sim.columns:
                sim_sub = sim.loc[obs.index.intersection(sim.index), [var]] if obs.empty else sim[[var]]
                extra_cols.append(sim_sub.rename(columns={var: f'{var}'}))
        if extra_cols:
            merged_all = pd.concat([merged_all] + extra_cols, axis=1)

    score = np.average(errors, weights=weights) if weights else np.nanmean(errors)
    return score, merged_all, errors

class VelmaModel:
    def __init__(self, config):
        self.config = config

    def modify_xml(self, params_dict=None, new_file=None):
        params_dict = params_dict or {}
        fixed_param = self.config.get('fixed_param', {})
        mode = self.config.get('param_type', 'value')  # 'value' or 'scale'

        tree = ET.parse(self.config['input_file'])
        root = tree.getroot()
        uid = str(uuid.uuid4())

        temp_dir = os.path.join(os.path.dirname(self.config['input_file']), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        new_file = new_file or os.path.join(temp_dir, f'{uid}.xml')

        # Step 1: Apply fixed parameters (direct replacement)
        for key, value in fixed_param.items():
            xpath = key if key.startswith('.') or '/' in key else f'.//{key}'
            for node in root.findall(xpath):
                node.text = str(value)

        # Step 2: Apply params_dict (based on mode)
        for key, value in params_dict.items():
            xpath = key if key.startswith('.') or '/' in key else f'.//{key}'
            for node in root.findall(xpath):
                if mode == 'scale':
                    try:
                        orig = float(node.text)
                        node.text = str(orig * value)
                    except (TypeError, ValueError):
                        pass
                else:
                    node.text = str(value)

        # Modify output path
        output_node = root.find('.//initializeOutputDataLocationRoot')
        if output_node is not None:
            output_node.text = output_node.text.rstrip('/') + f'/{uid}/'

        tree.write(new_file, xml_declaration=False)
        return new_file

    
    def resample_xml(self, lst):
        
        xml_files=[]
        tmp_file = self.config['input_file'].replace('.xml', '_tmp.xml')
        self.modify_xml(new_file=tmp_file)
        for downscale_factor in lst:
            xml_file=resample_xml(tmp_file, 'resampled', downscale_factor=downscale_factor, plot_dem=False, overwrite=True)     
            xml_files.append(xml_file)
        return xml_files

    
    def run_one(self, xml_file):
        print(f"[INFO] Running Velma simulation for {xml_file} on processor {os.getpid()}")
        log_path = os.path.abspath(f'opt_run_log/processor_{os.getpid()}.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        try:
            cmd = ['java', f'-Xmx{self.config["mem"]}g', '-cp', self.config['exe_file'],
                'gov.epa.velmasimulator.VelmaSimulatorCmdLine', xml_file]
            subprocess.run(cmd, check=True, stdout=open(log_path, 'w'), stderr=subprocess.STDOUT)
            with open(log_path) as f:
                for line in f:
                    if 'Output Data Location' in line:
                        out_dir = line.split('"')[1]
                        print(f"[INFO] Output data location: {out_dir} on processor {os.getpid()}")
                        sim_path = os.path.join(out_dir, 'DailyResults.csv')                 
                        sim = pd.read_csv(sim_path, usecols=['Year', 'Day'] + self.config['model_out_names'])
                        sim.index = pd.to_datetime(sim['Year'].astype(str) + sim['Day'].astype(str), format='%Y%j')
                        return sim, xml_file, out_dir, log_path
        except Exception as e:
            print(f"[ERROR] Failed to run {xml_file}: {e}")
        return None, xml_file, None, log_path  # always return fallback


    def run_batch(self, xml_files, nproc=4):
        with ProcessPoolExecutor(max_workers=nproc) as executor:
            return list(executor.map(self.run_one, xml_files))


    def run_batch_HPC(self, xml_files, wait=True,use_HPC='compute', nproc=40):
        file_list = f'opt_log_{self.config["label"]}/input_files.txt'
        with open(file_list, 'w') as f:
            f.write('\n'.join(xml_files))

        job_count = len(xml_files)
        script_path = f'opt_log_{self.config["label"]}/slurm_job_array.sh'
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=velma",
            f"#SBATCH --account=your_account",  # Replace with your account
            f"#SBATCH --partition={use_HPC}",
            f"#SBATCH --output=opt_run_log/slurm_%A_%a.out",
            f"#SBATCH --array=0-{job_count - 1}%{nproc}",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=1",
            "#SBATCH --time=3-5:00:00",
            "#SBATCH --mem=5G",
            "export LMOD_CACHED_LOADS=no",
            "module purge",
            "module load coenv/jdk",  # Replace with your module load command
            f'XML_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {file_list} | xargs)',
            f'java -Xmx{self.config["mem"]}g -cp {self.config["exe_file"]} gov.epa.velmasimulator.VelmaSimulatorCmdLine "$XML_FILE"',
        ]

        with open(script_path, 'w') as f:
            f.write('\n'.join(lines))

        result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] SLURM submit failed:\n{result.stderr}")
            return []

        job_id = result.stdout.strip().split()[-1]
        print(f"[INFO] Submitted job array ID: {job_id}")

        if wait:
            import time
            while True:
                check = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
                if job_id not in check.stdout:
                    break
                time.sleep(15)

        results = []
        for i, xml_file in enumerate(xml_files):
            log_file = f"opt_run_log/slurm_{job_id}_{i}.out"
            out_dir = None
            if os.path.exists(log_file):
                with open(log_file) as f:
                    for line in f:
                        if 'Output Data Location' in line:
                            out_dir = line.split('"')[1]
                            break
            sim_path = os.path.join(out_dir, 'DailyResults.csv')                 
            sim = pd.read_csv(sim_path, usecols=['Year', 'Day'] + self.config['model_out_names'])
            sim.index = pd.to_datetime(sim['Year'].astype(str) + sim['Day'].astype(str), format='%Y%j')
            results.append((sim, xml_file, out_dir, log_file))

        return results

    
    def evaluate_batch(self, lst, nproc=4, use_HPC=None, keep_files=False):

        if self.config.get('downscale', False):
            xml_files=self.resample_xml(lst)   
        else:
            keys = list(self.config['param_ranges'].keys())
            param_batch = [dict(zip(keys, ind)) for ind in lst]
            xml_files = [self.modify_xml(p) for p in param_batch]
        run_outputs = self.run_batch_HPC(xml_files, use_HPC=use_HPC, nproc=nproc) if use_HPC is not None else self.run_batch(xml_files, nproc=nproc)
        print(f"[INFO] Finished running {len(run_outputs)} simulations.")
        obs = raw = pd.DataFrame()
        obs_file = self.config.get('obs_file')
        has_obs = False

        if obs_file and os.path.isfile(obs_file) and os.path.getsize(obs_file) > 0:
            try:
                raw = pd.read_csv(obs_file)
                if not raw.empty:
                    raw.index = pd.to_datetime(raw.iloc[:, 0])
                    obs = raw
                    has_obs = True
            except Exception as e:
                print(f"[WARNING] Failed to read obs file: {obs_file}\n{e}")
        
        results = []

        for sim, xml_file, out_dir, log_path in run_outputs:
            try:
                print(f"[INFO] Processing results for {out_dir}")   
                if has_obs:        
                    raw_tmp = raw.loc[(raw.index >= sim.index.min()) & (raw.index <= sim.index.max())]
                    obs_tmp = obs.loc[(obs.index >= sim.index.min()) & (obs.index <= sim.index.max())]
                    sim = sim.loc[(sim.index >= raw.index.min()) & (sim.index <= raw.index.max())].copy()
                    if self.config.get('compute_var'):
                        raw_tmp, sim = self.config['compute_var'](raw_tmp, sim)
                        obs_tmp, sim = self.config['compute_var'](obs_tmp, sim) 
                        obs_tmp=remove_outliers(obs_tmp, var_list=self.config['cali_var_names'])
                    valid_vars = [v for v in self.config['cali_var_names'] if v in raw_tmp.columns]
                    raw_sel = raw_tmp[valid_vars].add_suffix('_raw')
                    obs_tmp = pd.concat([obs_tmp, raw_sel], axis=1)
                else:
                    obs_tmp = pd.DataFrame()
                          
                score, var, fit = compute_fit(obs_tmp, sim, self.config['cali_var_names'], self.config['metric'],soft_constraints=self.config.get('soft_constraints', None),
                                              weights=self.config.get('weights', None),extra_vars=self.config.get('extra_var_names', None))
                results.append([score, var, fit, xml_file])
                if not keep_files and log_path and os.path.exists(log_path):
                    os.remove(log_path)

            except Exception as e:
                print(f"[WARN] Failed to compute fitness for result: {out_dir} | Reason: {e}")
                results.append([float('-inf'), pd.DataFrame(), [float('-inf')]*len(self.config['cali_var_names']), xml_file])

            finally:
                if not keep_files:
                    if xml_file and os.path.exists(xml_file):
                        os.remove(xml_file)
                    if out_dir and os.path.exists(os.path.dirname(out_dir)):
                        print(f"[INFO] Removing output directory: {os.path.dirname(out_dir)}")
                        shutil.rmtree(os.path.dirname(out_dir), ignore_errors=True)

        
        return results # returns a list of [score, var_df, fit_list, xml_file] for each run
 

def plot_var(df, fitness, var_names, label, prefix='result', show=True):
    save_dir = f'opt_output_{label}'
    os.makedirs(save_dir, exist_ok=True)

    for i, var in enumerate(var_names):
        safe_var = re.sub(r'[\\/*?:"<>|]', "_", var)
        plt.figure()
        if f'{var}_raw' in df.columns:
            plt.scatter(df.index, df[f'{var}_raw'], label='Outlier', color='red', s=20)
        if f'{var}_obs' in df.columns:
            plt.scatter(df.index, df[f'{var}_obs'], label='Observed', color='black', s=20)
        plt.plot(df[f'{var}'].dropna(), label='Simulated', linewidth=1.5)
        plt.title(f'{var} (fitness = {fitness[i]:.3f})')
        plt.xlabel('Date')
        plt.ylabel(var)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_{safe_var}.png'), dpi=300)
        if show:
            plt.show()
        else:
            plt.close()


def ga_optimize(model, ngen, pop_size, nproc, use_HPC, show_plot=False):
    sys.stdout = Logger(f'opt_log_{model.config["label"]}/ga_optimization.log')
    param_ranges = model.config['param_ranges']

    if not creator.__dict__.get('FitnessSingle'):
        creator.create("FitnessSingle", base.Fitness, weights=(1.0,))  
        creator.create("Individual", list, fitness=creator.FitnessSingle)

    # Setup
    toolbox = base.Toolbox()
    dimensions = [Real(low, high) for low, high in param_ranges.values()]
    lhs = Lhs(criterion="maximin", iterations=200)
    lhs_samples = lhs.generate(dimensions, n_samples=pop_size)
    toolbox.register("population", lambda: [creator.Individual(ind) for ind in lhs_samples])
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=5.0,
                     low=list(zip(*param_ranges.values()))[0],
                     up=list(zip(*param_ranges.values()))[1],
                     indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initial population
    pop = toolbox.population()
    global_best_score = float('-inf')
    global_best_var = None
    global_best_fit = None
    global_best_param = None

    for gen in range(ngen + 1):
        # Evaluate
        if gen == 0:
            invalid_inds=pop
            results = model.evaluate_batch(pop, nproc=nproc, use_HPC=use_HPC)
            for ind, result in zip(pop, results):
                ind.fitness.values = (result[0],)
        else:
            offspring = algorithms.varOr(pop, toolbox, lambda_=pop_size, cxpb=0.8, mutpb=0.2)
            for ind in offspring:
                ind[:] = [max(min(x, high), low) for x, (low, high) in zip(ind, param_ranges.values())]
            invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
            results = model.evaluate_batch(invalid_inds, nproc=nproc, use_HPC=use_HPC)
            for ind, result in zip(invalid_inds, results):
                ind.fitness.values = (result[0],)
            pop = toolbox.select(pop + offspring, k=pop_size)

        # Find best in population
        sorted_combined = sorted(((*r, ind) for r, ind in zip(results, invalid_inds)), key=lambda x: x[0], reverse=True)
        sorted_scores, sorted_vars, sorted_fits, sorted_xmls, sorted_params = zip(*sorted_combined)

        # Best of current generation
        best_score= sorted_scores[0]
        best_var = sorted_vars[0]
        best_fit = sorted_fits[0]
        best_param = sorted_params[0]

        # Update global best
        if best_score > global_best_score:
            global_best_score = best_score
            global_best_var = best_var
            global_best_fit = best_fit
            global_best_param = best_param

        # Print and save
        print(f"GA_Gen: {gen} | Eval: {len(invalid_inds)} | "
              f"Best (current): {best_score:.4f}, components: {best_fit} | "
              f"Best (global): {global_best_score:.4f}, components: {global_best_fit}")
        
        # Save log
        with open(f'opt_output_{model.config["label"]}/ga_current.txt', 'w') as f:
            f.write(f"=== GA_Gen: {gen} ===\n")
            f.write(f"Global Best Score: {global_best_score:.4f}\n")
            f.write(f"Global Best Fitness: {global_best_fit}\n")
            f.write(f'Be Careful: the parameters type is {model.config.get("param_type","value")}.\n')
            params_str = ', '.join(f"{k}={v:.6g}" for k, v in zip(param_ranges, global_best_param))
            f.write(f"Global Best Param: {params_str}\n\n")
            current_ranges = {k: (np.min(col), np.max(col)) for k, col in zip(param_ranges.keys(), np.array(pop).T)}
            f.write("Current Param Ranges:\n")
            f.write(pprint.pformat(current_ranges) + "\n\n")        
            for score, fit, param in zip(sorted_scores, sorted_fits, sorted_params):
                param_str = ', '.join(f"{k}={v:.6g}" for k, v in zip(current_ranges.keys(), param))
                f.write(f"score={score:.4f} fit={fit} params={param_str}\n")
        
        # Save best result
        model.modify_xml(dict(zip(param_ranges.keys(), global_best_param)),
                         new_file=f'opt_output_{model.config["label"]}/ga_best_params.xml')
        global_best_var.to_csv(f'opt_output_{model.config["label"]}/ga_best_result.csv')
        plot_var(global_best_var, global_best_fit, model.config['cali_var_names'], model.config['label'], prefix='ga_best_result', show=show_plot)

    # Final output
    print("GA Optimization Final Best:")
    print(f"Best Fitness: {global_best_score:.4f}, components: {global_best_fit}")
    print(f'Be Careful: the parameters type is {model.config.get("param_type","value")}.')
    print("Best Parameters:", dict(zip(param_ranges.keys(), global_best_param)))   
    return global_best_param, global_best_score, global_best_var


def shrink_param_ranges(top_samples, param_ranges, shrink_buffer=0.05):
    top_samples = np.array(top_samples)
    new_ranges = {}
    for i, (k, (low, high)) in enumerate(param_ranges.items()):
        col = top_samples[:, i]
        full_range = high - low
        buffer = shrink_buffer * full_range
        new_min = max(low, np.min(col) - buffer)
        new_max = min(high, np.max(col) + buffer)
        new_ranges[k] = (new_min, new_max)
    return new_ranges


def lhs_optimize(model, n_iter, n_samples, nproc, use_HPC, show_plot=False, top_percent=0.3):
    sys.stdout = Logger(f'opt_log_{model.config["label"]}/lhs_optimization.log')
    param_ranges = model.config['param_ranges']
    current_ranges = param_ranges.copy()
    global_best_score = float('-inf')  
    global_best_var = None
    global_best_fit = None
    global_best_param = None

    for iteration in range(n_iter):
        # Generate LHS samples
        space = [Real(low, high) for low, high in current_ranges.values()]
        lhs = Lhs(criterion="maximin", iterations=200)
        samples = lhs.generate(space, n_samples)

        # Evaluate
        results = model.evaluate_batch(samples, nproc=nproc, use_HPC=use_HPC)
        top_k = int(np.ceil(top_percent * len(results)))
        sorted_combined = sorted(((*r, ind) for r, ind in zip(results, samples)), key=lambda x: x[0], reverse=True)
        sorted_scores, sorted_vars, sorted_fits, sorted_xmls, sorted_params = zip(*sorted_combined)
        top_params =sorted_params[:top_k]

        # Best of current iteration
        best_score= sorted_scores[0]
        best_var = sorted_vars[0]
        best_fit = sorted_fits[0]
        best_param = sorted_params[0]

        # Update global best
        if best_score > global_best_score:
            global_best_score = best_score
            global_best_var = best_var
            global_best_fit = best_fit
            global_best_param = best_param

        # Print current and global best
        print(f"LHS_Iter: {iteration+1} | Samples: {len(samples)} | "
              f"Best (current): {best_score:.4f}, components: {best_fit} | "
              f"Best (global): {global_best_score:.4f}, components: {global_best_fit}")

        # Save log
        with open(f'opt_output_{model.config["label"]}/lhs_current.txt', 'w') as f:
            f.write(f"=== LHS_Iter: {iteration+1} ===\n")
            f.write(f"Global Best Score: {global_best_score:.4f}\n")
            f.write(f"Global Best Fitness: {global_best_fit}\n")
            f.write(f'Be Careful: the parameters type is {model.config.get("param_type","value")}.\n')
            params_str = ', '.join(f"{k}={v:.6g}" for k, v in zip(param_ranges, global_best_param))
            f.write(f"Global Best Param: {params_str}\n\n")
            f.write("Current Param Ranges:\n")
            f.write(pprint.pformat(current_ranges) + "\n\n")       
            for score, fit, param in zip(sorted_scores, sorted_fits, sorted_params):
                param_str = ', '.join(f"{k}={v:.6g}" for k, v in zip(current_ranges.keys(), param))
                f.write(f"score={score:.4f} fit={fit} params={param_str}\n")

        # Save best result
        model.modify_xml(dict(zip(param_ranges.keys(), global_best_param)),
                         new_file=f'opt_output_{model.config["label"]}/lhs_best_params.xml')
        global_best_var.to_csv(f'opt_output_{model.config["label"]}/lhs_best_result.csv')
        plot_var(global_best_var, global_best_fit, model.config['cali_var_names'], model.config['label'], prefix='lhs_best_result', show=show_plot)

        # Keep global best in range
        if best_score < global_best_score:
            top_params = np.vstack([top_params, global_best_param])
        current_ranges = shrink_param_ranges(top_params, current_ranges)

    # Final output
    print("LHS Optimization Final Best:")
    print(f"Best Fitness: {global_best_score:.4f}, components: {global_best_fit}")
    print(f'Be Careful: the parameters type is {model.config.get("param_type","value")}.')
    print("Best Parameters:", dict(zip(param_ranges.keys(), global_best_param)))
    return global_best_param, global_best_score, global_best_var


def optimize_velma(method, model, ngen=5, pop_size=16, nproc=4, use_HPC=None, show_plot=True):
    os.makedirs(f'opt_log_{model.config["label"]}', exist_ok=True)
    os.makedirs(f'opt_output_{model.config["label"]}', exist_ok=True)
    
    if model.config.get('downscale', False):
        if isinstance(model.config['downscale_factor'], (int, float)):
            print(f"Downscaling with factor: {model.config['downscale_factor']}")
            model.config['input_file'] = model.resample_xml([model.config['downscale_factor']])[0]
            print(f"Downscaling complete. New input file: {model.config['input_file']}")
        elif isinstance(model.config['downscale_factor'], list):
            tmp_dir = model.config['obs_file']
            tmp_var_names = model.config['cali_var_names']
            tmp_weights = model.config['weights']
            model.config['obs_file'] = model.config['downscale_obs_file']
            model.config['cali_var_names'] = model.config['downscale_var_names']
            model.config['weights'] = model.config['downscale_weights']
            lst = list(model.config['downscale_factor'])
            print(f"Downscaling with factors: {lst}")
            results = model.evaluate_batch(lst, nproc=nproc, use_HPC=use_HPC,keep_files=True)
            results.sort(key=lambda x: x[0], reverse=True)
            scores, vars, fits, xml_files = map(list, zip(*results))
            print(f"Downscaling complete. Best XML: {xml_files[0]} with fitness {fits[0]:.4f}")
            model.config['input_file'] = xml_files[0]
            model.config['obs_file']=tmp_dir
            model.config['cali_var_names'] = tmp_var_names 
            model.config['weights'] = tmp_weights
        else:
            raise ValueError("Invalid downscale_factor configuration. Must be a single value or a list.") 
        model.config['downscale'] = False

    methods = {
        'ga': ga_optimize,
        'lhs': lhs_optimize
    }
    if method not in methods:
        raise ValueError(f"Unsupported method: {method}")
    return methods[method](model, ngen, pop_size, nproc, use_HPC, show_plot)

def compute_var(obs, sim):
    # define your computed variables here
    obs=obs.copy()
    obs['NO3_Loss(gN/day/m2)_Delineated_Average'] = obs['Result_Value_Nitrate + Nitrite as N']*sim['Runoff_All(mm/day)_Delineated_Average']/1000
    obs['Runoff_All(mm/day)_Delineated_Average']= obs['Result_Value_Flow']
    return obs, sim


if __name__ == '__main__':
    # Set parameters
    label = 'Big_Beef'
    downscale_factor = 8
    input_file = f'{label}/XML/1.xml'

    # Define parameter search ranges
    param_ranges = {
        "ks": (100, 400),
        "no3LossFraction": (0.1, 60)
    }

    # Model configuration
    model_config = {
        'label': label,
        'exe_file': 'Velma.jar',
        'input_file': input_file,
        'obs_file': f'{label}/obs.csv',
        'cali_var_names': [
            'Runoff_All(mm/day)_Delineated_Average',
            'NO3_Loss(gN/day/m2)_Delineated_Average'
            ],
        'weights': [1,1],
        "soft_constraints":{
            "Humus_Pool(gC/m2)_Delineated_Average":         [10000, 25000, -200, 500, -0.1, 0.1],   # [min_val, max_val, min_growth, max_growth, min_pct_growth, max_pct_growth]
        },
        'extra_var_names': ['Result_Value_Flow'],
        'model_out_names': [
            'Runoff_All(mm/day)_Delineated_Average',         
            'NO3_Loss(gN/day/m2)_Delineated_Average',            
            'Humus_Pool(gC/m2)_Delineated_Average',
        ],
        'compute_var':compute_var,
        'metric': 'nse',
        'param_ranges': param_ranges,  
        'param_type':'scale',
        'fixed_param': {'syear': 2010, 'eyear': 2010},
        'mem': 2, 
        'downscale': True,  # Set to True if using downscaling
        'downscale_factor': downscale_factor,  # Downscaling factor if downscale is True
        'downscale_var_names': ['Runoff_All(mm/day)_Delineated_Average'], 
        'downscale_obs_file': f'{label}/obs.csv',
        'downscale_weights': [1.0],  # Weights for downscaled variables
    }
    model = VelmaModel(model_config)

    # Optimization settings
    method = 'lhs'          # 'ga' or 'lhs'
    ngen = 2               # number of generations/iterations
    pop_size = 8           # number of individuals/samples
    nproc = 8              # number of processors 
    use_HPC = None          # None for local run or 'compute' for using job array

    # Run optimization
    optimize_velma(method, model, ngen, pop_size, nproc=nproc, use_HPC=use_HPC, show_plot=False)



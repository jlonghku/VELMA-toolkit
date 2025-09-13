import numpy as np
from pysheds.grid import Grid
import matplotlib.pyplot as plt
from pyproj import Proj
from matplotlib.colors import ListedColormap
from collections import deque

def sort_target_points(target_points):
    # Extract dependencies
    reverse_dependencies = [set() for _ in range(len(target_points))]
    for task_id, point in enumerate(target_points):
        for dep in point[5]:
            reverse_dependencies[dep].add(task_id)

    def calculate_branch_time(task_id, memo, path_tracker):
        """
        Recursively calculate the branch time for a single task and track the longest branch path.
        """
        if task_id in memo:  # Use memoized result if available
            return memo[task_id], path_tracker[task_id]

        # Current task duration
        branch_time = target_points[task_id][4]
        longest_path = [task_id]  # Initialize the path with the current task

        # If the task has dependent tasks, calculate the maximum branch time for those tasks
        if reverse_dependencies[task_id]:
            max_time = 0
            max_path = []
            for dep in reverse_dependencies[task_id]:
                dep_time, dep_path = calculate_branch_time(dep, memo, path_tracker)
                if dep_time > max_time:
                    max_time = dep_time
                    max_path = dep_path
            branch_time += max_time
            longest_path.extend(max_path)

        # Memoize results
        memo[task_id] = branch_time
        path_tracker[task_id] = longest_path
        return branch_time, longest_path

    # Memoization
    branch_times = {}
    path_tracker = {}

    # Calculate branch times and track paths for all tasks
    for i in range(len(target_points)):
        calculate_branch_time(i, branch_times, path_tracker)

    # Find the longest branch
    longest_branch_task_id = max(branch_times, key=branch_times.get)
    longest_branch_path = path_tracker[longest_branch_task_id]

    # Find the task with the longest duration in the longest branch
    max_duration = 0
    longest_task_id = None
    for task_id in longest_branch_path:
        if target_points[task_id][4] > max_duration:
            max_duration = target_points[task_id][4]
            longest_task_id = task_id

    # Get the basin_id of the task with the maximum duration
    longest_task_basin_id = target_points[longest_task_id][0]

    # Sort all tasks by branch time (descending order)
    sorted_indices = sorted(range(len(target_points)), key=lambda x: branch_times[x], reverse=True)
    adjusted_target_points = [target_points[i] for i in sorted_indices]

    return adjusted_target_points, longest_task_basin_id


def simulate_task_execution(n_processors, target_points, is_plot=False,fig_name=None):
    # Initialize task statuses (0: pending, 1: waiting, 2: running, 3: completed)
    statuses = [0] * len(target_points)
    dependencies = [set(point[5]) for point in target_points]
    processors = [[] for _ in range(n_processors)]  # Processor task schedules
    waiting_list = []  # Waiting task queue
    current_time = 0  # Current time
    active_tasks = []  # Tasks currently running


    # Simulate task execution
    while any(status != 3 for status in statuses):  # Loop until all tasks are completed
        # Add tasks with no dependencies to the waiting queue
        for i, (status, deps) in enumerate(zip(statuses, dependencies)):
            if status == 0 and not deps:  # If task is pending and has no dependencies
                statuses[i] = 1  # Mark as waiting
                waiting_list.append(i)

        # Assign tasks to available processors
        for processor in range(n_processors):
            if not processors[processor] or processors[processor][-1][2] <= current_time:
                if waiting_list:
                    task_id = waiting_list.pop(0)
                    start_time = current_time
                    end_time = start_time + target_points[task_id][4]  # Task duration
                    processors[processor].append((task_id, start_time, end_time))
                    active_tasks.append((task_id, end_time))
                    statuses[task_id] = 2  # Mark as running

        # Advance time to the next task completion
        if active_tasks:
            next_end_time = min(task[1] for task in active_tasks)
            current_time = next_end_time

            # Complete tasks and update dependencies
            completed_tasks = [task for task in active_tasks if task[1] <= current_time]
            for task in completed_tasks:
                task_id = task[0]
                active_tasks.remove(task)
                statuses[task_id] = 3  # Mark as completed
                for i, deps in enumerate(dependencies):
                    basin_id = target_points[task_id][0]
                    if basin_id in deps:
                        deps.remove(basin_id)

    # Calculate makespan (total execution time)
    makespan = max(task[2] for processor in processors for task in processor)

    # Calculate average processor utilization
    average_utilization = sum([i[4] for i in target_points]) /n_processors/ makespan

    # Plot Gantt chart if required
    if is_plot:
        plt.figure(figsize=(10, 6))
        tab20 = plt.colormaps['tab20']

        for i, processor in enumerate(processors):
            for task in processor:
                task_id, start_time, end_time = task
                basin_id = target_points[task_id][0]
                plt.barh(i, end_time - start_time, left=start_time, color=tab20(basin_id% tab20.N), edgecolor='black')
                plt.text(start_time + (end_time - start_time) / 2, i, f"{basin_id+1}", va='center', ha='center', color='white')

        plt.yticks(range(n_processors), [f"Processor {i+1}" for i in range(n_processors)])
        plt.xlabel(f"Time")
        plt.title(f"Task Execution Timeline (Makespan: {makespan}) ({fig_name.split('.')[0]})")
        plt.tight_layout()
        plt.savefig(fig_name, dpi=300) if fig_name is not None else None
        plt.show()
    print(f"Basin numbers: {len(target_points)}")
    print(f"Makespan: {makespan}")
    print(f"Average processor utilization: {average_utilization:.2f}")
    return makespan, average_utilization 


def build_dependencies(target_points, grid, fdir):
    """
    Build a list of dependencies between target points based on their sub-catchment areas.

    Parameters:
    - target_points: List of target points, each in the format [id, col, row, index, cells].
    - grid: The grid object used for calculating catchments.
    - fdir: Flow direction data for the grid.

    Returns:
    - dependency_list: A list of target points with their direct dependencies included.
    """
    dependency_list = []  # Store all dependencies
    dependency_dict = {}  # Record direct dependencies for each point by ID

    for target_point in target_points:
        basin_id, col, row, index, cells = target_point

        # Compute the sub-catchment mask for the current target point
        sub_catchment = grid.catchment(x=col, y=row, fdir=fdir, xytype='index')
        sub_catchment_mask = grid.view(sub_catchment)

        # Initialize dependencies for the current sub-catchment
        dependencies = set()

        # Identify dependencies by checking if other points fall within the sub-catchment
        for other_point in target_points:
            other_id, other_col, other_row, other_index, other_cells = other_point

            if sub_catchment_mask[other_row, other_col] == 1 and other_id != basin_id:
                dependencies.add(other_id)

        # Remove indirect dependencies from the list
        direct_dependencies = dependencies.copy()
        for dep in dependencies:
            if dep in dependency_dict:  # Check if the dependency has other dependencies
                direct_dependencies -= set(dependency_dict[dep])

        # Update the dependency dictionary and result list
        dependency_dict[basin_id] = list(direct_dependencies)
        dependency_list.append([basin_id, col, row, index, cells, list(direct_dependencies)])

    return dependency_list

def plot_utilization(opt_info):
    # Sort opt_info by the first column (Basin number)
    opt_info = sorted(opt_info)
    
    # Extract sorted data for plotting
    basins = [item[0] for item in opt_info]
    makespan = [item[1] for item in opt_info]
    utilization = [item[2] for item in opt_info]
    
    # Create a plot
    fig, ax1 = plt.subplots()
    
    # Plot Makespan
    ax1.plot(basins, makespan, marker='o', linestyle='-', color='r', label='Makespan')
    ax1.set_xlabel('Basin number')
    ax1.set_ylabel('Makespan', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    
    # Plot Average Utilization on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(basins, utilization, marker='o', linestyle='-', color='b', label='Average Utilization')
    ax2.set_ylabel('Average Utilization', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    # Add grid, title, and show the plot
    plt.title('Basin number vs Average Utilization and Makespan')
    fig.tight_layout()
    plt.grid(True)
    plt.show()

def plot_basin(basin,acc=None, cmap='Blues'):
    plt.figure(figsize=(6, 6))
    plt.imshow(basin, cmap='tab20')  
    if acc is not None: 
        plt.imshow(np.ma.masked_where(acc == 0, acc), cmap='binary', alpha=0.6, interpolation='nearest')
    plt.colorbar(label="Region Index")
    plt.title(f"Basins")
    plt.show()

def plot_branch(branch, threshold=None):
    
    if threshold is not None:
        branch = np.where(branch > threshold, branch, 0)

    plt.figure(figsize=(8, 6))
    plt.imshow(branch, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Branch Order')
    plt.title(f"Branch Order Map (Threshold: {threshold})")
    plt.axis('off')
    plt.show()
    
def build_branch(fdir, threshold=None, plot=False):
    # Initialize branch map
    branch = np.zeros_like(fdir, dtype=int)

    # Define D8 flow direction offsets
    offsets = {
        1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
        16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)
    }

    # Calculate in-degrees and build downstream map
    in_degree = np.zeros_like(fdir, dtype=int)
    downstream_map = {}
    for r, c in np.argwhere(fdir > 0):
        dr, dc = offsets[fdir[r, c]]
        nr, nc = r + dr, c + dc
        if fdir[nr, nc] != 0:
            in_degree[nr, nc] += 1
            downstream_map.setdefault((r, c), []).append((nr, nc))

    # Initialize queue with cells having in-degree 0
    queue = deque(np.argwhere((fdir > 0) & (in_degree == 0)))
    for r, c in queue:
        branch[r, c] = 1  # Start branch order at 1

    # Topological sorting to calculate branch order
    while queue:
        r, c = queue.popleft()
        if (r, c) in downstream_map:
            for nr, nc in downstream_map[(r, c)]:
                # Update branch order
                branch[nr, nc] = (
                    branch[r, c] + 1 if branch[nr, nc] == branch[r, c] else max(branch[nr, nc], branch[r, c])
                )
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:  # Add downstream cell to queue
                    queue.append((nr, nc))

    # Plot the result if requested
    if plot:
        plot_branch(branch, threshold)

    return branch

def find_upstream(acc, fdir,threshold=None,return_value=False,return_main=False,tmp_acc=None):
    
    d8_offsets = {
        1: (0, 1),      
        2: (1, 1),      
        4: (1, 0),      
        8: (1, -1),     
        16: (0, -1),    
        32: (-1, -1),  
        64: (-1, 0),   
        128: (-1, 1)    
    }
    if threshold is None:
        threshold=acc.max()
        step=max(int(threshold*0.01), 1)
        
        while True:   
            candidates = np.argwhere(acc >= threshold)
            candidates_set = set(map(tuple, candidates)) 

            best_upstream_points = []
            best_criterion_value = 0   

            
            for r, c in candidates:
                current_upstreams = []
                
                for direction, (dr, dc) in d8_offsets.items():
                    nr, nc = r - dr, c - dc   
                    if (nr, nc) in candidates_set and fdir[nr, nc] == direction:
                        current_upstreams.append((acc[nr, nc], [nr, nc]))

                
                if len(current_upstreams) > 1:
                    current_upstreams.sort(reverse=True)
                    
                    if current_upstreams[1][0] > best_criterion_value:
                        best_criterion_value = current_upstreams[1][0]
                        best_upstream_points = [item[1] for item in current_upstreams]
                    
            threshold -= step
            if best_upstream_points:
                break
            if threshold <= 0:
                return []
    else:
        candidates = np.argwhere(acc >= threshold)
        candidates_set = set(map(tuple, candidates)) 
        
        upstream_points = []
       
        
        for r, c in candidates: 
            current_upstreams = []        
            for direction, (dr, dc) in d8_offsets.items():
                nr, nc = r - dr, c - dc   
                if (nr, nc) in candidates_set and fdir[nr, nc] == direction:
                    acc_value=acc[nr, nc] if not return_main else tmp_acc[nr,nc] 
                    current_upstreams.append((acc_value, [nr, nc])) 
            if return_main: 
                if len(current_upstreams) >= 1:   
                    current_upstreams.sort(reverse=True)
                    upstream_points.append(current_upstreams[0]) 
            else:       
                if len(current_upstreams) > 1:
                    upstream_points.extend(current_upstreams)
        upstream_points.sort()
        best_upstream_points = [item[1] for item in upstream_points]
        best_criterion_value = upstream_points[0][0] if upstream_points else 0.0
        
        
    return best_upstream_points if not return_value else best_criterion_value 

def find_downstream(fdir,target_points,main_stream_points):
    d8_offsets = {
        1: (0, 1),      
        2: (1, 1),      
        4: (1, 0),      
        8: (1, -1),     
        16: (0, -1),    
        32: (-1, -1),  
        64: (-1, 0),   
        128: (-1, 1)    
    }
    r,c=target_points
    dr,dc=d8_offsets[fdir[r,c]]
    nr=r+dr
    nc=c+dc
   
    return True if [nr,nc] in main_stream_points else False
        
    
    
def subdivide_catchments(asc_file, col, row, num_processors, num_subbasins, method='layer', crs="EPSG:26910", is_plot=False,target_size=None,save_dir=None):
    # Initialize the Grid object and add DEM data
    grid = Grid.from_ascii(asc_file, crs=Proj(crs))
    dem = grid.read_ascii(asc_file, crs=Proj(crs))
    
    # Fill pits and depressions in the DEM
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)
    
    # Compute flow direction and accumulation
    fdir = grid.flowdir(inflated_dem)
    acc = grid.accumulation(fdir)
    
    # Extract the main catchment
    main_catchment = grid.catchment(x=col, y=row, fdir=fdir, xytype='index')
    catchment_mask = grid.view(main_catchment)
    
    # Initialize variables
    catchment_cells = np.sum(catchment_mask)
    subbasins = np.zeros_like(acc, dtype=int)
    colmax=subbasins.shape[1]
    basin_id = 1
    target_points = []  # List to store all target points
       
    if method == 'layer':
        opt_info = []
        start_size=np.sum(catchment_mask) // num_processors 
        max_target_size=int(1.5*start_size) if target_size is None else target_size
        target_size = int(0.8*start_size) if target_size is None else target_size
        size_step=int(0.05*target_size)
        best_makespan=np.inf
        best_target_points=[]
        best_subbasins=[]
        # Binary search to optimize last_area
        while target_size<=max_target_size:       
            basin_id = 1           
            points=find_upstream(acc*catchment_mask, fdir*catchment_mask,threshold=target_size)
            points.append([row,col])
            
            tmp_catchment_mask = catchment_mask.copy()  # Copy of the current catchment mask
            subbasins = np.zeros_like(acc, dtype=int)  # Array to store subbasin IDs
            target_points = []  # List to store target points
            remaining_cells = int(np.sum(tmp_catchment_mask))  # Remaining unprocessed cells

            while remaining_cells > 0:
                # Find the point with accumulation closest to the target size within the remaining catchment
                acc_masked = acc * tmp_catchment_mask
                
                target_point = np.unravel_index(
                    np.argmax(np.where(acc_masked < target_size, acc_masked, -np.inf)), acc_masked.shape
                )
                tmp_acc=grid.accumulation(fdir*tmp_catchment_mask)
                main_stream_points=find_upstream(acc_masked,fdir*tmp_catchment_mask,threshold=target_size,return_main=True,tmp_acc=tmp_acc)
                tmp_tmp_catchment_mask=tmp_catchment_mask.copy()
                while not find_downstream(fdir,target_point,main_stream_points):
                    tmp_catchment=grid.catchment(x=target_point[1], y=target_point[0], fdir=fdir, xytype="index")
                    tmp_mask=grid.view(tmp_catchment)
                    tmp_tmp_catchment_mask[tmp_mask==1]=0
                    tmp_acc_masked=acc*tmp_tmp_catchment_mask
                    target_point = np.unravel_index(
                    np.argmax(np.where(tmp_acc_masked < target_size, tmp_acc_masked, -np.inf)), tmp_acc_masked.shape
                )
                    
                
                # Extract the catchment (all upstream cells of the selected point)
                sub_catchment = grid.catchment(x=target_point[1], y=target_point[0], fdir=fdir, xytype="index")
                sub_catchment_mask = grid.view(sub_catchment)
                
                # Mark the sub-catchment as a new subbasin
                subbasins[sub_catchment_mask == 1] = basin_id
                tmp_catchment_mask[sub_catchment_mask == 1] = 0  # Update mask
                remaining_cells = int(np.sum(tmp_catchment_mask))  # Update remaining cells
                            
                # Add the target point to the list
                current_cells = int(np.sum(sub_catchment_mask))  # Number of cells in the subbasin
                target_points.append(
                    [basin_id - 1, target_point[1], target_point[0], target_point[0] * subbasins.shape[1] + target_point[1], current_cells])
                
                if basin_id == num_subbasins-len(points):                                 
                    
                    for target_point in points:
                        basin_id += 1
                        sub_catchment = grid.catchment(x=target_point[1], y=target_point[0], fdir=fdir*tmp_catchment_mask, xytype='index')
                        sub_catchment_mask = grid.view(sub_catchment)
                                    
                        subbasins[sub_catchment_mask == 1] = basin_id
                        
                        tmp_catchment_mask[sub_catchment_mask == 1] = 0
                        
                        target_points.append(
                            [basin_id - 1, target_point[1], target_point[0], target_point[0] * colmax + target_point[1], int(np.sum(sub_catchment_mask))]
                        )
                    break                                                  
                basin_id += 1

            # Recalculate flow direction for the remaining catchment
            fdir = grid.flowdir(inflated_dem)
            # Simulate task execution to evaluate makespan and average utilization
            target_points = build_dependencies(target_points, grid, fdir)
            print(f'Target size: {target_size}')
            makespan, average_utilization = simulate_task_execution(num_processors, target_points)
                
            #opt_info.append([basin_id, makespan, average_utilization])
            # Update binary search range based on the result
            if makespan <= best_makespan:
                best_makespan=makespan  
                best_target_points=target_points
                best_subbasins=subbasins 
            
            target_size+=size_step
        target_points=best_target_points  
        subbasins=best_subbasins  

        # Visualize the utilization of different configurations
        #plot_utilization(opt_info)
        
    if method == 'branch1':
        acc=build_branch(fdir*catchment_mask)
        all_points=[]
        all_points.append([basin_id-1,col,row,row * colmax + col])
        subbasins[catchment_mask == 1] = basin_id
        basin_id+=1
        makespan=0
        bottleneck_basin_id=0
        opt_info=[]
        while basin_id <=num_subbasins: 
            mask = subbasins == bottleneck_basin_id +1                    
            points=find_upstream(acc*mask, fdir*mask) 
            if len(points)==0:
                break
            tmp_fdir=fdir.copy()
            for target_point in points:
                sub_catchment = grid.catchment(x=target_point[1], y=target_point[0], fdir=tmp_fdir, xytype='index')
                sub_catchment_mask = grid.view(sub_catchment)
                combined_mask = mask & sub_catchment_mask           
                subbasins[combined_mask == 1] = basin_id
                all_points.append([basin_id-1, target_point[1], target_point[0], target_point[0] * colmax + target_point[1]])
                basin_id+=1
                if basin_id ==num_subbasins+1:
                    break                 
            target_points = [point + [int(np.sum(subbasins == point[0]+1))] for point in all_points]
            fdir = grid.flowdir(inflated_dem)
            target_points=build_dependencies(target_points, grid, fdir)
            target_points,bottleneck_basin_id=sort_target_points(target_points)
            makespan, average_utilization=simulate_task_execution(num_processors, target_points)
            opt_info.append([basin_id,makespan,average_utilization]) 
        plot_utilization(opt_info)

    if method == 'branch':
        all_points=[]
        all_points.append([basin_id-1,col,row,row * colmax + col])
        subbasins[catchment_mask == 1] = basin_id
        basin_id+=1
        makespan=0
        bottleneck_basin_id=0
        opt_info=[]
        while basin_id <=num_subbasins: 
            mask = subbasins == bottleneck_basin_id +1                    
            points=find_upstream(acc*mask, fdir*mask) 
            if len(points)==0:
                break
            tmp_fdir=fdir.copy()
            for target_point in points:
                sub_catchment = grid.catchment(x=target_point[1], y=target_point[0], fdir=tmp_fdir, xytype='index')
                sub_catchment_mask = grid.view(sub_catchment)
                combined_mask = mask & sub_catchment_mask           
                subbasins[combined_mask == 1] = basin_id
                all_points.append([basin_id-1, target_point[1], target_point[0], target_point[0] * colmax + target_point[1]])
                basin_id+=1
                if basin_id ==num_subbasins+1:
                    break                 
            target_points = [point + [int(np.sum(subbasins == point[0]+1))] for point in all_points]
            fdir = grid.flowdir(inflated_dem)
            target_points=build_dependencies(target_points, grid, fdir)
            target_points,bottleneck_basin_id=sort_target_points(target_points)
            makespan, average_utilization=simulate_task_execution(num_processors, target_points)
            opt_info.append([basin_id,makespan,average_utilization]) 
        plot_utilization(opt_info)

    if method == 'equal':
        target_size = catchment_cells // num_subbasins
        remaining_cells = np.sum(catchment_mask) 
        while basin_id <= num_subbasins and remaining_cells > 0:
            # Find the point with accumulation closest to the target size within the remaining catchment
            acc_masked = acc * catchment_mask
            target_point = np.unravel_index(np.argmin(np.abs(acc_masked - target_size)), acc_masked.shape)
            
            # Extract the catchment (all upstream cells of the selected point)
            sub_catchment = grid.catchment(x=target_point[1], y=target_point[0], fdir=fdir, xytype='index')
            sub_catchment_mask = grid.view(sub_catchment)
            
            # Mark the sub-catchment as a new subbasin
            subbasins[sub_catchment_mask == 1] = basin_id
           
            # Update the remaining catchment mask
            catchment_mask[sub_catchment_mask == 1] = 0
            
            # Update the accumulation grid
            fdir = fdir * catchment_mask
            acc = grid.accumulation(fdir)
           
            # Update remaining cells  
            remaining_cells = np.sum(catchment_mask)        
            if basin_id == num_subbasins and remaining_cells>0: 
                target_points.pop()  
                basin_id -=1    
                             
            target_size= remaining_cells // (num_subbasins - basin_id) if  num_subbasins>basin_id else 0 
            # Add the target point to the list      
            target_points.append([basin_id-1, target_point[1], target_point[0], target_point[0] * colmax + target_point[1], int(np.sum(sub_catchment_mask))])
            print(f"Target point: {target_point}, Subbasin cells: {np.sum(sub_catchment_mask)}, Remaining cells: {remaining_cells}")
            basin_id += 1 
        fdir = grid.flowdir(inflated_dem)
        target_points=build_dependencies(target_points, grid, fdir)
    
    if is_plot:
        # Visualize the subbasins
        plt.figure(figsize=(10, 8))
        num_colors=subbasins.max()
        plt.title(f"Divided into {num_colors} Subbasins ({method})", fontsize=16)
        
        
        tab20 = plt.colormaps['tab20']
        tab20_colors =  [tab20(i%tab20.N) for i in range(num_colors)]
        
        custom_cmap = ListedColormap(tab20_colors)
        plt.imshow(np.ma.masked_less_equal(subbasins, 0), cmap=custom_cmap , interpolation='nearest')
        # Mark target points and their indices
        for point in target_points:
            id, c, r, index, cells,_ = point           
            plt.plot(c, r, 'bo')  
            #plt.text(c, r, f'{index}', color='black', fontsize=8, ha='left', va='top')
        
        # Mark the main outlet point
        plt.plot(col, row, 'ro', label='Main Outlet')
        
        plt.legend()
        plt.colorbar(label='Subbasin ID',boundaries=np.arange(0.5, num_colors + 1.5), ticks=np.arange(2, num_colors + 1, 2))
        save_dir = save_dir or "."
        plt.savefig(f'{save_dir}/subbasins_{method}.png', dpi=300)
        plt.show()
    
    # Print all target points with indices
    indexes=[]
    print("\nAll Target Points with Indices:")
    for point in target_points:
        id, c, r, index, cells,_ = point
        indexes.append(index)
        print(f"ID {id}: Target Point x={c}, y={r}, Index: {index}, Cells: {cells}")
    print("All indices: ")    
    print(" ".join(map(str, indexes)))
    save_dir = save_dir or "."
    simulate_task_execution(num_processors, target_points,is_plot=True,fig_name=f'{save_dir}/Gantt_Chart_{method}.png')
    return " ".join(map(str, indexes))

if __name__=='__main__':
    # Example usage
    crs = "EPSG:26910"
    asc_file_path = 'Big_Beef/DataInputs30m/m_1_DEM/Big_Beef_DEM30m_dredgemask_EEX.asc'
    col, row = 257, 32  # Main outlet coordinates
    num_processors =8 # Number of processors
    num_subbasins=50 # Divide into 100 subbasins
    subdivide_catchments(asc_file_path, col, row, num_processors, num_subbasins, method='layer', crs=crs, is_plot=True)
    
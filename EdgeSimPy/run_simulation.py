import random
import numpy as np
from tqdm import tqdm 
# Ensure the import path is correct for your environment
from edge_sim_py.central_controller.controller import LyapunovPSOScheduler 
from edge_sim_py.components.edge_server import EdgeServer
from utils import parse_variables

# --- 1. Load Simulation Parameters ---
input_file = 'variables.txt'
variables = parse_variables(input_file)

NUM_NODES = variables['num_node']
F_initial = np.array(variables['F']) 
R = np.array(variables['R'])
C = variables['C']
L = variables['L']
CommCost = np.array(variables['CommCost'])
V = variables.get('V', 1000)
E_avg = variables.get('E_avg', 20)
arrived_lists = variables.get('arrived_lists', None)

# Determine Simulation Length (T) based on available workload data
T_SIM = len(arrived_lists) if arrived_lists else 100 

print("Loaded parameters from variables.txt")
print(f"Nodes: {NUM_NODES}, Simulation Length (T): {T_SIM} slots")
print(f"CPU Capacities (F_initial): {F_initial}")
print(f"Control Parameters: V={V}, E_avg={E_avg}")

# --- 2. Initialize EdgeServer Agents ---
edge_servers = {
    i: EdgeServer(
        obj_id=i,
        cpu=int(F_initial[i]), # Set initial total CPU capacity
        memory=8,
        disk=64,
    ) for i in range(NUM_NODES)
}

# --- 3. Initialize Scheduler (Central Controller) ---
scheduler = LyapunovPSOScheduler(
    edge_servers=list(edge_servers.keys()),
    num_node=NUM_NODES,
    F=F_initial, 
    R=R,
    C=C,
    L=L,
    CommCost=CommCost,
    V=V,
    E_avg=E_avg
)
scheduler._agents = edge_servers


# --- 4. Simulation Loop (Sequential Time Steps) ---
for t in tqdm(range(T_SIM), desc=f"Running Simulation (V={V}, E_avg={E_avg})"):

    # 4a. Advance time step on each edge server (internal EdgeSimPy cleanup)
    for server in edge_servers.values():
        server.step() 

    # 4b. Generate Workload (SEQUENTIAL ACCESS)
    if arrived_lists and t < len(arrived_lists):
        workload_data = arrived_lists[t]
    else:
        # Fallback for missing data
        workload_data = [random.randint(0, 3) for _ in range(NUM_NODES)]
    
    # 4c. Assign new tasks to local queues
    for i, server in edge_servers.items():
        num_tasks = int(workload_data[i])
        
        # Inject tasks as dictionaries with CPU demand into the server's task_queue
        new_tasks = []
        for j in range(num_tasks):
             new_tasks.append({'cpu_demand': C, 'id': f"Task_{t}_{i}_{j}"})
             
        server.task_queue.extend(new_tasks)

    # 4d. Update scheduler's F to reflect total capacity (F_initial)
    scheduler.F = F_initial 

    # 4e. Run scheduling step (PASS WORKLOAD_DATA DIRECTLY)
    scheduler.step(workload_data)

# --- 5. Print Final Statistics (Capture ALL new metrics) ---
T_avg, E_avg_result, Q_final, avg_throughput, offloading_ratio, avg_energy_per_task = scheduler.get_results()

print("\n=== Simulation Summary ===")
print(f"Average Task Latency (T_avg): {T_avg:.4f} (sec/task)")
print(f"Average Energy Cost (E_avg): {E_avg_result:.4f} (J/slot)")
print(f"Final Virtual Queue Level (Q_final): {Q_final:.4f}")
print(f"Average Throughput: {avg_throughput:.4f} (tasks/slot)")
print(f"Overall Offloading Ratio: {offloading_ratio:.4f}")
print(f"Avg Energy per Task: {avg_energy_per_task:.6f} (J/task)")
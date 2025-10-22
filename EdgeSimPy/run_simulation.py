import random
import numpy as np
from edge_sim_py.central_controller.controller import LyapunovPSOScheduler
from edge_sim_py.components.edge_server import EdgeServer
from utils import parse_variables

# Load simulation parameters from variables.txt
input_file = 'variables.txt'
variables = parse_variables(input_file)

NUM_NODES = variables['num_node']
TIME_SLOTS = variables['len_T']
F = np.array(variables['F'])
R = np.array(variables['R'])
C = variables['C']
L = variables['L']
T = variables['t']
CommCost = np.array(variables['CommCost'])
V = variables.get('V', 1000)
E_avg = variables.get('E_avg', 20)
arrived_lists = variables.get('arrived_lists', None)

print("Loaded parameters from variables.txt")
print(f"Nodes: {NUM_NODES}, Time Slots: {TIME_SLOTS}")
print(f"CPU Capacities (F): {F}")
print(f"Bandwidth Matrix (R): {R.shape}")
print(f"CommCost Matrix (CommCost): {CommCost.shape}")
print(f"L={L}, C={C}, V={V}, E_avg={E_avg}")

# Initialize EdgeServer agents with capacity from variables
edge_servers = {
    i: EdgeServer(
        obj_id=i,
        cpu=int(F[i]),
        memory=8,
        disk=64,
    ) for i in range(NUM_NODES)
}

# Initialize scheduler (central controller)
scheduler = LyapunovPSOScheduler(
    edge_servers=list(edge_servers.keys()),
    num_node=NUM_NODES,
    F=F,
    R=R,
    C=C,
    L=L,
    CommCost=CommCost,
    V=V,
    E_avg=E_avg
)
scheduler._agents = edge_servers

# Simulation loop with task assignment, processing, resource update, scheduling step
for t in range(T):
    print(f"\n--- Time Slot {t+1} ---")

    # Advance time step on each edge server (release resources of completed tasks)
    for server in edge_servers.values():
        server.step()  # invokes process_time_step internally

    # Generate workload for this timeslot (random or predefined)
    if arrived_lists:
        # NOTE: The original code uses a random index, which might not be T. 
        # I'll keep the original logic but note it's unusual.
        workload_data = arrived_lists[random.randint(0,TIME_SLOTS - 1)] 
    else:
        workload_data = [random.randint(0, 3) for _ in range(NUM_NODES)]

    # Assign new tasks with resource demands
    for i, server in edge_servers.items():
        server.assign_tasks(workload_data[i])  # Add new tasks
        print(f"EdgeServer {i} Active Tasks: {len(server.processing_tasks)}, CPU Demand: {server.cpu_demand}")

    # Update scheduler's view of available CPU capacity (capacity - demand)
    available_cpu = np.array([server.cpu - server.cpu_demand for server in edge_servers.values()])
    scheduler.F = available_cpu

    # Run scheduling step to update internal states and schedule tasks
    scheduler.step()

    print(f"Energy (E_t): {scheduler.E[-1]:.4f}")
    print(f"Lyapunov Queue (Q_t): {scheduler.Q[-1]:.4f}")

# After simulation, print final statistics
# UNPACKING NEW METRICS: Avg_Throughput, OOR, E_per_Task
T_avg, E_avg_result, Q_final, Avg_Throughput, OOR, E_per_Task = scheduler.get_results()

print("\n=== Simulation Summary ===")
print(f"Average Task Processing Time (T_avg): {T_avg:.4f}")
print(f"Average Energy Consumption (E_avg): {E_avg_result:.4f}")
print(f"Final Virtual Queue Level (Q_final): {Q_final:.4f}")
print(f"Average Throughput (tasks/slot): {Avg_Throughput:.4f}")
print(f"Overall Offloading Ratio: {OOR:.4f}")
print(f"Avg Energy per Task (E/task): {E_per_Task:.4f}")
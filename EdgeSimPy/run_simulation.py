"""run_simulation.py"""
import random
import numpy as np
from edge_sim_py.central_controller.controller import LyapunovPSOScheduler
from utils import parse_variables  # ✅ Import parser

# -----------------------------
# Load Parameters from variables.txt
# -----------------------------
input_file = 'variables.txt'
variables = parse_variables(input_file)

NUM_NODES = variables['num_node']
TIME_SLOTS = variables['len_T']
F = np.array(variables['F'])
R = np.array(variables['R'])
C = variables['C']
L = variables['L']
CommCost = np.array(variables['CommCost'])
V = variables.get('V', 1000)       # default if not present
E_avg = variables.get('E_avg', 20) # default if not present
arrived_lists = variables.get('arrived_lists', None)

print("✅ Loaded parameters from variables.txt")
print(f"Nodes: {NUM_NODES}, Time Slots: {TIME_SLOTS}")
print(f"CPU Capacities (F): {F}")
print(f"Bandwidth Matrix (R): {R.shape}")
print(f"CommCost Matrix (CommCost): {CommCost.shape}")
print(f"L={L}, C={C}, V={V}, E_avg={E_avg}")

# -----------------------------
# Mock mobile device agents
# -----------------------------
class MobileDevice:
    def __init__(self, id):
        self.id = id
        self.task_queue = []

    def generate_tasks(self, base_workload):
        """Simulate tasks for each node using pre-loaded or random workload."""
        if base_workload is not None:
            # use preloaded task pattern if available
            self.task_queue = [1] * int(base_workload)
        else:
            num_tasks = random.randint(0, 5)
            self.task_queue = [1] * num_tasks

# -----------------------------
# Initialize agents
# -----------------------------
agents = {i: MobileDevice(i) for i in range(NUM_NODES)}

# -----------------------------
# Initialize PSO-based Scheduler
# -----------------------------
scheduler = LyapunovPSOScheduler(
    edge_servers=list(range(NUM_NODES)),
    num_node=NUM_NODES,
    F=F,
    R=R,
    C=C,
    L=L,
    CommCost=CommCost,
    V=V,
    E_avg=E_avg
)
scheduler._agents = agents

# -----------------------------
# Run Simulation
# -----------------------------
for t in range(TIME_SLOTS):
    print(f"\n--- Time Slot {t+1} ---")

    # Step 1: Generate workload
    if arrived_lists:
        workload_data = arrived_lists[t % len(arrived_lists)]
    else:
        workload_data = [random.randint(0, 5) for _ in range(NUM_NODES)]

    for i, agent in agents.items():
        agent.generate_tasks(workload_data[i])
        print(f"Agent {i} tasks: {len(agent.task_queue)}")

    # Step 2: Run PSO + Harmony scheduling
    scheduler.step()

    # Step 3: Print metrics
    # print("Scheduling Decision:")
    # for n in range(NUM_NODES):
    #     print(f"Node {n}: {scheduler.scheduling_decision[n]}")

    print(f"Energy (E_t): {scheduler.E[-1]:.4f}")
    print(f"Lyapunov Queue (Q_t): {scheduler.Q[-1]:.4f}")

# -----------------------------
# Summary
# -----------------------------
T_avg, E_avg_result, Q_final = scheduler.get_results()
print("\n=== Simulation Summary ===")
print(f"Average Task Processing Time (T_avg): {T_avg:.4f}")
print(f"Average Energy Consumption (E_avg): {E_avg_result:.4f}")
print(f"Final Virtual Queue Level (Q_final): {Q_final:.4f}")

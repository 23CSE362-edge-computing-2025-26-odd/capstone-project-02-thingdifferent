"""run_simulation.py"""
import random
import numpy as np
from edge_sim_py.central_controller.controller import LyapunovPSOScheduler

# -----------------------------
# Simulation parameters
# -----------------------------
NUM_NODES = 4                # Number of edge servers
TIME_SLOTS = 50              # Total simulation time
TASK_ARRIVAL_RATE = 5        # Max tasks per time slot

# Lyapunov & system parameters
V = 1.0                       # Lyapunov tradeoff parameter
E_avg = 10.0                  # Average energy consumption
C = 1.0                       # Computation cycles per task
F = [10, 15, 12, 20]          # CPU capacities of edge nodes
R = np.ones((NUM_NODES, NUM_NODES)) * 2  # Bandwidth matrix (symmetric)
L = 1.0                       # Data size per task
CommCost = np.array([
    [1, 2, 1, 3],
    [2, 1, 2, 1],
    [1, 2, 1, 2],
    [3, 1, 2, 1]
])                            # Communication energy cost

# -----------------------------
# Mock mobile device agents
# -----------------------------
class MobileDevice:
    def __init__(self, id):
        self.id = id
        self.task_queue = []

    def generate_tasks(self):
        num_tasks = random.randint(0, TASK_ARRIVAL_RATE)
        self.task_queue = [1] * num_tasks  # Each task = 1 unit

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

# Register mobile devices as agents
scheduler._agents = agents

# -----------------------------
# Run Simulation
# -----------------------------
for t in range(TIME_SLOTS):
    print(f"\n--- Time Slot {t+1} ---")

    # Step 1: Each device generates tasks
    for agent in agents.values():
        agent.generate_tasks()
        print(f"Agent {agent.id} tasks: {len(agent.task_queue)}")

    # Step 2: Scheduler executes PSO + Harmony scheduling
    scheduler.step()

    # Step 3: Print decisions and metrics
    print("Scheduling Decision:")
    for n in range(NUM_NODES):
        print(f"Node {n}: {scheduler.scheduling_decision[n]}")

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

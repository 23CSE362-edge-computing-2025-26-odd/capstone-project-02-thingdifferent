"""run_simulation.py"""
import random
import numpy as np
from edge_sim_py.central_controller.controller import LyapunovScheduler

# -----------------------------
# Simulation parameters
# -----------------------------
NUM_NODES = 4                # Number of edge servers
TIME_SLOTS = 50              # Total simulation time
TASK_ARRIVAL_RATE = 5        # Max tasks per time slot
V = 1.0                       # Lyapunov tradeoff parameter
E_avg = 10.0                  # Average energy consumption
C = 1.0                       # Computation cost factor
F = [10, 15, 12, 20]          # Processing capacity of edge nodes
R = [1, 1, 1, 1]              # Some resource vector
L = [1, 1, 1, 1]              # Task length per node
CommCost = [[1,2,1,3],
            [2,1,2,1],
            [1,2,1,2],
            [3,1,2,1]]        # Communication cost between nodes

# -----------------------------
# Mock mobile device agents
# -----------------------------
class MobileDevice:
    def __init__(self, id):
        self.id = id
        self.task_queue = []

    def generate_tasks(self):
        num_tasks = random.randint(0, TASK_ARRIVAL_RATE)
        self.task_queue = [1]*num_tasks  # Each task = 1 unit

# -----------------------------
# Initialize agents
# -----------------------------
agents = {i: MobileDevice(i) for i in range(NUM_NODES)}

# -----------------------------
# Initialize Lyapunov Scheduler
# -----------------------------
scheduler = LyapunovScheduler(
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

# Register agents
scheduler._agents = agents

# -----------------------------
# Run simulation
# -----------------------------
for t in range(TIME_SLOTS):
    print(f"\n--- Time Slot {t+1} ---")

    # Step 1: Mobile devices generate tasks
    for agent in agents.values():
        agent.generate_tasks()
        print(f"Agent {agent.id} tasks: {len(agent.task_queue)}")

    # Step 2: Scheduler decides task allocation
    scheduler.step()

    # Step 3: Print scheduling decision and energy
    print("Scheduling Decision:")
    for n in range(NUM_NODES):
        print(f"Node {n}: {scheduler.scheduling_decision[n]}")

    print(f"Energy this slot: {scheduler.E[-1]:.2f}")
    print(f"Lyapunov Queue: {scheduler.Q[-1]:.2f}")

# -----------------------------
# Summary
# -----------------------------
total_energy = sum(scheduler.E)
print(f"\nTotal energy over {TIME_SLOTS} slots: {total_energy:.2f}")

import numpy as np
import random
from tqdm import tqdm
from .utils import distribute_evenly


class LyapunovPSOScheduler:
    def __init__(self, edge_servers, num_node, F, R, C, L, CommCost, V, E_avg):
        self.edge_servers = edge_servers
        self.num_node = num_node
        self.F = F                  # CPU capacity
        self.R = R                  # Bandwidth matrix
        self.C = C                  # Computation cost per task
        self.L = L                  # Data size per task
        self.CommCost = CommCost    # Communication energy cost matrix
        self.V = V                  # Lyapunov tradeoff parameter
        self.E_avg = E_avg          # Average energy threshold

        # Lyapunov states
        self.Q = [0]                # Virtual queue for energy constraint
        self.E = [0]                # Energy per timeslot
        self.T = []                 # Total time per slot (comp + comm)
        self.t = -1
        self.steps = 0
        self._agents = {}
        self.scheduling_decision = [[0]*num_node for _ in range(num_node)]

    # ---------------------------------------------------------------------
    # Computation and Energy Calculation
    # ---------------------------------------------------------------------
    def calculate_T_E(self, assignment, workload_data):
        CompTimeTotal = 0
        CommTimeTotal = 0
        E_cost = 0
        for i in range(self.num_node):
            if assignment[i] == 0:  # Source
                for j in range(self.num_node):
                    if assignment[j] == 1:  # Sink
                        comp_time = self.C * workload_data[i] / (self.F[j] + 1e-10)
                        comm_time = self.L * workload_data[i] / (self.R[i][j] + 1e-10)
                        CompTimeTotal += comp_time
                        CommTimeTotal += comm_time
                        E_cost += self.CommCost[i][j] * workload_data[i]
        return CompTimeTotal, CommTimeTotal, E_cost

    # ---------------------------------------------------------------------
    # Lyapunov Drift + Penalty Objective
    # ---------------------------------------------------------------------
    def simplified_drift_plus_penalty(self, assignment, workload_data):
        num_sources = np.sum(assignment == 0)
        num_sinks = np.sum(assignment == 1)
        if num_sources == 0 or num_sinks == 0:
            return 1e9  # Invalid configuration (no source/sink)
        comp_time, comm_time, e_cost = self.calculate_T_E(assignment, workload_data)
        return self.V * (comp_time + comm_time) + self.Q[-1] * (e_cost - self.E_avg)

    # ---------------------------------------------------------------------
    # PSO for Node Role Assignment
    # ---------------------------------------------------------------------
    def PSO_Node_Assignment(self, workload_data):
        popsize, c1, c2 = 20, 2.0, 2.0
        w_initial, w_final = 0.9, 0.4
        G = 50  # Number of PSO generations

        # Initialize population with random roles (0=source, 1=sink, 2=isolated)
        pop = np.random.randint(0, 3, size=(popsize, self.num_node))
        pbest = np.copy(pop)
        pbest_fx = np.array([self.simplified_drift_plus_penalty(p, workload_data) for p in pop])

        # Global best initialization
        gbest_idx = np.argmin(pbest_fx)
        gbest = np.copy(pop[gbest_idx])
        gbest_fx = pbest_fx[gbest_idx]

        # Main PSO loop
        for i in range(1, G + 1):
            w = w_initial - (w_initial - w_final) * (i / G)
            for j in range(popsize):
                r1, r2 = np.random.rand(self.num_node), np.random.rand(self.num_node)
                update_prob = w + c1 * r1 + c2 * r2
                new_pop_j = np.copy(pop[j])

                for k in range(self.num_node):
                    if update_prob[k] > 2.0:
                        new_pop_j[k] = gbest[k]
                    elif update_prob[k] > 1.0:
                        new_pop_j[k] = pbest[j, k]
                    else:
                        new_pop_j[k] = np.random.randint(0, 3)

                fx_j = self.simplified_drift_plus_penalty(new_pop_j, workload_data)
                if fx_j < pbest_fx[j]:
                    pbest[j] = new_pop_j
                    pbest_fx[j] = fx_j

            # Update global best
            min_fx = np.min(pbest_fx)
            if min_fx < gbest_fx:
                gbest_fx = min_fx
                gbest = np.copy(pbest[np.argmin(pbest_fx)])

        return gbest, gbest_fx

    # ---------------------------------------------------------------------
    # Harmony Search-Based Task Scheduler
    # ---------------------------------------------------------------------
    def Harmony_Search_Task_Scheduler(self, best_assignment, workload_data, t):
        print(f"\n--- Time Slot {t+1} ---")
        print(f"Node Workloads: {[f'{w:.4f}' for w in workload_data]}")
        print(f"PSO Optimal Assignment: {best_assignment}")

        # Identify node roles
        sources = [i for i, role in enumerate(best_assignment) if role == 0]
        sinks = [i for i, role in enumerate(best_assignment) if role == 1]
        isolated = [i for i, role in enumerate(best_assignment) if role == 2]

        print(f"Identified Sources: {sources}")
        print(f"Identified Sinks: {sinks}")
        print(f"Identified Isolated Nodes: {isolated}")

        # Compute capacities and total workload
        total_offload = sum(workload_data[s] for s in sources)
        sink_cap = {j: self.F[j] / self.C for j in sinks}
        total_cap = sum(sink_cap.values())

        print(f"Total Offload Workload: {total_offload:.4f}")
        print(f"Total Sink Capacity: {total_cap:.4f}")

        # Task distribution
        if total_offload > 0 and total_cap > 0:
            for s in sources:
                src_load = workload_data[s]
                print(f"  Source Node {s+1} (Workload {src_load:.4f}) offloads:")
                for j in sinks:
                    frac = sink_cap[j] / total_cap
                    tasks = src_load * frac
                    print(f"    → {tasks:.4f} units to Sink Node {j+1}")
        else:
            print("No feasible offloading due to lack of sink capacity or workload.")

        # Isolated nodes process their own tasks
        if isolated:
            for i in isolated:
                print(f"  Isolated Node {i+1} processes its own workload: {workload_data[i]:.4f}")

        # Compute metrics
        comp_time, comm_time, e_cost = self.calculate_T_E(best_assignment, workload_data)
        print(f"Computed Metrics — Computation Time: {comp_time:.4f}, "
              f"Communication Time: {comm_time:.4f}, Energy Cost: {e_cost:.4f}")
        print(f"Updated Virtual Queue Q[{t}] = {self.Q[-1]:.4f}")

        return comp_time, comm_time, e_cost

    # ---------------------------------------------------------------------
    # One Simulation Step
    # ---------------------------------------------------------------------
    def step(self):
        self.steps += 1
        self.t += 1

        # Gather current workload
        workload_data = [len(agent.task_queue) for agent in self._agents.values()]

        # Run PSO + Harmony scheduling
        best_assignment, _ = self.PSO_Node_Assignment(workload_data)
        comp_t, comm_t, e_t = self.Harmony_Search_Task_Scheduler(best_assignment, workload_data, self.t)

        # Update Lyapunov metrics
        self.T.append(comp_t + comm_t)
        self.E.append(e_t)
        Q_next = max(self.Q[-1] + e_t - self.E_avg, 0)
        self.Q.append(Q_next)

        # Clear processed tasks
        for agent in self._agents.values():
            agent.task_queue = []

    # ---------------------------------------------------------------------
    # Results Summary
    # ---------------------------------------------------------------------
    def get_results(self):
        T_avg = np.mean(self.T) if self.T else 0
        E_avg_ = np.mean(self.E) if self.E else 0
        return T_avg, E_avg_, self.Q[-1]

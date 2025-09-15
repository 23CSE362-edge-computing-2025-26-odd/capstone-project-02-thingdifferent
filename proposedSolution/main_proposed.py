import random
import numpy as np
from tqdm import tqdm
from utils import parse_variables

class LyapunovHeuristicScheduler:
    def __init__(self, num_node, arrived_lists, len_T, E_avg, F, R, C, L, CommCost, V):
        self.num_nodes = num_node
        self.len_T = len_T
        self.arrived_lists = arrived_lists
        self.E_avg = E_avg
        self.F = F
        self.R = R
        self.C = C
        self.L = L
        self.CommCost = CommCost
        self.V = V

        self.Q = [0] * len_T
        self.T = [0] * len_T
        self.E = [0] * len_T

    
    def calculate_T_E(self, assignment, workload_data):
        CompTimeTotal = 0
        CommTimeTotal = 0
        E_cost = 0

        for i in range(self.num_nodes):
            if assignment[i] == 0:  # Source node
                for j in range(self.num_nodes):
                    if assignment[j] == 1:  # Sink node
                        comp_time = self.C * workload_data[i] / self.F[j]
                        comm_time = self.L * workload_data[i] / self.R[i][j]
                        CompTimeTotal += comp_time
                        CommTimeTotal += comm_time
                        E_cost += self.CommCost[i][j] * workload_data[i]

        return CompTimeTotal, CommTimeTotal, E_cost


    def simplified_drift_plus_penalty(self, assignment, workload_data, t):
        num_sources = np.sum(assignment == 0)
        num_sinks = np.sum(assignment == 1)

        if num_sources == 0 or num_sinks == 0:
            return 1e9

        comp_time, comm_time, e_cost = self.calculate_T_E(assignment, workload_data)
        objective = self.V * (comp_time + comm_time) + self.Q[t] * (e_cost - self.E_avg)
        return objective
    
        
    def PSO_Node_Assignment(self, workload_data, t):
        popsize = 20
        c1 = 2.0
        c2 = 2.0
        w_initial = 0.9
        w_final = 0.4
        G = 50

        pop = np.random.randint(0, 3, size=(popsize, self.num_nodes))
        pbest = np.copy(pop)
        pbest_fx = np.array([self.simplified_drift_plus_penalty(p, workload_data, t) for p in pop])

        gbest_index = np.argmin(pbest_fx)
        gbest = np.copy(pop[gbest_index])
        gbest_fx = pbest_fx[gbest_index]

        for i in range(1, G + 1):
            w = w_initial - (w_initial - w_final) * (i / G)

            for j in range(popsize):
                r1 = np.random.rand(self.num_nodes)
                r2 = np.random.rand(self.num_nodes)

                update_prob = w + c1 * r1 + c2 * r2

                new_pop_j = np.copy(pop[j])
                for k in range(self.num_nodes):
                    if update_prob[k] > 2.0:
                        new_pop_j[k] = gbest[k]
                    elif update_prob[k] > 1.0:
                        new_pop_j[k] = pbest[j, k]
                    else:
                        new_pop_j[k] = np.random.randint(0, 3)

                current_fx_j = self.simplified_drift_plus_penalty(new_pop_j, workload_data, t)

                if current_fx_j < pbest_fx[j]:
                    pbest[j] = new_pop_j
                    pbest_fx[j] = current_fx_j

            min_pbest_fx = np.min(pbest_fx)
            if min_pbest_fx < gbest_fx:
                gbest_fx = min_pbest_fx
                gbest = pbest[np.argmin(pbest_fx)]

        return gbest, gbest_fx
    
    def Harmony_Search_Task_Scheduler(self, best_assignment, workload_data, t):
        print(f"\n--- Time Slot {t+1} ---")
        print(f"Node Workloads: {[f'{w:.4f}' for w in workload_data]}")
        print(f"PSO Optimal Assignment: {best_assignment}")

        sources = [i for i, role in enumerate(best_assignment) if role == 0]
        sinks = [i for i, role in enumerate(best_assignment) if role == 1]
        isolated = [i for i, role in enumerate(best_assignment) if role == 2]

        print(f"Identified Sources: {sources}")
        print(f"Identified Sinks: {sinks}")
        print(f"Identified Isolated Nodes: {isolated}")

        total_offload_workload = sum(workload_data[s] for s in sources)
        total_sink_capacity = sum(1 - workload_data[s] for s in sinks)

        print(f"Total Offload Workload: {total_offload_workload:.4f}")
        print(f"Total Sink Capacity: {total_sink_capacity:.4f}")

        if total_offload_workload > 0 and total_sink_capacity > 0:
            for source_node in sources:
                source_offload = workload_data[source_node]
                print(f"  Source Node {source_node+1} (Workload {source_offload:.4f}) offloads:")

                for sink_node in sinks:
                    sink_fraction = (1 - workload_data[sink_node]) / total_sink_capacity
                    tasks_to_send = source_offload * sink_fraction
                    print(f"    → {tasks_to_send:.4f} units to Sink Node {sink_node+1}")

        else:
            print("No offloading or sink capacity available.")

        if isolated:
            for isolated_node in isolated:
                print(f"  Isolated Node {isolated_node+1} processes its own workload: {workload_data[isolated_node]:.4f}")

        comp_time, comm_time, e_cost = self.calculate_T_E(best_assignment, workload_data)
        self.T[t] = comp_time + comm_time
        self.E[t] = e_cost

        print(f"Computed Metrics — Computation Time: {comp_time:.4f}, Communication Time: {comm_time:.4f}, Energy Cost: {e_cost:.4f}")
        print(f"Updated Virtual Queue Q[{t}] = {self.Q[t]:.4f}")

    def step(self, workload_data, t):
        best_assignment, _ = self.PSO_Node_Assignment(workload_data, t)
        self.Harmony_Search_Task_Scheduler(best_assignment, workload_data, t)

        if t < self.len_T - 1:
            self.Q[t + 1] = max(self.Q[t] + self.E[t] - self.E_avg, 0)

    def run(self):
        for t in tqdm(range(self.len_T), desc="Running"):
            workload_data = self.arrived_lists[t]
            self.step(workload_data, t)
        T_avg = sum(self.T) / self.len_T
        E_avg_ = sum(self.E) / self.len_T
        print("\nT_avg:", T_avg, "E_avg:", E_avg_)


if __name__ == '__main__':
    input_file = 'variables.txt'
    variables = parse_variables(input_file)

    num_node = variables['num_node']
    arrived_lists = variables['arrived_lists']
    F = variables['F'] #CPU Capacity
    R = variables['R'] #bandwidth
    C = variables['C'] #energy efficiency
    L = variables['L'] #data size per task
    CommCost = variables['CommCost']
    V = 100000
    E_avg = 20
    len_T = variables['len_T']

    print("Loaded variables from variables.txt")

    scheduler = LyapunovHeuristicScheduler(num_node, arrived_lists, len_T, E_avg, F, R, C, L, CommCost, V)
    scheduler.run()

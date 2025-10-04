import numpy as np
import random
from tqdm import tqdm
from .utils import distribute_evenly

class LyapunovPSOScheduler:
    def __init__(self, edge_servers, num_node, F, R, C, L, CommCost, V, E_avg):
        self.edge_servers = edge_servers
        self.num_node = num_node
        self.F = F
        self.R = R
        self.C = C
        self.L = L
        self.CommCost = CommCost
        self.V = V
        self.E_avg = E_avg

        # Lyapunov states
        self.Q = [0]          # Virtual queue
        self.E = [0]          # Energy per slot
        self.T = []           # Time metrics
        self.t = -1
        self.steps = 0
        self._agents = {}     # Mobile devices
        self.scheduling_decision = [[0]*num_node for _ in range(num_node)]

    # ---- Computation and Energy Calculation ----
    def calculate_T_E(self, assignment, workload_data):
        CompTimeTotal = 0
        CommTimeTotal = 0
        E_cost = 0
        for i in range(self.num_node):
            if assignment[i] == 0:  # Source node
                for j in range(self.num_node):
                    if assignment[j] == 1:  # Sink node
                        comp_time = self.C * workload_data[i] / (self.F[j] + 1e-10)
                        comm_time = self.L * workload_data[i] / (self.R[i][j] + 1e-10)
                        CompTimeTotal += comp_time
                        CommTimeTotal += comm_time
                        E_cost += self.CommCost[i][j] * workload_data[i]
        return CompTimeTotal, CommTimeTotal, E_cost

    # ---- Lyapunov Drift-Penalty Objective ----
    def simplified_drift_plus_penalty(self, assignment, workload_data):
        num_sources = np.sum(assignment == 0)
        num_sinks = np.sum(assignment == 1)
        if num_sources == 0 or num_sinks == 0:
            return 1e9

        comp_time, comm_time, e_cost = self.calculate_T_E(assignment, workload_data)
        objective = self.V * (comp_time + comm_time) + self.Q[-1] * (e_cost - self.E_avg)
        return objective

    # ---- PSO for Node Role Assignment ----
    def PSO_Node_Assignment(self, workload_data):
        popsize = 20
        c1, c2 = 2.0, 2.0
        w_initial, w_final = 0.9, 0.4
        G = 50

        pop = np.random.randint(0, 3, size=(popsize, self.num_node))
        pbest = np.copy(pop)
        pbest_fx = np.array([self.simplified_drift_plus_penalty(p, workload_data) for p in pop])

        gbest_idx = np.argmin(pbest_fx)
        gbest = np.copy(pop[gbest_idx])
        gbest_fx = pbest_fx[gbest_idx]

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

            min_fx = np.min(pbest_fx)
            if min_fx < gbest_fx:
                gbest_fx = min_fx
                gbest = np.copy(pbest[np.argmin(pbest_fx)])

        return gbest, gbest_fx

    # ---- Harmony-based Task Distribution ----
    def Harmony_Search_Task_Scheduler(self, best_assignment, workload_data):
        sources = [i for i, r in enumerate(best_assignment) if r == 0]
        sinks = [i for i, r in enumerate(best_assignment) if r == 1]
        isolated = [i for i, r in enumerate(best_assignment) if r == 2]

        total_offload = sum(workload_data[s] for s in sources)
        sink_cap = {j: self.F[j]/self.C for j in sinks}
        total_cap = sum(sink_cap.values())

        scheduling_decision = [[0]*self.num_node for _ in range(self.num_node)]

        if total_offload > 0 and total_cap > 0:
            for s in sources:
                src_load = workload_data[s]
                for j in sinks:
                    frac = sink_cap[j] / total_cap
                    tasks = src_load * frac
                    scheduling_decision[s][j] = tasks
        else:
            for i in isolated:
                scheduling_decision[i][i] = workload_data[i]

        comp_time, comm_time, e_cost = self.calculate_T_E(best_assignment, workload_data)
        return scheduling_decision, comp_time, comm_time, e_cost

    # ---- One Simulation Step ----
    def step(self):
        self.steps += 1
        self.t += 1

        workload_data = [len(agent.task_queue) for agent in self._agents.values()]
        best_assignment, _ = self.PSO_Node_Assignment(workload_data)
        scheduling_decision, comp_t, comm_t, e_t = self.Harmony_Search_Task_Scheduler(best_assignment, workload_data)

        self.T.append(comp_t + comm_t)
        self.E.append(e_t)
        self.scheduling_decision = scheduling_decision

        # Update Lyapunov queue
        Q_next = max(self.Q[-1] + e_t - self.E_avg, 0)
        self.Q.append(Q_next)

        # Clear processed tasks
        for agent in self._agents.values():
            agent.task_queue = []

    # ---- Summary ----
    def get_results(self):
        T_avg = np.mean(self.T)
        E_avg_ = np.mean(self.E)
        return T_avg, E_avg_, self.Q[-1]

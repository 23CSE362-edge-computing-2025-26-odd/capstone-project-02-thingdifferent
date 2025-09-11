import random
import numpy as np
from tqdm import tqdm

class LyapunovHeuristicScheduler:
    def _init_(self, num_node, arrived_lists, len_T, E_avg, F, R, C, L, CommCost, V):
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


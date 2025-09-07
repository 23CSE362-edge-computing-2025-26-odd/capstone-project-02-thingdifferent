"""controller.py"""
import numpy as np
import random
import math
from .utils import generate_random_list, boundary_check, copyx, crossover, mutation, distribute_evenly


class LyapunovScheduler:
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

        self.Q = [0]          # Virtual queue
        self.E = [0]          # Energy per time slot
        self.T = []           # Optional: computation + communication time
        self.t = -1           # Current time slot
        self.steps = 0
        self._agents = {}     # MobileDevices registered as agents
        self.scheduling_decision = [[0]*num_node for _ in range(num_node)]

    def step(self):
        self.steps += 1
        self.t += 1

        # Gather tasks per agent at this time slot
        arrived_lists = [len(agent.task_queue) for agent in self._agents.values()]

        # Select nodes with non-zero F
        non_zero_indices = [i for i, f in enumerate(self.F) if f != 0]

        # GA Parameters
        xlim = [0, 1, 2]
        popsize = len(non_zero_indices)
        chromlength = self.num_node
        pc, pm, G = 0.6, 0.1, 50

        # Generate initial population using paper's logic
        pop = np.zeros((popsize, chromlength), dtype=int)
        for idx in range(popsize):
            pop[idx] = generate_random_list(len(non_zero_indices), xlim)
        
        # Evaluate initial population
        scheduling_decisions, obj = self.calobjvalue_GA(pop, arrived_lists)
        best_obj_idx = np.argmin(obj)
        best_decision = scheduling_decisions[best_obj_idx]
        obj_best = [obj[best_obj_idx]]
        scheduling_decision_best = [best_decision]

        # GA iterations
        for g in range(1, G):
            newpop = copyx(pop, obj, popsize)
            newpop = crossover(non_zero_indices, newpop, pc, popsize)
            newpop = mutation(non_zero_indices, newpop, pm, popsize)

            _, new_obj = self.calobjvalue_GA(newpop, arrived_lists)
            index = np.where(new_obj < obj)[0]
            pop[index] = newpop[index]

            scheduling_decisions, obj = self.calobjvalue_GA(pop, arrived_lists)
            best_val = min(obj)
            best_scheduling_decision = scheduling_decisions[np.argmin(obj)]
            obj_best.append(best_val)
            scheduling_decision_best.append(best_scheduling_decision)

        # Select the best decision
        best_obj = min(obj_best)
        best_decision = scheduling_decision_best[np.argmin(obj_best)]
        self.scheduling_decision = best_decision

        # Compute energy and update virtual queue
        E_t = self.get_E(best_decision)
        self.E.append(E_t)
        Q_next = max(self.Q[-1] + E_t - self.E_avg, 0)
        self.Q.append(Q_next)

        # Clear tasks
        for agent in self._agents.values():
            agent.task_queue = []

    def calobjvalue_GA(self, pop, arrived_lists):
        obj = np.zeros(pop.shape[0])
        scheduling_decisions = []

        for idx in range(pop.shape[0]):
            source_nodes = [i for i in range(self.num_node) if pop[idx][i] == 0]
            sink_nodes = [i for i in range(self.num_node) if pop[idx][i] == 1]

            non_zero_indices = [[] for _ in range(self.num_node)]
            for i in range(self.num_node):
                if i in source_nodes:
                    if self.F[i] == 0:
                        non_zero_indices[i].extend(sink_nodes)
                    else:
                        non_zero_indices[i].append(i)
                        non_zero_indices[i].extend(sink_nodes)
                elif i in sink_nodes:
                    non_zero_indices[i].append(i)
                else:
                    non_zero_indices[i].append(i)

            dim_opt = [len(non_zero_indices[i]) - 1 for i in range(self.num_node)]

            if any(x >= 1 for x in dim_opt):
                # Harmony Search bounds
                lb = [0]*sum(dim_opt)
                ub = []
                for i in range(self.num_node):
                    for _ in range(dim_opt[i]):
                        ub.append(arrived_lists[i])

                hms, iter_, hmcr, par, bw = 30, max(ub), 0.8, 0.1, 0.1
                nnew = len(lb)
                scheduling_decision, obj[idx] = self.Harmony_Search(non_zero_indices, dim_opt, hms, iter_,
                                                                    hmcr, par, bw, nnew, lb, ub, arrived_lists)
                scheduling_decisions.append(scheduling_decision)
            else:
                # Fallback distribution
                scheduling_decision = [[0]*self.num_node for _ in range(self.num_node)]
                non_zero_idx = [i for i, x in enumerate(self.F) if x != 0]
                dist_count = len(non_zero_idx)
                for n in range(self.num_node):
                    dist = distribute_evenly(arrived_lists[n], dist_count)
                    for k, val in enumerate(dist):
                        scheduling_decision[n][non_zero_idx[k]] = val
                obj[idx] = 1e10
                scheduling_decisions.append(scheduling_decision)

        return scheduling_decisions, obj

    def Harmony_Search(self, non_zero_indices, dim_opt, hms, iter_, hmcr, par, bw, nnew,
                        lb, ub, arrived_lists):
        pos = [[random.randint(lb[j], ub[j]) for j in range(len(lb))] for _ in range(hms)]
        score = [self.calobjvalue_HS(p, non_zero_indices, dim_opt, arrived_lists) for p in pos]

        gbest = min(score)
        gbest_pos = pos[score.index(gbest)].copy()

        for _ in range(iter_):
            new_pos, new_score = [], []
            for _ in range(nnew):
                temp_pos = []
                for j in range(len(lb)):
                    if random.random() < hmcr:
                        ind = random.randint(0, hms-1)
                        val = pos[ind][j]
                        if random.random() < par:
                            val += math.floor(random.normalvariate(0,1)*bw*(ub[j]-lb[j]))
                        temp_pos.append(val)
                    else:
                        temp_pos.append(random.randint(lb[j], ub[j]))
                temp_pos = boundary_check(temp_pos, lb, ub)
                new_pos.append(temp_pos)
                new_score.append(self.calobjvalue_HS(temp_pos, non_zero_indices, dim_opt, arrived_lists))

            pos.extend(new_pos)
            score.extend(new_score)
            sorted_idx = np.argsort(score)[:hms]
            pos = [pos[i] for i in sorted_idx]
            score = [score[i] for i in sorted_idx]

            if score[0] < gbest:
                gbest = score[0]
                gbest_pos = pos[0].copy()

        # Build final scheduling decision
        g = [0]*self.num_node
        for i in range(self.num_node):
            if dim_opt[i]==0:
                g[i] = arrived_lists[i]
            else:
                g[i] = arrived_lists[i] - sum(gbest_pos[sum(dim_opt[:i]):sum(dim_opt[:i+1])])

        if all(x>=0 for x in g):
            scheduling_decision = [[0]*self.num_node for _ in range(self.num_node)]
            for j in range(self.num_node):
                if dim_opt[j]==0:
                    scheduling_decision[j][non_zero_indices[j][0]] = arrived_lists[j]
                else:
                    for idx_ in range(dim_opt[j]):
                        scheduling_decision[j][non_zero_indices[j][idx_]] = gbest_pos[sum(dim_opt[:j])+idx_]
                    scheduling_decision[j][non_zero_indices[j][-1]] = g[j]
            return scheduling_decision, gbest
        else:
            # fallback
            scheduling_decision = [[0]*self.num_node for _ in range(self.num_node)]
            non_zero_idx = [i for i, x in enumerate(self.F) if x != 0]
            dist_count = len(non_zero_idx)
            for n in range(self.num_node):
                dist = distribute_evenly(arrived_lists[n], dist_count)
                for k, val in enumerate(dist):
                    scheduling_decision[n][non_zero_idx[k]] = val
            return scheduling_decision, 1e10

    def calobjvalue_HS(self, temp_pos, non_zero_indices, dim_opt, arrived_lists):
        g = [0]*self.num_node
        for i in range(self.num_node):
            if dim_opt[i]==0:
                g[i] = arrived_lists[i]
            else:
                g[i] = arrived_lists[i] - sum(temp_pos[sum(dim_opt[:i]):sum(dim_opt[:i+1])])

        if not all(x>=0 for x in g):
            return 1e10

        scheduling_decision = [[0]*self.num_node for _ in range(self.num_node)]
        for j in range(self.num_node):
            if dim_opt[j]==0:
                scheduling_decision[j][non_zero_indices[j][0]] = arrived_lists[j]
            else:
                for idx_ in range(dim_opt[j]):
                    scheduling_decision[j][non_zero_indices[j][idx_]] = temp_pos[sum(dim_opt[:j])+idx_]
                scheduling_decision[j][non_zero_indices[j][-1]] = g[j]

        T = sum([self.C*sum(scheduling_decision[n])/(self.F[n]+1e-10) for n in range(self.num_node)])
        E = sum([self.CommCost[n][m]*scheduling_decision[n][m] for n in range(self.num_node) for m in range(self.num_node)])
        return self.V*T + self.Q[self.t]*(E - self.E_avg)

    def get_E(self, scheduling_decision):
        return sum([self.CommCost[n][m]*scheduling_decision[n][m] for n in range(self.num_node) for m in range(self.num_node)])

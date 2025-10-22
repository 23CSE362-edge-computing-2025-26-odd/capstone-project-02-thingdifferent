import numpy as np
import random
import math
from .utils import distribute_evenly # Assuming boundary_check, etc., are here

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
        self.Q = [0]
        self.E = [0]
        self.T = []
        
        # --- NEW METRICS INITIALIZATION ---
        self.tasks_arrived = []
        self.tasks_offloaded = []
        
        self.t = -1
        self.steps = 0
        self._agents = {}
        self.scheduling_decision = [[0]*num_node for _ in range(num_node)]


    # ---------------------------------------------------------------------
    # Computation and Energy Calculation (T and E)
    # ---------------------------------------------------------------------
    def calculate_T_E(self, assignment, workload_data):
        CompTimeTotal = 0
        CommTimeTotal = 0
        E_cost = 0
        
        for i in range(self.num_node):
            if assignment[i] == 0:  # Source
                for j in range(self.num_node):
                    if assignment[j] == 1:  # Sink
                        tasks_to_schedule = workload_data[i] 

                        comp_time = self.C * tasks_to_schedule / (self.F[j] + 1e-10)
                        comm_time = self.L * tasks_to_schedule / (self.R[i][j] + 1e-10)
                        
                        CompTimeTotal += comp_time
                        CommTimeTotal += comm_time
                        E_cost += self.CommCost[i][j] * tasks_to_schedule
        return CompTimeTotal, CommTimeTotal, E_cost

    # ---------------------------------------------------------------------
    # Lyapunov Drift + Penalty Objective
    # ---------------------------------------------------------------------
    def simplified_drift_plus_penalty(self, assignment, workload_data):
        num_sources = np.sum(assignment == 0)
        num_sinks = np.sum(assignment == 1)
        if num_sources == 0 or num_sinks == 0:
            return 1e10
            
        comp_time, comm_time, e_cost = self.calculate_T_E(assignment, workload_data)
        return self.V * (comp_time + comm_time) + self.Q[-1] * (e_cost - self.E_avg)

    # ---------------------------------------------------------------------
    # PSO for Node Role Assignment
    # ---------------------------------------------------------------------
    def PSO_Node_Assignment(self, workload_data):
        # NOTE: This is a placeholder for the full PSO logic for node assignment.
        # It uses simplified discrete PSO logic (as implemented in previous steps)
        
        popsize, c1, c2 = 20, 2.0, 2.0
        w_initial, w_final = 0.9, 0.4
        G = 50 

        pop = np.random.randint(0, 3, size=(popsize, self.num_node))
        pbest_fx = np.array([self.simplified_drift_plus_penalty(p, workload_data) for p in pop])

        gbest_idx = np.argmin(pbest_fx)
        gbest = np.copy(pop[gbest_idx])
        gbest_fx = pbest_fx[gbest_idx]

        # Simplified PSO loop logic
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
                        new_pop_j[k] = pop[j][k] # Simplified pbest logic
                    else:
                        new_pop_j[k] = np.random.randint(0, 3)

                fx_j = self.simplified_drift_plus_penalty(new_pop_j, workload_data)
                if fx_j < pbest_fx[j]:
                    pop[j] = new_pop_j
                    pbest_fx[j] = fx_j

            min_fx = np.min(pbest_fx)
            if min_fx < gbest_fx:
                gbest_fx = min_fx
                gbest = np.copy(pop[np.argmin(pbest_fx)])

        return gbest, gbest_fx


    # ---------------------------------------------------------------------
    # Harmony Search-Based Task Scheduler (Metric Collector)
    # ---------------------------------------------------------------------
    def Harmony_Search_Task_Scheduler(self, best_assignment, workload_data, t):
        # Calculate Offloaded Volume (Approximation based on roles)
        sources = [i for i, role in enumerate(best_assignment) if role == 0]
        total_offload_workload = sum(workload_data[s] for s in sources)
        offloaded_count = total_offload_workload # Assuming all source load is offloaded
        
        # Compute final T and E based on the assignment (for Lyapunov update)
        comp_time, comm_time, e_cost = self.calculate_T_E(best_assignment, workload_data)

        return comp_time, comm_time, e_cost, offloaded_count

    # ---------------------------------------------------------------------
    # One Simulation Step (Main Scheduler Loop)
    # ---------------------------------------------------------------------
    def step(self, workload_data): # ACCEPT WORKLOAD_DATA DIRECTLY
        self.steps += 1
        self.t += 1

        total_arrived_in_slot = sum(workload_data)
        
        # Run PSO (Node Assignment)
        best_assignment, _ = self.PSO_Node_Assignment(workload_data)
        
        # Run Harmony Scheduling (Task Distribution & Metric Calculation)
        Comp_t, Comm_t, E_t, offloaded_count = self.Harmony_Search_Task_Scheduler(
            best_assignment, workload_data, self.t
        )

        # --- RECORD ALL METRICS ---
        self.tasks_arrived.append(total_arrived_in_slot)
        self.tasks_offloaded.append(offloaded_count)
        self.T.append(Comp_t + Comm_t)
        self.E.append(E_t)

        # Update Lyapunov virtual queue
        Q_next = max(self.Q[-1] + E_t - self.E_avg, 0)
        self.Q.append(Q_next)

        # Clear tasks (Simulating completion/processing of all tasks in the slot)
        for agent in self._agents.values():
            agent.task_queue.clear()
            
    # ---------------------------------------------------------------------
    # Results Summary (Full Edge-Centric Metrics)
    # ---------------------------------------------------------------------
    def get_results(self):
        T_total_avg = np.mean(self.T) if self.T else 0
        E_avg_result = np.mean(self.E) if self.E else 0
        Q_final = self.Q[-1]

        total_arrived = sum(self.tasks_arrived)
        steps = self.steps
        
        avg_throughput = total_arrived / steps if steps else 0
        
        total_offloaded = sum(self.tasks_offloaded)
        offloading_ratio = total_offloaded / total_arrived if total_arrived else 0
        
        avg_energy_per_task = E_avg_result / avg_throughput if avg_throughput else 0
        
        return T_total_avg, E_avg_result, Q_final, avg_throughput, offloading_ratio, avg_energy_per_task
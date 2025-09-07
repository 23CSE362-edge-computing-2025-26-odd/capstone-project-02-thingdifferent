# edge_sim_py/components/custom_task.py

class Task:
    def __init__(self, simulator, origin, cpu_cycles, data_size, deadline, creation_time):
        self.simulator = simulator
        self.origin = origin
        self.cpu_cycles = cpu_cycles
        self.data_size = data_size
        self.deadline = deadline
        self.creation_time = creation_time

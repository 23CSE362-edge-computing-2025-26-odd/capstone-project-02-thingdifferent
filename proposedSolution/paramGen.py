# Parameter Setup file
import random

# Parameter settings
file_name = 'numbers.txt'  


arrived_lists = []

with open(file_name, 'r') as file:
    for line in file:
        row = list(map(int, line.strip().split()))
        arrived_lists.append(row)

len_T = len(arrived_lists)

num_node = len(arrived_lists[0])

E_avg = 20 

F = [random.uniform(50e10, 60e10), random.uniform(40e10, 50e10), random.uniform(30e10, 40e10),
     random.uniform(20e10, 30e10), random.uniform(10e10, 20e10)]
R = []
for i in range(num_node):
    R.append([random.uniform(1 * 10 ** 10, 1 * 10 ** 11) for _ in range(num_node)])
    R[i][i] = 0

C = random.uniform(0.5 * 10 ** 8, 0.5 * 10 ** 9)


L = random.uniform(2 * 10 ** 5, 5 * 10 ** 5)

CommCost = []
for i in range(num_node):
    CommCost.append([random.uniform(0.4, 2.4) for _ in range(num_node)])
    CommCost[i][i] = 0

V = 100

# Write variables to a file
output_file = 'variables.txt'
with open(output_file, 'w') as file:
    file.write(f"num_node={num_node}\n")
    file.write(f"arrived_lists={arrived_lists}\n")
    file.write(f"len_T={len_T}\n")
    file.write(f"F={F}\n")
    file.write(f"R={R}\n")
    file.write(f"C={C}\n")
    file.write(f"L={L}\n")
    file.write(f"CommCost={CommCost}\n")
    file.write(f"V={V}\n")
    file.write(f"E_avg={E_avg}\n")

print(f"Variables written to {output_file}")

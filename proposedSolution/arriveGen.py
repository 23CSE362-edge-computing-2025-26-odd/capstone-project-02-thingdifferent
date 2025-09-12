#task Generator
import numpy as np
import random


def generate_initial_numbers(n, range_list):
    numbers = []
    for _ in range(n):
        row = [
            random.randint(range_list[0][0], range_list[0][1]),
            random.randint(range_list[1][0], range_list[1][1]),
            random.randint(range_list[2][0], range_list[2][1]),
            random.randint(range_list[3][0], range_list[3][1]),
            random.randint(range_list[4][0], range_list[4][1]),
        ]
        numbers.append(row)
    return numbers


def save_file(filename, numbers):
    with open(filename, 'w') as file:
        for row in numbers:
            file.write(" ".join(map(str, row)) + "\n")


n = 1000 
range_list = [
    (10, 20),
    (20, 30), 
    (30, 40),
    (40, 50),
    (50, 60),
]
numbers = generate_initial_numbers(n, range_list=range_list)
save_file('numbers.txt', numbers)

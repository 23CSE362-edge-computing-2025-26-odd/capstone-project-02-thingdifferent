# Utilities Implementation
def parse_variables(file_path):
    variables = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=', 1)
            variables[key] = ast.literal_eval(value)
    return variables
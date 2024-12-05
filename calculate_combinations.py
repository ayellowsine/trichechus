import csv
from tabulate import tabulate

def calculate_combinations(parameters, parameter_names):
    total_combinations = 1
    table_data = []

    # Calculate the total number of combinations
    for name, values in zip(parameter_names, parameters):
        total_combinations *= values
        table_data.append([name, values])

    return total_combinations, table_data


def read_csv_file(filename):
    parameters = []
    parameter_names = []

    # Read the simplified CSV file
    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            parameter_names.append(row['Parameter Name'])
            try:
                values = int(row['Values'])
            except ValueError:
                values = 0  # Default to 0 if the value is missing or invalid
            parameters.append(values)

    return parameters, parameter_names


# Main script
csv_filename = 'parameters.csv'  # Path to your CSV file

# Read the parameter details from the CSV file
parameters, parameter_names = read_csv_file(csv_filename)

# Calculate the total number of combinations
total_combinations, table_data = calculate_combinations(parameters, parameter_names)

# Display the table of parameter details
print(tabulate(table_data, headers=["Parameter Name", "Number of Variations"], tablefmt="grid"))

# Display the total number of combinations
print(f"\nThe total number of combinations is: {total_combinations:,}")

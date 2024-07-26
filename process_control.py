import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data(file_path):
    """
    Reads data from a CSV or JSON file.
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        data = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Only CSV and JSON are supported.")
    return data

def calculate_control_limits(data, sigma=3):
    """
    Calculate the control limits for the process data.
    """
    mean = data.mean()
    std_dev = data.std()
    
    upper_control_limit = mean + sigma * std_dev
    lower_control_limit = mean - sigma * std_dev
    
    return mean, upper_control_limit, lower_control_limit

def plot_control_charts(data, control_limits):
    """
    Plot the control charts for the process data.
    """
    num_columns = data.shape[1]
    fig, axes = plt.subplots(num_columns, 1, figsize=(12, 4 * num_columns))

    if num_columns == 1:
        axes = [axes]

    for ax, column in zip(axes, data.columns):
        mean, ucl, lcl = control_limits[column]
        ax.plot(data[column], marker='o', linestyle='-', color='b')
        ax.axhline(y=mean, color='g', linestyle='--', label='Mean')
        ax.axhline(y=ucl, color='r', linestyle='--', label='UCL')
        ax.axhline(y=lcl, color='r', linestyle='--', label='LCL')
        ax.set_title(f'Control Chart for {column}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(pad=3.0)
    plt.show()

def check_process_control(data, control_limits):
    """
    Check if the process is in control.
    """
    out_of_control_points = {}
    for column in data.columns:
        ucl = control_limits[column][1]
        lcl = control_limits[column][2]
        out_of_control_points[column] = data[(data[column] > ucl) | (data[column] < lcl)].index.tolist()

    return out_of_control_points

def suggest_new_targets(data):
    """
    Suggest new targets based on the data distribution.
    """
    new_targets = {}
    for column in data.columns:
        new_targets[column] = data[column].mean()
    return new_targets

def main(file_path):
    # Read data
    data = read_data(file_path)

    # Calculate control limits for each column
    control_limits = {}
    for column in data.columns:
        control_limits[column] = calculate_control_limits(data[column])

    # Plot control charts
    plot_control_charts(data, control_limits)

    # Check if the process is in control
    out_of_control_points = check_process_control(data, control_limits)
    for column, points in out_of_control_points.items():
        if points:
            print(f"The process for {column} is out of control at points: {points}")
            new_target = suggest_new_targets(data)[column]
            print(f"Suggested new target for {column}: {new_target}")
        else:
            print(f"The process for {column} is in control.")

if __name__ == "__main__":
    file_path = input("Enter the path to the CSV or JSON file: ")
    main(file_path)

import numpy as np
import matplotlib.pyplot as plt

def eggholder_function(x, y):
    """
    Compute the Eggholder function for given x and y coordinates.
    f(x, y) = -(y + 47) * sin(sqrt(abs(x / 2 + (y + 47)))) - x * sin(sqrt(abs(x - (y + 47))))
    """

    term1 = -(y + 47.0) * np.sin(np.sqrt(np.abs(x / 2.0 + (y + 47.0))))
    term2 = -x * np.sin(np.sqrt(np.abs(x - (y + 47.0))))
    return term1 + term2


def hill_climbing_eggholder(max_stagnation = 100, step_size = 1.0, bounds = (-10000, 10000)):
    """
    Perform a single run of Hill Climbing algorithm (descending) on the Eggholder function.
    Terminates if no better position is found in the last 100 steps.
    """
    x = np.random.uniform(bounds[0], bounds[1])
    y = np.random.uniform(bounds[0], bounds[1])
    current_val = eggholder_function(x, y)

    stagnation_counter = 0

    while stagnation_counter < max_stagnation:
        x_new_position = (np.random.rand() - 0.5) * step_size + x
        y_new_position = (np.random.rand() - 0.5) * step_size + y

        if not (bounds[0] < x_new_position < bounds[1] and bounds[0] < y_new_position < bounds[1]):
            stagnation_counter += 1
            continue

        new_val = eggholder_function(x_new_position, y_new_position)

        if new_val < current_val:
            x, y = x_new_position, y_new_position
            current_val = new_val
            stagnation_counter = 0
        else:
            stagnation_counter += 1

    return (x, y), current_val


def run_experiment(num_runs=100):
    """
    Repeat the hill climbing algorithm for 100 runs and collects results
    """
    best_minima_values = []
    best_positions = []

    print(f"Running Hill Climbing algorithms on Eggholder function for {num_runs} runs...")
    for i in range(num_runs):
        pos, val = hill_climbing_eggholder()
        best_positions.append(pos)
        best_minima_values.append(val)

    return np.array(best_positions), np.array(best_minima_values)

def plot_distribution(minima_values):
    """
    Display the minima distribution on a plotted graph figure.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(minima_values, bins=20, color='skyblue', edgecolor='black',alpha=0.8)
    plt.title('Distribution of Minima Found by Hill Climbing (100 Runs)')
    plt.xlabel('Objective Function Value (Eggholder Minimum)')
    plt.ylable('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    positions, values = run_experiment(100)

    print("\n--- Experiment Summary ---")
    print(f"Best minimum value found across all runs: {np.min(values):.4f}")
    print(f"Mean minimum value: {np.mean(values):.4f}")

    plot_distribution(values)
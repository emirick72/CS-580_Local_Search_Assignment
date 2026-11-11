import numpy as np
import matplotlib.pyplot as plt

def calculate_attacks(state):
    """
    Compute the number of pairwise attacks in an N-Queens board state.
    State: index is the column number, value is the row number
    """
    n = len(state)
    attacks = 0
    for i in range(n):
        for j in range(i + 1, n):
            if state[i] == state[j]:
                attacks += 1
            elif abs(state[i] - state[j]) == abs(i - j):
                attacks += 1
    return attacks

def get_neighbors(state):
    """
    Generate all neighboring states by moving each queen to a different row in its respective column
    """
    n = len(state)
    neighbors = []
    for col in range(n):
        for row in range(n):
            if state[col] != row:
                neighbor = list(state)
                neighbor[col] = row
                neighbors.append(neighbor)
    return neighbors

def hill_climbing_nqueens(n, max_steps = 1000):
    """
    Perform a single run of Hill Climbing algorithm to minimize attacks for the N-Queens problem.
    """
    current_state = np.random.randint(0, n, size=n)
    current_attacks = calculate_attacks(current_state)
    initial_state = list(current_state)

    for _ in range(max_steps):
        if current_attacks == 0:
            break
        neighbors = get_neighbors(current_state)
        best_neighbor = current_state
        best_neighbor_attacks = current_attacks

        for neighbor in neighbors:
            n_attacks = calculate_attacks(neighbor)
            if n_attacks < best_neighbor_attacks:
                best_neighbor = np.array(neighbor)
                best_neighbor_attacks = n_attacks

        if best_neighbor_attacks >= current_attacks:
            break

        current_state = best_neighbor
        current_attacks = best_neighbor_attacks

    return initial_state, list(current_state), current_attacks

def run_nqueens_experiment(n, num_runs=100):
    """
    Run the experiment 100 times for a given N value (value is 8, 16, and 32),
        counting how many times a solution (0 attacks) is found.
    Save the first 10 runs' initial and final states for graph plot points.
    """
    success_count = 0
    first_10_runs = []

    print(f"Running N-Queens for N = {n} ({num_runs} runs)...")
    for i in range(num_runs):
        init_state, final_state, final_attacks = hill_climbing_nqueens(n)

        if final_attacks == 0:
            success_count += 1

        if i < 10:
            first_10_runs.append((init_state, final_state, final_attacks == 0))

    return success_count, first_10_runs

def plot_boards(runs_data, n):
    """
    Plot the initial and finals states of the game boards for the first 10 runs.
    """
    fig, axes = plt.subplots(10, 2, figsize=(6, 25))

    for i, (init_state, final_state, is_solved) in enumerate(runs_data):
        ax_init = axes[i, 0]
        board_init = np.zeros((n, n))
        for col, row in enumerate(init_state):
            board_init[row, col] = 1
        ax_init.imshow(board_init, cmap='Blues', interpolation='nearest')
        ax_init.set_title(f"Run {i+1} Initial", fontsize=9)
        ax_init.axis('off')

        ax_final = axes[i, 1]
        board_final = np.zeros((n, n))
        for col, row, in enumerate(final_state):
            board_final[row, col] = 1
        ax_final.imshow(board_final, cmap='Greens' if is_solved else 'Reds', interpolation='nearest')
        ax_final.set_title(f"Run {i+1} Final ({'Solved' if is_solved else 'Stuck'})", fontsize=9)
        ax_final.axis('off')

    plt.suptitle(f"N-Queens (N={n}) - First 10 Runs (Initial vs. Final)", fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    n_values = [8, 16, 32]

    for n in n_values:
        successes, first_10_runs = run_nqueens_experiment(n, num_runs=100)
        print(f"N = {n}: Successfully found solutions in {successes}/100 runs.\n")
        plot_boards(first_10_runs, n)
import time
import sys
import csv
import tkinter as tk
import numpy as np
import random
from collections import defaultdict

# Global variable for visualization
cell_size = 30
total_time_taken = 0

def load_maze(filename="maze.txt"):
    """Reads the maze from a text file and returns it as a 2D list."""
    import os
    
    # Try multiple possible locations for the maze file
    possible_paths = [
        filename,  # Try the direct filename first (for command line use)
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),  # Try script directory
        os.path.join(os.getcwd(), filename)  # Try current working directory
    ]
    
    for path in possible_paths:
        try:
            with open(path, "r") as f:
                maze = [list(line.strip()) for line in f]
            print(f"Successfully loaded maze from: {path}")
            return maze
        except FileNotFoundError:
            continue
    
    # If we got here, we couldn't find the file
    raise FileNotFoundError(f"Could not find maze file '{filename}' in any expected location")
    return maze

def find_start_goal(maze):
    """Finds the start (S) and goal (G) positions in the maze."""
    start, goal = None, None
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == 'S':
                start = (i, j)
            elif maze[i][j] == 'G':
                goal = (i, j)
    return start, goal

def create_mdp_from_maze(maze, goal):
    """
    Creates a sparse MDP representation from the maze.
    Only valid states and actions are stored to optimize memory.
    """
    rows, cols = len(maze), len(maze[0])
    directions = [(0, 1, 'RIGHT'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (-1, 0, 'UP')]
    
    # Using defaultdict for sparse representation
    states = set()
    actions = {'UP', 'DOWN', 'LEFT', 'RIGHT'}
    transitions = defaultdict(list)
    rewards = defaultdict(float)
    
    # Terminal state
    goal_state = goal
    
    # Find all valid states
    for i in range(rows):
        for j in range(cols):
            if maze[i][j] != '#':  # If not a wall
                state = (i, j)
                states.add(state)
                
                # Set high reward for reaching the goal
                if state == goal_state:
                    rewards[(state, None)] = 100
                    continue  # No transitions from goal state
                
                # Default small negative reward for each step
                for _, _, action in directions:
                    rewards[(state, action)] = -1
                
                # Calculate transitions
                for di, dj, action in directions:
                    next_i, next_j = i + di, j + dj
                    
                    # Check if the next position is valid
                    if (0 <= next_i < rows and 0 <= next_j < cols and maze[next_i][next_j] != '#'):
                        next_state = (next_i, next_j)
                        
                        # Deterministic transition
                        transitions[(state, action)].append((1.0, next_state))
                    else:
                        # Stay in place if hitting a wall
                        transitions[(state, action)].append((1.0, state))
    
    return states, actions, transitions, rewards, goal_state

def value_iteration(states, actions, transitions, rewards, goal_state, gamma=0.95, epsilon=1e-4):
    """
    Implements Value Iteration algorithm with sparse representation.
    Enhanced with value propagation for better path extraction.
    """
    # Initialize value function with small random values to break symmetry
    V = {state: random.uniform(-0.1, 0.1) for state in states}
    V[goal_state] = 100  # Initialize goal state with reward
    
    explored = []
    policy = {}
    
    iteration = 0
    delta = float('inf')
    
    # Set maximum iterations to prevent infinite loops
    max_iterations = 1000
    
    print(f"Starting Value Iteration with gamma={gamma}, epsilon={epsilon}")
    
    while delta > epsilon and iteration < max_iterations:
        delta = 0
        iteration += 1
        
        for state in states:
            explored.append(state)
            
            if state == goal_state:
                continue  # Skip goal state
                
            # Keep track of the previous value
            v_old = V[state]
            
            # Compute the value for each action
            action_values = {}
            for action in actions:
                if (state, action) in transitions:
                    # Calculate expected value for this action
                    next_value = 0
                    
                    # Track if this action leads to a new state
                    action_leads_to_new_state = False
                    
                    for prob, next_state in transitions[(state, action)]:
                        if next_state != state:
                            action_leads_to_new_state = True
                        reward = rewards[(state, action)]
                        next_value += prob * (reward + gamma * V[next_state])
                    
                    # Only consider actions that lead to a new state
                    if action_leads_to_new_state:
                        action_values[action] = next_value
            
            # If actions are available for this state
            if action_values:
                # Choose the action with max value
                best_action = max(action_values, key=action_values.get)
                V[state] = action_values[best_action]
                policy[state] = best_action
                
                # Update delta
                delta = max(delta, abs(v_old - V[state]))
        
        # Print progress every 10 iterations
        if iteration % 10 == 0:
            print(f"Value Iteration - Iteration {iteration}, Delta: {delta}")
    
    print(f"Value Iteration converged after {iteration} iterations")
    
    # Enhance value function by propagating goal values
    propagate_goal_values(V, transitions, goal_state, states, gamma)
    
    return policy, V, [], explored

def propagate_goal_values(V, transitions, goal_state, states, gamma, iterations=10):
    """Propagate high goal values backward to create a value gradient toward the goal."""
    print("Propagating goal values to enhance value function...")
    
    # Map from state to actions that can be taken from it
    state_to_actions = defaultdict(list)
    for (state, action) in transitions:
        state_to_actions[state].append(action)
    
    # Map from state to states that can reach it (reverse transitions)
    reverse_transitions = defaultdict(list)
    for (state, action) in transitions:
        for _, next_state in transitions[(state, action)]:
            if next_state != state:  # Only consider actual moves
                reverse_transitions[next_state].append(state)
    
    # Start from goal and work backwards
    current_states = {goal_state}
    visited = set()
    
    for _ in range(iterations):
        next_states = set()
        for state in current_states:
            visited.add(state)
            for prev_state in reverse_transitions[state]:
                if prev_state not in visited:
                    # Update value based on best neighbor
                    best_val = float('-inf')
                    for action in state_to_actions[prev_state]:
                        for _, next_s in transitions[(prev_state, action)]:
                            if next_s != prev_state:  # Only consider actual moves
                                val = -1 + gamma * V[next_s]  # Simple reward of -1 for each step
                                best_val = max(best_val, val)
                    
                    if best_val > V[prev_state]:
                        V[prev_state] = best_val
                    
                    next_states.add(prev_state)
        
        if not next_states:
            break
            
        current_states = next_states
    
    print("Value propagation complete")

def extract_path(start, goal, policy, maze, value_function):
    """Extracts a path from start to goal using the policy with enhanced path finding."""
    print("Extracting path from start to goal...")
    
    path = [start]
    current = start
    directions = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
    
    # Maximum steps as a safety measure
    max_steps = len(maze) * len(maze[0]) * 2
    steps = 0
    
    # Set to track visited states to avoid cycles
    visited = {start}
    
    while current != goal and steps < max_steps:
        next_state = None
        
        # First, try to use the policy if available
        if current in policy:
            action = policy[current]
            di, dj = directions[action]
            ni, nj = current[0] + di, current[1] + dj
            
            if (0 <= ni < len(maze) and 0 <= nj < len(maze[0]) and 
                maze[ni][nj] != '#' and (ni, nj) not in visited):
                next_state = (ni, nj)
        
        # If policy doesn't work, use value function to guide us
        if next_state is None:
            best_value = float('-inf')
            best_next = None
            
            for action, (di, dj) in directions.items():
                ni, nj = current[0] + di, current[1] + dj
                
                if (0 <= ni < len(maze) and 0 <= nj < len(maze[0]) and 
                    maze[ni][nj] != '#' and (ni, nj) not in visited):
                    next_val = value_function.get((ni, nj), float('-inf'))
                    
                    if next_val > best_value:
                        best_value = next_val
                        best_next = (ni, nj)
            
            next_state = best_next
        
        # If we still don't have a next state, try to find any valid move
        if next_state is None:
            for di, dj in directions.values():
                ni, nj = current[0] + di, current[1] + dj
                
                if (0 <= ni < len(maze) and 0 <= nj < len(maze[0]) and 
                    maze[ni][nj] != '#' and (ni, nj) not in visited):
                    next_state = (ni, nj)
                    break
        
        # If we have a valid next state, move there
        if next_state:
            current = next_state
            path.append(current)
            visited.add(current)
        else:
            # If we're stuck, try backtracking along the path
            if len(path) > 1:
                path.pop()  # Remove current
                current = path[-1]  # Go back to previous
                print(f"Backtracking to {current}")
            else:
                print("Cannot find path to goal")
                break
        
        steps += 1
        
        # If we're getting close to the maximum steps, print debug info
        if steps >= max_steps - 10:
            print(f"Warning: Approaching maximum steps. Current position: {current}, Goal: {goal}")
    
    if current == goal:
        print(f"Path found with {len(path)} steps")
    else:
        print("Failed to reach goal within step limit")
    
    return path

def calculate_memory_usage(value_function, policy):
    """Calculates memory usage for the value function and policy dictionaries."""
    memory_usage = sys.getsizeof(value_function) + sys.getsizeof(policy)
    
    # Add memory for dictionary entries
    for state, value in value_function.items():
        memory_usage += sys.getsizeof(state) + sys.getsizeof(value)
    
    for state, action in policy.items():
        memory_usage += sys.getsizeof(state) + sys.getsizeof(action)
    
    return memory_usage

def validate_policy(policy, maze):
    """Validates and fixes policy to ensure it doesn't lead to walls or out of bounds."""
    rows, cols = len(maze), len(maze[0])
    directions = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
    
    fixed_policy = policy.copy()
    fixed_count = 0
    
    for state, action in policy.items():
        i, j = state
        di, dj = directions[action]
        next_i, next_j = i + di, j + dj
        
        # Check if action leads to a valid cell
        if (next_i < 0 or next_i >= rows or 
            next_j < 0 or next_j >= cols or 
            maze[next_i][next_j] == '#'):
            
            # Find valid alternatives
            valid_actions = []
            for alt_action, (di, dj) in directions.items():
                ni, nj = i + di, j + dj
                if (0 <= ni < rows and 0 <= nj < cols and maze[ni][nj] != '#'):
                    valid_actions.append(alt_action)
            
            if valid_actions:
                # Choose an alternative action
                fixed_policy[state] = valid_actions[0]
                fixed_count += 1
                print(f"Fixed policy at {state}: {action} -> {valid_actions[0]}")
            else:
                # No valid actions, remove from policy
                del fixed_policy[state]
                print(f"Removed invalid state {state} from policy")
    
    print(f"Fixed {fixed_count} invalid policy actions")
    return fixed_policy

def save_metrics_to_csv(metrics, filename="metrics.csv"):
    """Saves the metrics to a CSV file with the specified format."""
    # Check if the file exists to write the header
    try:
        with open(filename, 'r') as file:
            pass  # File exists, no need to write header
    except FileNotFoundError:
        # File doesn't exist, write the header
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Algorithm", "Search Type", "Path Length", "Cells Explored", "Memory (KB)", "Time (Seconds)"])

    # Append the metrics to the CSV file
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(metrics)

def visualize_search(maze, path, explored, value_function=None):
    """Creates a Tkinter GUI to visualize the maze solving process with a size-limited window."""
    global total_time_taken
    window = tk.Tk()
    window.title("Value Iteration Maze Solver")

    # Get screen dimensions
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    
    # Calculate maze dimensions
    maze_height = len(maze)
    maze_width = len(maze[0])
    
    # Adjust cell size for large mazes
    global cell_size
    original_cell_size = cell_size
    
    # Calculate maximum window size (70% of screen)
    max_width = int(screen_width * 0.7)
    max_height = int(screen_height * 0.7)
    
    # Calculate what cell size would fit in max dimensions
    width_cell_size = max_width // maze_width
    height_cell_size = max_height // maze_height
    
    # Take the smaller of the two to ensure it fits both dimensions
    if maze_width > 50 or maze_height > 50:
        cell_size = min(width_cell_size, height_cell_size, original_cell_size)
        print(f"Adjusted cell size to {cell_size} for large maze")
    
    # Calculate canvas dimensions
    canvas_width = maze_width * cell_size
    canvas_height = maze_height * cell_size

    # Ensure window fits on screen
    total_height = canvas_height + 150  # Add space for info panel
    
    # Set window size and position it centered
    window.geometry(f"{canvas_width}x{total_height}")
    window.update_idletasks()  # Update to get actual window size
    
    # Center the window
    x_position = (screen_width - window.winfo_width()) // 2
    y_position = (screen_height - window.winfo_height()) // 2
    window.geometry(f"+{x_position}+{y_position}")

    # Create canvas for maze with scrollbars for very large mazes
    frame = tk.Frame(window)
    frame.pack(fill="both", expand=True)
    
    # Add scrollbars if the maze is large
    if canvas_width > max_width or canvas_height > max_height:
        # Create scrollbars
        h_scrollbar = tk.Scrollbar(frame, orient="horizontal")
        v_scrollbar = tk.Scrollbar(frame, orient="vertical")
        
        # Position scrollbars
        h_scrollbar.pack(side="bottom", fill="x")
        v_scrollbar.pack(side="right", fill="y")
        
        # Create canvas with scrollbars
        canvas = tk.Canvas(frame, width=min(canvas_width, max_width), 
                           height=min(canvas_height, max_height - 150),
                           bg="white",
                           xscrollcommand=h_scrollbar.set,
                           yscrollcommand=v_scrollbar.set)
        
        # Configure scrollbars
        h_scrollbar.config(command=canvas.xview)
        v_scrollbar.config(command=canvas.yview)
        
        # Configure canvas scroll region
        canvas.config(scrollregion=(0, 0, canvas_width, canvas_height))
    else:
        # Create canvas without scrollbars for smaller mazes
        canvas = tk.Canvas(frame, width=canvas_width, height=canvas_height, bg="white")
    
    canvas.pack(side="left", fill="both", expand=True)

    # Create info panel below the canvas
    info_panel = tk.Frame(window, height=150, bg="white")
    info_panel.pack(fill="x")

    # Title label
    title_label = tk.Label(info_panel, text="Value Iteration Maze Solver", font=("Arial", 16, "bold"), bg="white")
    title_label.pack()

    # Create a frame for the legend
    legend_frame = tk.Frame(info_panel, bg="white")
    legend_frame.pack()

    # Create legend items
    legend_explored = tk.Canvas(legend_frame, width=20, height=20, bg="gray")
    legend_explored.grid(row=0, column=0, padx=5)
    legend_label_explored = tk.Label(legend_frame, text="Explored Nodes", font=("Arial", 12), bg="white")
    legend_label_explored.grid(row=0, column=1)

    legend_path = tk.Canvas(legend_frame, width=20, height=20, bg="orange")
    legend_path.grid(row=0, column=2, padx=5)
    legend_label_path = tk.Label(legend_frame, text="Optimal Path", font=("Arial", 12), bg="white")
    legend_label_path.grid(row=0, column=3)

    legend_start = tk.Canvas(legend_frame, width=20, height=20, bg="green")
    legend_start.grid(row=0, column=4, padx=5)
    legend_label_start = tk.Label(legend_frame, text="Start Position", font=("Arial", 12), bg="white")
    legend_label_start.grid(row=0, column=5)

    legend_goal = tk.Canvas(legend_frame, width=20, height=20, bg="red")
    legend_goal.grid(row=0, column=6, padx=5)
    legend_label_goal = tk.Label(legend_frame, text="Goal Position", font=("Arial", 12), bg="white")
    legend_label_goal.grid(row=0, column=7)

    # Metrics Label (Initially Empty)
    metrics_label = tk.Label(info_panel, text="Metrics", font=("Arial", 12, "bold"), fg="#2E86C1", bg="white")
    metrics_label.pack()

    def draw_cell(x, y, color):
        """Draws a single cell on the canvas."""
        x1, y1 = y * cell_size, x * cell_size
        x2, y2 = x1 + cell_size, y1 + cell_size
        canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
        
        # Display value if available and cell size is big enough
        if value_function and (x, y) in value_function and cell_size >= 20:
            value = value_function[(x, y)]
            if abs(value) < 1000:  # Only show reasonable values
                canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, 
                                  text=f"{value:.1f}", font=("Arial", 8))

    # Draw the initial maze layout
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            color = "white"
            if maze[i][j] == '#':
                color = "black"
            elif maze[i][j] == 'S':
                color = "green"
            elif maze[i][j] == 'G':
                color = "red"
            draw_cell(i, j, color)

    # Use unique states for visualization
    unique_explored = list(dict.fromkeys(explored))
    explored_idx = 0

    def animate_explored():
        """Animates the explored cells."""
        nonlocal explored_idx
        if explored_idx < len(unique_explored):
            x, y = unique_explored[explored_idx]
            if maze[x][y] not in ['S', 'G']:  # Don't color start/goal
                draw_cell(x, y, "gray")
            explored_idx += 1
            window.after(5, animate_explored)  # Speed up for large mazes

    path_idx = 0

    def animate_path():
        """Animates the optimal path."""
        nonlocal path_idx
        if path_idx < len(path):
            x, y = path[path_idx]
            if maze[x][y] not in ['S', 'G']:  # Don't color start/goal
                draw_cell(x, y, "orange")
            path_idx += 1
            window.after(50, animate_path)  # Speed up for large mazes
        else:
            # Final time calculation
            elapsed_time = time.time() - start_time
            global total_time_taken
            total_time_taken = elapsed_time

            # Save metrics
            memory_usage = calculate_memory_usage(value_function or {}, {})

            # Update metrics in info panel
            metrics_text = f"📊 Metrics:\nCells Explored: {len(unique_explored)}\nPath Length: {len(path)}\nMemory Usage: {memory_usage / 1024:.2f} KB\nExecution Time: {total_time_taken:.2f} sec"
            metrics_label.config(text=metrics_text, font=("Arial", 12), fg="black")

            # Save metrics to CSV file
            save_metrics_to_csv(["Value Iteration", "MDP", len(path), len(unique_explored), memory_usage / 1024, total_time_taken])

    start_time = time.time()
    animate_explored()

    def start_path_animation():
        """Starts the path animation after the exploration animation is done."""
        animation_delay = 5 * len(unique_explored)  # Speed up for large mazes
        window.after(animation_delay, animate_path)

    window.after(5 * len(unique_explored), start_path_animation)  # Speed up for large mazes
    window.mainloop()
    
# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Value Iteration on a maze.')
    parser.add_argument('--maze', type=str, default='maze.txt', help='The maze file to use')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor (default: 0.95)')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Convergence threshold (default: 1e-4)')
    
    args = parser.parse_args()
    
    print("Running Value Iteration...")
    maze = load_maze(args.maze)
    start, goal = find_start_goal(maze)
    
    if start is None or goal is None:
        print("Error: Start or Goal not found in the maze!")
    else:
        # Create MDP from maze
        states, actions, transitions, rewards, goal_state = create_mdp_from_maze(maze, goal)
        
        # Run Value Iteration with specified parameters
        start_time = time.time()
        policy, value_function, _, explored = value_iteration(
            states, actions, transitions, rewards, goal_state, 
            gamma=args.gamma, epsilon=args.epsilon
        )
        
        # Validate policy before path extraction
        policy = validate_policy(policy, maze)
        
        # Extract path using policy and value function
        path = extract_path(start, goal, policy, maze, value_function)
        
        # Visualize
        visualize_search(maze, path, explored, value_function)
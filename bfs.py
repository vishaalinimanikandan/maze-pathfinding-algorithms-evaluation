import time
import sys
import csv
import tkinter as tk
from collections import deque

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

def bfs(maze, start, goal):
    """
    Performs Breadth-First Search (BFS) to find the shortest path from start to goal.
    Returns the path and the list of explored cells.
    """
    rows, cols = len(maze), len(maze[0])
    directions = [(0, 1, 'RIGHT'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (-1, 0, 'UP')]
    
    queue = deque()
    queue.append((start, []))  # (current_position, path)
    visited = set()
    explored = []

    while queue:
        (current, path) = queue.popleft()
        if current == goal:
            return path + [current], explored
        
        if current in visited:
            continue
        
        visited.add(current)
        explored.append(current)
        
        for di, dj, action in directions:
            next_i, next_j = current[0] + di, current[1] + dj
            next_state = (next_i, next_j)
            
            if (0 <= next_i < rows and 0 <= next_j < cols and 
                maze[next_i][next_j] != '#' and next_state not in visited):
                queue.append((next_state, path + [current]))
    
    return [], explored  # No path found

def calculate_memory_usage(explored, policy=None):
    """
    Calculates memory usage for the explored states and optionally policy.
    Handles both BFS (just explored states) and MDP algorithms (with value functions and policies)
    """
    memory_usage = sys.getsizeof(explored)
    
    # Add memory for explored states
    if isinstance(explored, dict):
        # For value function dictionaries
        for state, value in explored.items():
            memory_usage += sys.getsizeof(state) + sys.getsizeof(value)
    else:
        # For lists of explored states
        for state in explored:
            memory_usage += sys.getsizeof(state)
    
    # Add memory for policy if provided
    if policy is not None:
        memory_usage += sys.getsizeof(policy)
        for state, action in policy.items():
            memory_usage += sys.getsizeof(state) + sys.getsizeof(action)
    
    return memory_usage

def save_metrics_to_csv(metrics, filename="metrics.csv"):
    """Saves the metrics to a CSV file with the specified format."""
    try:
        with open(filename, 'r') as file:
            pass  # File exists, no need to write header
    except FileNotFoundError:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Algorithm", "Search Type", "Path Length", "Cells Explored", "Memory (KB)", "Time (Seconds)"])

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(metrics)

def visualize_search(maze, path, explored, value_function=None):
    """Creates a Tkinter GUI to visualize the maze solving process with a size-limited window."""
    global total_time_taken
    window = tk.Tk()
    window.title("BFS Maze Solver")

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
    title_label = tk.Label(info_panel, text="BFS Maze Solver", font=("Arial", 16, "bold"), bg="white")
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

            # Save metrics - for BFS we just pass the explored list
            memory_usage = calculate_memory_usage(explored)

            # Update metrics in info panel
            metrics_text = f"📊 Metrics:\nCells Explored: {len(unique_explored)}\nPath Length: {len(path)}\nMemory Usage: {memory_usage / 1024:.2f} KB\nExecution Time: {total_time_taken:.2f} sec"
            metrics_label.config(text=metrics_text, font=("Arial", 12), fg="black")

            # Save metrics to CSV file
            save_metrics_to_csv(["BFS", "Graph Search", len(path), len(unique_explored), memory_usage / 1024, total_time_taken])

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
    
    parser = argparse.ArgumentParser(description='Run BFS on a maze.')
    parser.add_argument('--maze', type=str, default='maze.txt', help='The maze file to use')
    
    args = parser.parse_args()
    
    print("Running BFS...")
    maze = load_maze(args.maze)
    start, goal = find_start_goal(maze)
    
    if start is None or goal is None:
        print("Error: Start or Goal not found in the maze!")
    else:
        start_time = time.time()
        path, explored = bfs(maze, start, goal)
        
        if path:
            print("Path found!")
        else:
            print("No path found!")
        
        visualize_search(maze, path, explored)
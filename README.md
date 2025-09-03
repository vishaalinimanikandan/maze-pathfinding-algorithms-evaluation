# Maze Pathfinding Algorithms: Performance Analysis

This repository presents an in-depth evaluation of five prominent maze-solving algorithms — Depth-First Search (DFS), Breadth-First Search (BFS), A*, Value Iteration, and Policy Iteration — applied to procedurally generated mazes of varying sizes.

## Algorithms Implemented
- **DFS (Depth-First Search)** — Stack-based search, not guaranteed to be optimal.
- **BFS (Breadth-First Search)** — Queue-based search, finds shortest path in unweighted graphs.
- **A\*** — Heuristic search using Manhattan distance for optimal and efficient solutions.
- **Value Iteration** — MDP-based iterative solution using Bellman optimality.
- **Policy Iteration** — MDP-based with alternating policy evaluation and improvement.

##  Features
- Random maze generation using iterative DFS.
- Real-time visualization of algorithm execution using `Tkinter`.
- Metrics tracked for each algorithm:
  - Path Length
  - Cells Explored
  - Execution Time
  - Memory Usage
  - Exploration Ratio
  - Path Optimality Ratio
  - Time/Memory per Cell
- Support for mazes of various sizes:
  - Small (9×9)
  - Medium (21×21 – 41×41)
  - Large (67×67 – 159×159)

## Project Structure
├── maze_generator.py # Maze creation script
├── bfs_solver.py # BFS algorithm with visualization
├── dfs_solver.py # DFS algorithm with visualization
├── astar_solver.py # A* search algorithm
├── value_iteration.py # MDP - Value Iteration implementation
├── policy_iteration.py # MDP - Policy Iteration implementation
├── utils/
│ ├── metrics.py # Memory/time usage and result logging
│ └── visualization.py # Shared Tkinter visualization methods
├── data/
│ ├── maze.txt # Sample maze files
│ └── metrics.csv # Logged results of all algorithms
├── README.md
└── requirements.txt


## 📊 Results Summary

| Algorithm         | Best Use Case                | Time Efficient | Memory Efficient | Optimal Path |
|------------------|------------------------------|----------------|------------------|--------------|
| A\*              | Large mazes with constraints | ✅             | ✅               | ✅           |
| BFS              | Small mazes, accurate search | ❌             | ✅               | ✅           |
| DFS              | Fast but non-optimal         | ✅             | ❌               | ❌           |
| Value Iteration  | Full maze policies           | ❌             | ❌               | ✅           |
| Policy Iteration | Optimal policy derivation    | ❌             | ❌               | ✅           |

## Installation

```bash
git clone https://github.com/yourusername/maze-pathfinding-algorithms-evaluation.git
cd maze-pathfinding-algorithms-evaluation
pip install -r requirements.txt

# Generate a new maze
python maze_generator.py

# Run BFS solver
python bfs_solver.py --maze maze.txt

# Run DFS solver
python dfs_solver.py --maze maze.txt

# Other scripts follow similar structure

## Dependencies

Python 3.8+

Tkinter (for visualization)

NumPy

Pandas


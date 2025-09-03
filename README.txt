# Maze Solving Algorithms - README

This project implements and compares five different maze-solving algorithms: three search-based (DFS, BFS, A*) and two based on Markov Decision Processes (Value Iteration and Policy Iteration).

## Requirements

- Python 3.6 or higher
- Required packages: numpy, matplotlib, pandas, tkinter

You can install the required packages using pip:
pip install numpy matplotlib pandas

Note: Tkinter usually comes bundled with Python installations.

## File Structure

- maze_generator.py: Script to generate random mazes of various sizes
- dfs.py: Implementation of Depth-First Search
- bfs.py: Implementation of Breadth-First Search
- A_star.py: Implementation of A* Search
- value_fix.py: Implementation of Value Iteration
- policy_fix.py: Implementation of Policy Iteration
- derived_metrics.py: Script to calculate additional performance metrics
- metrics_*size.csv: Raw performance data for different maze sizes
- all_metrics_combined.csv: Consolidated performance data

## How to Run

### 1. Generate a maze:
python maze_generator.py

Follow the prompts to specify maze dimensions. The maze will be saved as 'maze.txt'.

### 2. Run algorithms:

#### Depth-First Search:
python dfs.py --maze maze.txt

#### Breadth-First Search:
python bfs.py --maze maze.txt

#### A* Search:
python A_star.py --maze maze.txt

#### Value Iteration:
python value_iteration.py --maze maze.txt --gamma 0.95 --epsilon 1e-4

#### Policy Iteration:
python policy_iteration.py --maze maze.txt --gamma 0.95 --epsilon 1e-4

### 3. Calculate derived metrics:
python derived_metrics.py

This script will combine individual metrics files and calculate additional performance metrics. 

### Note 
The derived metrics script will work when appending the individual metrics csv file with actual maze size. 
For example: If the maze size is 41X41 then rename the metric file as "metrics_41size.csv"

## Command Line Arguments

- --maze: Path to the maze file (default: maze.txt)
- --gamma: Discount factor for MDP algorithms (default: 0.95)
- --epsilon: Convergence threshold for MDP algorithms (default: 1e-4)

## Output

Each algorithm produces:
- Visual representation of the maze solution
- Performance metrics saved to CSV files
- Animated visualization of the search process and final path

## Notes

- For large mazes, the visualization window will automatically adjust the cell size
- The MDP algorithms may take significantly longer on larger mazes
- All performance metrics are saved to the metrics_*size.csv files automatically
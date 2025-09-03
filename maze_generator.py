import random
import os
def create_maze(rows, cols):
    """Generates a random maze using iterative DFS."""
    # Initialize maze with walls ('#')
    maze = [['#' for _ in range(cols)] for _ in range(rows)]

    # Define possible movement directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def is_valid(nx, ny):
        """Checks if a position is valid for carving."""
        return 0 < nx < rows - 1 and 0 < ny < cols - 1 and maze[nx][ny] == '#'

    # Use an explicit stack for DFS
    stack = [(1, 1)]
    maze[1][1] = ' '  # Start position

    while stack:
        x, y = stack[-1]  # Get the last element
        random.shuffle(directions)  # Shuffle directions for randomness
        carved = False

        for dx, dy in directions:
            nx, ny = x + dx * 2, y + dy * 2  # Move two steps

            if is_valid(nx, ny):
                maze[x + dx][y + dy] = ' '  # Remove wall between
                maze[nx][ny] = ' '  # Open new cell
                stack.append((nx, ny))
                carved = True
                break  # Move in the chosen direction

        if not carved:
            stack.pop()  # Backtrack if no valid move

    # Place Start (S) and Goal (G) in valid locations
    maze[1][1] = 'S'
    goal_x, goal_y = rows - 2, cols - 2  # Default bottom-right goal

    # Ensure goal is placed in an open space
    while maze[goal_x][goal_y] == '#':
        goal_x, goal_y = random.randint(1, rows - 2), random.randint(1, cols - 2)
    maze[goal_x][goal_y] = 'G'

    return maze

def save_maze_to_file(maze, filename="maze.txt"):
    """Saves the generated maze to a text file."""
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create full path for the output file
    file_path = os.path.join(script_dir, filename)
    with open(filename, "w") as f:
        for row in maze:
            f.write("".join(row) + "\n")

def main():
    # Ask user for maze size
    rows = int(input("Enter number of rows (odd number >= 5): "))
    cols = int(input("Enter number of columns (odd number >= 5): "))

    # Ensure valid maze dimensions
    if rows % 2 == 0:
        rows += 1
    if cols % 2 == 0:
        cols += 1

    # Generate and save maze
    maze = create_maze(rows, cols)
    save_maze_to_file(maze)

    print(f"Maze generated and saved to 'maze.txt'.")

if __name__ == "__main__":
    main()

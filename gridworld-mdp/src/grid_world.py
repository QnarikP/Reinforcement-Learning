"""
This module implements a simple gridworld environment used to demonstrate state transitions and reward functions in reinforcement learning.
It defines a square grid with special states A and B (and their corresponding destination states A' and B') where specific rewards are given.
It also provides a drawing function to visualize either the state-value function or a policy grid (indicated by action arrows).
"""

import numpy as np
import matplotlib.pyplot as plt
import re

class GridWorld:
    """
    A gridworld environment for demonstrating reinforcement learning dynamics.

    Attributes:
        grid_size (int): The dimensions of the square grid.
        A_coordinates (list): Coordinates for special state A.
        A_prime_coordinates (list): Destination coordinates for state A.
        B_coordinates (list): Coordinates for special state B.
        B_prime_coordinates (list): Destination coordinates for state B.
        SPECIAL_STATES_LABELS (dict): Mapping of special state coordinates to their display labels.
        actions (list): List of numpy arrays representing possible actions (left, up, right, down).
        arrows (list): List of arrow symbols corresponding to actions.
    """

    def __init__(self, grid_size=5):
        self.grid_size = grid_size

        # Special states and their corresponding destination states.
        self.A_coordinates = [0, 1]
        self.A_prime_coordinates = [4, 1]
        self.B_coordinates = [0, 3]
        self.B_prime_coordinates = [2, 3]

        # Mapping of special state coordinates to labels for display.
        self.SPECIAL_STATES_LABELS = {
            tuple(self.A_coordinates): " (A)",
            tuple(self.A_prime_coordinates): " (A')",
            tuple(self.B_coordinates): " (B)",
            tuple(self.B_prime_coordinates): " (B')"
        }

        # Define possible actions (left, up, right, down) as numpy arrays.
        self.actions = [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0])]

        # Corresponding arrows for policy visualization.
        self.arrows = ['←', '↑', '→', '↓']

    def step(self, state, action):
        """
        Step from the current state to the next state.

        :param state: Current state as a list of two integers.
        :param action: Action taken in the current state (numpy array).
        :return: Tuple of next state (list) and obtained reward (float).
        """
        # Special transitions: from state A and state B yield fixed rewards and destinations.
        if state == self.A_coordinates:
            return self.A_prime_coordinates, 10
        if state == self.B_coordinates:
            return self.B_prime_coordinates, 5

        # Compute the next state by adding the action vector to the current state.
        next_state = (np.array(state) + action).tolist()
        x, y = next_state

        # If the next state is outside the grid, remain in the same state and incur a penalty.
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return state, -1.0

        # Otherwise, move to the next state with no reward.
        return next_state, 0

    def draw(self, grid, is_policy: bool = False, title: str = None):
        """
        Draw a grid representing either the state-value function or a policy.

        This method uses plt.imshow to render the grid with the 'Blues' colormap, centering the colormap at the grid's mean value.
        Each cell is annotated with either its numeric value or the best action arrow(s) (if displaying a policy),
        and special state labels (A, A', B, B') are appended as needed.

        Instead of taking a save path, this method accepts a title for the plot.
        When a title is provided, the method automatically replaces spaces with underscores and saves the plot in the
        "../generated_images" folder with the title as the file name.

        :param grid: 2D numpy array representing state values or policy data.
        :param is_policy: If True, the grid represents a policy (action arrows); otherwise, numeric state values.
        :param title: Title of the plot. If provided, the plot is saved in the generated images folder with spaces replaced by underscores.
        """
        # Create a new figure with a size based on the grid dimensions.
        plt.figure(figsize=(1.5 * self.grid_size, 1.5 * self.grid_size))

        # Calculate the mean of the grid to center the colormap.
        mean_value = np.mean(grid)
        # Create a normalization instance to center the colormap at the mean value.
        norm = plt.cm.colors.TwoSlopeNorm(vmin=np.min(grid), vcenter=mean_value, vmax=np.max(grid))

        # Display the grid using imshow with the 'Blues' colormap and custom normalization.
        plt.imshow(grid, cmap='Blues', norm=norm, interpolation='none')

        # Annotate each cell in the grid.
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                # Retrieve the current cell's value.
                cell_value = grid[i, j]

                # If the cell is numeric, convert it to a string for display.
                if isinstance(cell_value, (int, float, np.number)):
                    display_text = f"{cell_value}"
                else:
                    display_text = str(cell_value)

                # If displaying a policy, determine the best action(s) for the current cell.
                if is_policy:
                    next_values = []
                    for action in self.actions:
                        next_state, _ = self.step([i, j], action)
                        next_values.append(grid[next_state[0], next_state[1]])
                    # Get indices of the actions that yield the maximum value.
                    best_actions = np.where(np.array(next_values) == np.max(next_values))[0]
                    # Replace the display text with the corresponding arrow symbols.
                    display_text = ''.join(self.arrows[idx] for idx in best_actions)

                # Append special state labels if this cell's coordinates are in the labels dictionary.
                if (i, j) in self.SPECIAL_STATES_LABELS:
                    display_text += self.SPECIAL_STATES_LABELS[(i, j)]

                # Annotate the cell at position (j, i) because imshow's x corresponds to columns and y to rows.
                plt.text(j, i, display_text, ha='center', va='center', color='black', fontsize=12)

        # Set tick marks corresponding to grid cell centers.
        plt.xticks(np.arange(grid.shape[1]))
        plt.yticks(np.arange(grid.shape[0]))
        plt.title(f"{title}")
        # Invert the y-axis so that the first row is displayed at the top.
        # plt.gca().invert_yaxis()
        # Disable grid lines for clarity.
        plt.grid(False)

        # If a title is provided, construct a filename by replacing spaces with underscores and save the plot.
        if title:
            # Convert to lowercase and replace spaces and dots with underscores
            filename = re.sub(r"[^\w\s]", "", title.lower()).replace(" ", "_").replace(".", "_") + ".png"
            save_path = f"../generated_images/{filename}"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            # Otherwise, simply display the plot.
            plt.show()
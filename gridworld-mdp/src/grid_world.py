"""
This module implements a simple gridworld environment used to demonstrate state transitions and reward functions in reinforcement learning.
It defines a square grid with special states A and B (and their corresponding destination states A' and B') where specific rewards are given.
It also provides a drawing function to visualize either the state-value function or a policy grid (indicated by action arrows).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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

    def draw(self, grid, is_policy: bool = False, save_path: str = None):
        """
        Draw a grid representing either the state-value function or a policy.

        This method uses plt.imshow to render the grid with the 'Blues' colormap.
        It then annotates each cell with either the numeric value (if the cell contains a number)
        or with the best action arrows (if displaying a policy), and appends special state labels (A, A', B, B')
        if applicable. Detailed comments explain each step.

        :param grid: 2D numpy array representing state values or policy data.
        :param is_policy: If True, the grid represents a policy (action arrows); otherwise, numeric state values.
        :param save_path: Path to save the generated plot. If provided, the plot is saved instead of shown.
        """
        # Use the 'Blues' colormap and create a new figure.
        plt.figure(figsize=(1.5*self.grid_size, 1.5*self.grid_size))

        # Calculate the mean of the grid to use as the center for the colormap.
        mean_value = np.mean(grid)

        # Create a TwoSlopeNorm to center the colormap at the mean value.
        norm = colors.TwoSlopeNorm(vmin=np.min(grid), vcenter=mean_value, vmax=np.max(grid))

        # Display the grid as an image with the 'Blues' colormap and custom normalization.
        plt.imshow(grid, cmap='Blues', norm=norm, interpolation='none')

        # Iterate over each cell in the grid to add annotations.
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                # Fetch the current cell's value.
                cell_value = grid[i, j]

                # By default, if the cell contains a number, convert it to a string.
                # This check ensures we only format numeric values (int, float, or numpy number).
                if isinstance(cell_value, (int, float, np.number)):
                    display_text = f"{cell_value}"
                else:
                    display_text = str(cell_value)

                # If this grid is for a policy, compute the best action(s) for the current cell.
                if is_policy:
                    next_values = []
                    # For each action, get the next state and corresponding grid value.
                    for action in self.actions:
                        next_state, _ = self.step([i, j], action)
                        next_values.append(grid[next_state[0], next_state[1]])
                    # Find the indices of the actions that yield the maximum next state value.
                    best_actions = np.where(np.array(next_values) == np.max(next_values))[0]
                    # Use the corresponding arrow symbols for the best actions.
                    display_text = ''.join(self.arrows[idx] for idx in best_actions)

                # Before adding special state labels, check if the cell's coordinates exist in the labels dictionary.
                if (i, j) in self.SPECIAL_STATES_LABELS:
                    display_text += self.SPECIAL_STATES_LABELS[(i, j)]

                # Annotate the cell: plt.text uses (j, i) because imshow plots columns as x and rows as y.
                plt.text(j, i, display_text, ha='center', va='center', color='black', fontsize=12)

        # Set ticks to match the grid dimensions.
        plt.xticks(np.arange(grid.shape[1]))
        plt.yticks(np.arange(grid.shape[0]))
        # Invert the y-axis so that the first row is at the top.
        # plt.gca().invert_yaxis()
        # Remove the grid lines for clarity.
        plt.grid(False)

        # Save the plot if a path is provided; otherwise, display the plot.
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

class GridWorld:
    # region Constructor

    def __init__(self, grid_size=5):
        # region Summary
        """
        Initialize GridWorld environment.
        :param grid_size: Size of the square grid.
        """
        # endregion Summary

        # region Fields

        self.grid_size = grid_size  # Size of rectangular (square) gridworld

        # Coordinates of special state A
        self.A_coordinates = [0, 1]

        # Coordinates of special state A'
        self.A_prime_coordinates = [4, 1]

        # Coordinates of special state B
        self.B_coordinates = [0, 3]

        # Coordinates of special state B'
        self.B_prime_coordinates = [2, 3]

        # Possible 4 actions on a grid (left, up, right, down)
        self.actions = [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0])]

        # Arrows corresponding to actions
        self.arrows = ['←', '↑', '→', '↓']

        # endregion Fields

    # endregion Constructor

    # region Functions

    def step(self, state, action):
        # region Summary
        """
        Step from current state to next state.
        :param state: Current state (denoted as 𝑠)
        :param action: Action taken in current state (denoted as 𝑎)
        :return: Next state (denoted as 𝑠′) and obtained reward (denoted as 𝑟)
        """
        # endregion Summary

        # region Body

        # From state A, all 4 actions yield a reward of +10 and take the agent to A'.
        if state == self.A_coordinates:
            return self.A_prime_coordinates, 10

        # From state B, all 4 actions yield a reward of +5 and take the agent to B'.
        if state == self.B_coordinates:
            return self.B_prime_coordinates, 5

        # Next state is obtained by taking an action in the current state.
        next_state = (np.array(state) + action).tolist()

        # Get the next state's coordinates.
        x, y = next_state

        # Actions that would take the agent off the grid leave its location unchanged, but also result in a reward of -1.
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            reward = -1.0
            next_state = state
        else:  # Other actions result in a reward of 0, except those that move the agent out of the special states A and B.
            reward = 0

        return next_state, reward

        # endregion Body

    def draw(self, grid, is_policy=False):
        # region Summary
        """
        Draw grid of state-value function or grid of policy.
        :param grid: State-value function or policy grid.
        :param is_policy: True, if grid represents policy; otherwise, False.
        """
        # endregion Summary

        # region Body

        figure, axis = plt.subplots()
        axis.set_axis_off()
        table = Table(axis, bbox=[0, 0, 1, 1])

        width, height = 1.0 / grid.shape[1], 1.0 / grid.shape[0]

        # Add cells
        for (i, j), cell_value in np.ndenumerate(grid):
            if is_policy:
                # Create an empty list of next values
                next_values = []

                # For every action
                for action in self.actions:
                    # Get the current state
                    state = [i, j]

                    # Get the next state
                    next_state, _ = self.step(state, action)

                    # Append the grid's next state to the list of next values
                    next_values.append(grid[next_state[0], next_state[1]])

                # Get the best actions
                best_actions = np.where(next_values == np.max(next_values))[0]

                # Add the arrows corresponding to the best actions to the cell value
                cell_value = ''
                for best_action in best_actions:
                    cell_value += self.arrows[best_action]

            # Add special state labels to the cell value
            if [i, j] == self.A_coordinates:
                cell_value = str(cell_value) + " (A)"
            if [i, j] == self.A_prime_coordinates:
                cell_value = str(cell_value) + " (A')"
            if [i, j] == self.B_coordinates:
                cell_value = str(cell_value) + " (B)"
            if [i, j] == self.B_prime_coordinates:
                cell_value = str(cell_value) + " (B')"

            table.add_cell(i, j, width, height, text=cell_value, loc='center', facecolor='white')

        # Add external labels for row and column numbers
        for i in range(len(grid)):
            table.add_cell(i, -1, width, height, text=i, loc='right', edgecolor='none', facecolor='none')
            table.add_cell(-1, i, width, height / 2, text=i, loc='center', edgecolor='none', facecolor='none')

        axis.add_table(table)
        plt.show()

        # endregion Body

    # endregion Functions

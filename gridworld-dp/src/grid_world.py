import numpy as np
import matplotlib.pyplot as plt
import re

class GridWorld:
    def __init__(self, grid_size=4, discount=1.0, threshold=1e-4, in_place=True):
        """
        Initialize the gridworld environment.
        :param grid_size: Size of the grid (grid is square).
        :param discount: Discount factor (gamma).
        :param threshold: Convergence threshold.
        :param in_place: If True, update state-values in place.
        """
        self.grid_size = grid_size
        self.discount = discount
        self.threshold = threshold
        self.in_place = in_place
        # Define possible actions: left, up, right, down.
        self.actions = [np.array([0, -1]), np.array([-1, 0]),
                        np.array([0, 1]), np.array([1, 0])]
        # Each action is equally likely.
        self.action_probability = 0.25
        # Mapping of action indices to Unicode arrows.
        self.arrows = {0: '←', 1: '↑', 2: '→', 3: '↓'}
        # Optionally, you can define special state labels (e.g., start and goal)
        self.SPECIAL_STATES_LABELS = {}  # e.g., {(0,0): "A", (grid_size-1, grid_size-1): "B"}

    def is_terminal(self, state):
        """
        Check if the given state is terminal.
        :param state: [x, y] coordinates.
        :return: True if terminal; otherwise, False.
        """
        x, y = state
        return (x == 0 and y == 0) or (x == self.grid_size - 1 and y == self.grid_size - 1)

    def step(self, state, action):
        """
        Take a step in the gridworld.
        :param state: Current state [x, y].
        :param action: Action to take.
        :return: Tuple (next_state, reward).
        """
        if self.is_terminal(state):
            return state, 0

        next_state = (np.array(state) + action).tolist()
        x, y = next_state
        # If next state is off-grid, remain in the current state.
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            next_state = state

        # Reward is -1 for every transition.
        reward = -1
        return next_state, reward

    def get_policy_grid(self, state_values):
        """
        Compute the greedy policy based on the current state-value function.
        For each non-terminal state the method computes a one-step lookahead and returns
        the arrow(s) corresponding to the action(s) with maximum value.
        :param state_values: 2D numpy array of state values.
        :return: A 2D numpy object array containing arrow symbols (or 'T' for terminal states).
        """
        policy_grid = np.empty((self.grid_size, self.grid_size), dtype=object)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = [i, j]
                if self.is_terminal(state):
                    policy_grid[i, j] = 'T'
                else:
                    next_values = []
                    for action in self.actions:
                        next_state, reward = self.step(state, action)
                        q = reward + self.discount * state_values[next_state[0], next_state[1]]
                        next_values.append(q)
                    # Identify all actions that achieve the maximum value.
                    best_actions = np.where(np.array(next_values) == np.max(next_values))[0]
                    policy_grid[i, j] = ''.join(self.arrows[idx] for idx in best_actions)
        return policy_grid

    def draw(self, grid, is_policy: bool = False, title: str = None):
        """
        Draw a grid representing either the state-value function or a policy.

        This method uses plt.imshow to render the grid with the 'Blues' colormap, centering the colormap at the grid's mean value.
        Each cell is annotated with either its numeric value or the best action arrow(s) (if displaying a policy),
        and special state labels (if defined) are appended as needed.

        Instead of taking a save path, this method accepts a title for the plot.
        When a title is provided, the method automatically replaces spaces with underscores and saves the plot in the
        "../generated_images" folder with the title as the file name.

        :param grid: 2D numpy array representing state values.
        :param is_policy: If True, the grid represents a policy (action arrows); otherwise, numeric state values.
        :param title: Title of the plot. If provided, the plot is saved in the generated images folder.
        """
        plt.figure(figsize=(1.5 * self.grid_size, 1.5 * self.grid_size))
        # Calculate the mean of the grid (numeric conversion for state values)
        try:
            numeric_grid = grid.astype(np.float64)
        except (ValueError, TypeError):
            # If grid is not numeric (e.g., policy arrows), then use a dummy numeric grid for colormap.
            numeric_grid = np.zeros_like(grid, dtype=np.float64)

        mean_value = np.mean(numeric_grid)
        norm = plt.cm.colors.TwoSlopeNorm(vmin=np.min(numeric_grid), vcenter=mean_value, vmax=np.max(numeric_grid))
        plt.imshow(numeric_grid, cmap='Blues', norm=norm, interpolation='none')

        # Annotate each cell.
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                cell_value = grid[i, j]
                # Default: if cell_value is numeric, show it.
                if isinstance(cell_value, (int, float, np.number)):
                    display_text = f"{round(cell_value, 2)}"
                else:
                    display_text = str(round(cell_value, 2))

                if is_policy and not isinstance(cell_value, (str)):
                    # When drawing a policy from state-values, compute best actions.
                    next_values = []
                    for action in self.actions:
                        next_state, _ = self.step([i, j], action)
                        next_values.append(numeric_grid[next_state[0], next_state[1]])
                    best_actions = np.where(np.array(next_values) == np.max(next_values))[0]
                    display_text = ''.join(self.arrows[idx] for idx in best_actions)

                # Append special state labels if defined.
                if (i, j) in self.SPECIAL_STATES_LABELS:
                    display_text += self.SPECIAL_STATES_LABELS[(i, j)]
                plt.text(j, i, display_text, ha='center', va='center', color='black', fontsize=12)

        plt.xticks(np.arange(grid.shape[1]))
        plt.yticks(np.arange(grid.shape[0]))
        plt.title(f"{title}")
        plt.grid(False)

        if title:
            # Construct filename by replacing spaces and punctuation.
            filename = re.sub(r"[^\w\s]", "", title.lower()).replace(" ", "_").replace(".", "_") + ".png"
            save_path = f"../generated_images/{filename}"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def compute_state_value(self, show_policy=False, show_arrows=False):
        """
        Compute state-values using iterative policy evaluation.
        Every 5 iterations, if show_policy is True:
          - If show_arrows is False, the state-value grid is drawn.
          - If show_arrows is True, the greedy policy (arrows) is computed and drawn,
            but only if at least one arrow has changed relative to the previous drawn policy.
        :param show_policy: Whether to draw the grid every 5 iterations.
        :param show_arrows: When True, draw policy arrows instead of numeric state values.
        :return: Tuple (final state_values, number of iterations).
        """
        V = np.zeros((self.grid_size, self.grid_size))
        iteration = 0
        previous_policy = None

        while True:
            state_values = V if self.in_place else V.copy()
            old_values = state_values.copy()

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    value = 0
                    state = [i, j]
                    for action in self.actions:
                        next_state, reward = self.step(state, action)
                        value += self.action_probability * (
                            reward + self.discount * state_values[next_state[0], next_state[1]]
                        )
                    V[i, j] = value

            iteration += 1

            if show_policy and iteration % 5 == 0:
                if show_arrows:
                    # Compute current greedy policy as an array of arrows.
                    current_policy = self.get_policy_grid(V)
                    # Draw only if the policy has changed.
                    if previous_policy is None or not np.array_equal(current_policy, previous_policy):
                        self.draw(V, is_policy=False, title=f"Iteration {iteration} (Policy)")
                        previous_policy = current_policy.copy()
                else:
                    self.draw(V, is_policy=False, title=f"Iteration {iteration} (State Values)")

            # Check convergence.
            if np.max(np.abs(old_values - V)) < self.threshold:
                break

        return V, iteration


# Example usage:
if __name__ == '__main__':
    env = GridWorld(grid_size=4, discount=1.0, threshold=1e-4, in_place=True)

    # Compute state-values. To see drawings every 5 iterations:
    # - Set show_policy=True.
    # - Set show_arrows=True to display policy arrows (only when the policy changes).
    state_values, num_iter = env.compute_state_value(show_policy=True, show_arrows=True)

    print("Final state values after {} iterations:".format(num_iter))
    print(state_values)
    # Draw the final state-value table.
    env.draw(state_values, is_policy=True, title="Final State Values")

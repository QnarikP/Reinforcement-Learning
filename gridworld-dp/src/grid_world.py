import numpy as np
import matplotlib.pyplot as plt
import re


class GridWorld:
    def __init__(self, grid_size=4, discount=1.0, threshold=1e-4, in_place=True):
        """
        Initialize the GridWorld environment.

        The environment represents a square grid where each non-terminal state has
        four possible actions (left, up, right, down).
        Initially, all actions have equal probability (0.25).
        The state-value function V(s) is computed via
        iterative policy evaluation (under the fixed uniform policy), and then used
        to improve the policy by choosing actions that maximize the expected return.

        :param grid_size: Size of the grid (grid is square).
        :param discount: Discount factor (gamma).
        :param threshold: Convergence threshold for value updates.
        :param in_place: If True, update state-values in place.
        """
        self.grid_size = grid_size
        self.discount = discount
        self.threshold = threshold
        self.in_place = in_place
        # Define possible actions: left, up, right, down.
        self.actions = [np.array([0, -1]), np.array([-1, 0]),
                        np.array([0, 1]), np.array([1, 0])]
        # Each action is initially equally likely.
        self.action_probability = 0.25
        # Mapping of action indices to Unicode arrows.
        self.arrows = {0: '←', 1: '↑', 2: '→', 3: '↓'}
        # For labeling, also define full action names.
        self.action_names = {0: "Left", 1: "Up", 2: "Right", 3: "Down"}
        # Special state labels (if needed).
        self.SPECIAL_STATES_LABELS = {}  # e.g., {(0, 0): "A", (grid_size-1, grid_size-1): "B"}

    def is_terminal(self, state):
        """
        Check if the given state is terminal.

        :param state: [x, y] coordinates.
        :return: True if the state is terminal; otherwise, False.
        """
        x, y = state
        return (x == 0 and y == 0) or (x == self.grid_size - 1 and y == self.grid_size - 1)

    def step(self, state, action):
        """
        Execute a step in the gridworld.

        :param state: Current state [x, y].
        :param action: Action to take.
        :return: Tuple (next_state, reward).
        """
        if self.is_terminal(state):
            return state, 0

        next_state = (np.array(state) + action).tolist()
        x, y = next_state
        # If the next state is off-grid, remain in the current state.
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            next_state = state

        # Reward is -1 for every transition.
        reward = -1
        return next_state, reward

    def get_policy_grid(self, state_values):
        """
        Compute the greedy policy (action arrows) based on the current state-value function.

        For each non-terminal state, this method computes a one-step lookahead and selects
        the action(s) with the maximum Q-value (r + gamma * V(s')).
        It returns a 2D array (grid) of arrow symbols (or 'T' for terminal states) for visualization.

        :param state_values: 2D numpy array of state values.
        :return: 2D numpy object array with arrow symbols.
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
                    best_actions = np.where(np.array(next_values) == np.max(next_values))[0]
                    policy_grid[i, j] = ''.join(self.arrows[idx] for idx in best_actions)
        return policy_grid

    def improve_policy(self, state_values):
        """
        Improve the current policy using a one-step lookahead based on the state-value function.

        For each non-terminal state, the method computes the Q-value for each action and then
        assigns equal probability among the actions that maximize the Q-value.
        For terminal states, the policy is set to zeros.

        The returned policy is a 2D numpy array of shapes (grid_size**2, 4), where each row
        corresponds to a state (flattened) and each column to an action.

        :param state_values: 2D numpy array of state values.
        :return: 2D numpy array of shape (grid_size**2, 4) representing the improved policy probabilities.
        """
        policy = np.zeros((self.grid_size, self.grid_size, len(self.actions)))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = [i, j]
                if self.is_terminal(state):
                    policy[i, j, :] = 0
                else:
                    q_values = []
                    for action in self.actions:
                        next_state, reward = self.step(state, action)
                        q = reward + self.discount * state_values[next_state[0], next_state[1]]
                        q_values.append(q)
                    q_values = np.array(q_values)
                    max_q = np.max(q_values)
                    best_actions = (q_values == max_q)
                    num_best = np.sum(best_actions)
                    policy[i, j, best_actions] = 1.0 / num_best
        # Flatten the policy: shape becomes (grid_size**2, number_of_actions)
        return policy.reshape((self.grid_size * self.grid_size, len(self.actions)))

    def draw(self, grid, is_policy: bool = False, title: str = None):
        """
        Draw a grid representing either the state-value function or a policy.

        This method uses plt.imshow to render the grid with the 'Blues' colormap,
        centering the colormap on the grid's mean value.
        Each cell is annotated with either its numeric value or (if is_policy is True) the best action arrow(s)
        computed from a one-step lookahead on the state-value function.

        If a title is provided, the plot is saved in the "../generated_images" folder.

        :param grid: 2D numpy array representing state values or a policy grid.
        :param is_policy: If True, the grid is interpreted as a policy (action arrows); otherwise, numeric state values.
        :param title: Title of the plot.
        If provided, the plot is saved.
        """
        plt.figure(figsize=(1.5 * self.grid_size, 1.5 * self.grid_size))
        try:
            numeric_grid = grid.astype(np.float64)
        except (ValueError, TypeError):
            numeric_grid = np.zeros_like(grid, dtype=np.float64)

        mean_value = np.mean(numeric_grid)
        norm = plt.cm.colors.TwoSlopeNorm(vmin=np.min(numeric_grid), vcenter=mean_value, vmax=np.max(numeric_grid))
        plt.imshow(numeric_grid, cmap='Blues', norm=norm, interpolation='none')

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                cell_value = grid[i, j]
                if isinstance(cell_value, (int, float, np.number)):
                    display_text = f"{round(cell_value, 2)}"
                else:
                    display_text = str(cell_value)

                if is_policy and not isinstance(cell_value, str):
                    next_values = []
                    for action in self.actions:
                        next_state, _ = self.step([i, j], action)
                        next_values.append(numeric_grid[next_state[0], next_state[1]])
                    best_actions = np.where(np.array(next_values) == np.max(next_values))[0]
                    display_text = ''.join(self.arrows[idx] for idx in best_actions)

                if (i, j) in self.SPECIAL_STATES_LABELS:
                    display_text += self.SPECIAL_STATES_LABELS[(i, j)]
                plt.text(j, i, display_text, ha='center', va='center', color='black', fontsize=12)

        plt.xticks(np.arange(grid.shape[1]))
        plt.yticks(np.arange(grid.shape[0]))
        plt.title(f"{title}")
        plt.grid(False)

        if title:
            filename = re.sub(r"[^\w\s]", "", title.lower()).replace(" ", "_").replace(".", "_") + ".png"
            save_path = f"../generated_images/{filename}"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def draw_policy_probabilities(self, policy, title: str = None, show_arrows: bool = False):
        """
        Draw a grid visualizing the policy probabilities matrix as it is.

        The policy is expected to be a 2D numpy array with shape (grid_size**2, number_of_actions)
        where each row corresponds to a state and each column to an action.
        The x-axis is labeled with action names ("Left", "Up", "Right", "Down") and the y-axis with the state indices.

        If show_arrows is True, only the arrow symbol(s) for the best action(s) (i.e., with maximum probability)
        are displayed in each row.
        Otherwise, only the numerical probabilities are shown.

        :param policy: 2D numpy array of shape (grid_size**2, 4) representing action probabilities.
        :param title: Title of the plot.
        If provided, the plot is saved.
        :param show_arrows: If True, display only the best action arrow(s); otherwise, display only numerical probabilities.
        """
        # Create a figure sized appropriately for a matrix of shape (grid_size**2, 4).
        num_states = self.grid_size * self.grid_size
        num_actions = len(self.actions)
        plt.figure(figsize=(num_actions * 2, num_states * 0.5))

        # Plot the policy matrix.
        plt.imshow(policy, cmap='Blues', interpolation='none')

        # Set x-ticks with action names and y-ticks with state indices.
        plt.xticks(np.arange(num_actions), [self.action_names[i] for i in range(num_actions)])
        plt.yticks(np.arange(num_states), [str(i) for i in range(num_states)])

        # Annotate each cell with either the probability value or arrow(s).
        for i in range(num_states):
            for j in range(num_actions):
                cell_value = policy[i, j]
                if show_arrows:
                    # For each state (row), find the best action(s)
                    best_actions = np.where(policy[i, :] == np.max(policy[i, :]))[0]
                    # Only show arrows in the cell if this column is among the best actions.
                    display_text = self.arrows[j] if j in best_actions else ""
                else:
                    display_text = f"{round(cell_value, 2)}"
                plt.text(j, i, display_text, ha='center', va='center', color='black', fontsize=10)

        plt.title(f"{title}")
        plt.grid(False)

        if title:
            filename = re.sub(r"[^\w\s]", "", title.lower()).replace(" ", "_").replace(".", "_") + ".png"
            save_path = f"../generated_images/{filename}"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def compute_state_value(self, show_policy=False, show_arrows=False):
        """
        Compute the state-value function V(s) using iterative policy evaluation.

        The current implementation uses a fixed policy (equal probability 0.25 for all actions)
        to update V(s) using the Bellman equation:

            V(s) = sum_a [ π(a|s) * (r(s, a) + γ * V(s')) ]

        Where π(a|s) is 0.25 for all actions.
        This method returns the final state-value function.
        Optionally, it visualizes the state values (or the greedy policy based on V) every 5 iterations.

        :param show_policy: Whether to draw the grid every 5 iterations.
        :param show_arrows: When True, draw policy arrows computed from V; otherwise, draw state values.
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
                    current_policy = self.get_policy_grid(V)
                    if previous_policy is None or not np.array_equal(current_policy, previous_policy):
                        self.draw(V, is_policy=True, title=f"Iteration {iteration} (Policy Arrows)")
                        previous_policy = current_policy.copy()
                else:
                    self.draw(V, is_policy=False, title=f"Iteration {iteration} (State Values)")

            if np.max(np.abs(old_values - V)) < self.threshold:
                break

        return V, iteration


# Example usage:
if __name__ == '__main__':
    env = GridWorld(grid_size=4, discount=1.0, threshold=1e-4, in_place=True)

    # Compute the state-value function V(s) using iterative policy evaluation.
    state_values, num_iter = env.compute_state_value(show_policy=False, show_arrows=False)
    print("Final state values after {} iterations:".format(num_iter))
    print(state_values)

    # Visualize the final state-value function.
    env.draw(state_values, is_policy=False, title="Final State Values")

    # Improve the policy based on the computed state values.
    improved_policy = env.improve_policy(state_values)
    # improved_policy now has shape (grid_size**2, 4)

    # Visualize the improved policy probabilities.
    # Set show_arrows=False to display only numerical probabilities.
    env.draw_policy_probabilities(improved_policy, title="Improved Policy Probabilities", show_arrows=False)
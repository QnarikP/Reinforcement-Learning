import numpy as np
from tqdm import tqdm

# region Hyper-parameters
# Constants for the random walk problem
# Denote: i=0 is the left terminal state; i=1,2,3,4,5 represent the non-terminal states A,B,C,D,E; i=6 is the right terminal state.
# For convenience, we assume all rewards are 0, and the left terminal state has value 0, the right terminal state has value 1.
# This trick has been used in Gambler's Problem.

# Global constants
NUM_STATES = 7
NON_TERMINAL_STATES = range(1, 6)
LEFT_TERMINAL = 0
RIGHT_TERMINAL = 6
START_STATE = 3  # Center state C


# endregion Hyper-parameters


class RandomWalkEnvironment:
    """Environment for the Random Walk problem"""

    def __init__(self):
        # Actions
        self.ACTIONS = {"left": 0, "right": 1}

        # True state-values
        self.true_values = np.zeros(NUM_STATES)
        # The true value of each non-terminal state is the probability of terminating on the right if starting from that state
        self.true_values[NON_TERMINAL_STATES] = np.arange(1, 6) / 6.0
        # The true value of the right terminal state
        self.true_values[RIGHT_TERMINAL] = 1.0

    def reset(self):
        """Reset the environment to the start state"""
        return START_STATE

    def step(self, state, action=None):
        """
        Take a step in the environment

        Args:
            state: Current state
            action: Action to take (if None, choose randomly)

        Returns:
            next_state: Next state
            reward: Reward received
            done: Whether the episode is done
        """
        # If action is not provided, choose randomly
        if action is None:
            action = np.random.binomial(n=1, p=0.5)

        # Update state based on action
        if action == self.ACTIONS["left"]:
            next_state = state - 1
        else:
            next_state = state + 1

        # Determine if the episode is done
        done = next_state == LEFT_TERMINAL or next_state == RIGHT_TERMINAL

        # Determine reward (only +1 when reaching right terminal state)
        reward = 1.0 if next_state == RIGHT_TERMINAL else 0.0

        return next_state, reward, done


class ValueFunctionApproximator:
    """Approximates the state-value function"""

    def __init__(self):
        # Initialize approximate values
        self.values = np.zeros(NUM_STATES)
        # Initialize the approximate values of non-terminal states to the intermediate value
        self.values[NON_TERMINAL_STATES] = 0.5
        # The approximate value of the right terminal state
        self.values[RIGHT_TERMINAL] = 1.0

    def reset(self):
        """Reset the value function to initial values"""
        self.values = np.zeros(NUM_STATES)
        self.values[NON_TERMINAL_STATES] = 0.5
        self.values[RIGHT_TERMINAL] = 1.0

    def copy(self):
        """Return a copy of the current values"""
        return np.copy(self.values)

    def update(self, state, target, step_size):
        """
        Update the value function for a state

        Args:
            state: State to update
            target: Target value
            step_size: Step size for update
        """
        self.values[state] += step_size * (target - self.values[state])


class Agent:
    """Agent that learns using Monte Carlo or TD methods"""

    def __init__(self, env):
        self.env = env
        self.value_function = ValueFunctionApproximator()

    def monte_carlo_episode(self, step_size=0.1, batch=False):
        """
        Run a Monte Carlo episode

        Args:
            step_size: Step size for updates
            batch: Whether to update values (False) or just return trajectories (True)

        Returns:
            states_trajectory: List of states visited
            rewards: List of rewards received
        """
        # Reset the environment
        state = self.env.reset()

        # Add the 1st state to the trajectory of states
        states_trajectory = [state]

        # Episode trajectory
        while True:
            # Take a random action
            next_state, reward, done = self.env.step(state)

            # Append the new state to the trajectory of states
            states_trajectory.append(next_state)

            # Break if terminal state
            if done:
                break

            # Update state
            state = next_state

        # For Monte Carlo, we use the terminal reward for all states
        rewards = [reward] * (len(states_trajectory) - 1)

        # Update value function if not in batch mode
        if not batch:
            for visited_state in states_trajectory[:-1]:
                self.value_function.update(visited_state, reward, step_size)

        return states_trajectory, rewards

    def temporal_difference_episode(self, step_size=0.1, batch=False):
        """
        Run a Temporal Difference episode

        Args:
            step_size: Step size for updates
            batch: Whether to update values (False) or just return trajectories (True)

        Returns:
            states_trajectory: List of states visited
            rewards: List of rewards received
        """
        # Reset the environment
        state = self.env.reset()

        # Add the 1st state to the trajectory of states
        states_trajectory = [state]

        # Create a list of rewards (initialized with 0 for proper indexing with states)
        rewards = []

        # Episode trajectory
        while True:
            # Preserve the old state
            old_state = state

            # Take a random action
            next_state, reward, done = self.env.step(state)

            # Append the new state to the trajectory of states
            states_trajectory.append(next_state)

            # Update value function if not in batch mode
            if not batch:
                # TD update (Equation (6.2))
                target = reward + self.value_function.values[next_state]
                self.value_function.update(old_state, target, step_size)

            # Break if terminal state
            # if done:
            #     break
            # **always** record this reward, even if it ends the episode**
            rewards.append(reward)
            if done:
                break
            # Update state
            state = next_state

            # Append the reward to the list of rewards
            rewards.append(reward)

        return states_trajectory, rewards

    def batch_learning(self, method, episodes, step_size=0.001, threshold=1e-3, runs=100):
        """
        Batch updating for MC or TD methods

        Args:
            method: "TD" or "MC"
            episodes: Number of episodes
            step_size: Step size parameter
            threshold: Threshold for convergence
            runs: Number of independent runs

        Returns:
            total_errors: Array of errors for each episode, averaged over runs
        """
        # Create an array of total errors filled with 0s
        total_errors = np.zeros(episodes)

        # Reference to the true values for error calculation
        true_values = self.env.true_values

        # For every run
        for _ in tqdm(range(runs)):
            # Initialize a fresh value function
            current_values = np.zeros(NUM_STATES)
            current_values[NON_TERMINAL_STATES] = -1  # Initial value as per the original code
            current_values[RIGHT_TERMINAL] = 1.0

            # Lists to store trajectories and rewards
            states_trajectories = []
            rewards_list = []

            # For every episode
            for episode in range(episodes):
                # Run episode and collect trajectory
                if method == 'TD':
                    trajectory, reward = self.temporal_difference_episode(batch=True)
                else:  # Monte Carlo
                    trajectory, reward = self.monte_carlo_episode(batch=True)

                # Store trajectory and rewards
                states_trajectories.append(trajectory)
                rewards_list.append(reward)

                # Batch update until convergence
                while True:
                    updates = np.zeros(NUM_STATES)

                    # Process all trajectories seen so far
                    for traj, reward in zip(states_trajectories, rewards_list):
                        for i in range(len(traj) - 1):
                            if method == 'TD':
                                # TD update
                                updates[traj[i]] += reward[i] + current_values[traj[i + 1]] - current_values[traj[i]]
                            else:  # Monte Carlo
                                # MC update
                                updates[traj[i]] += reward[i] - current_values[traj[i]]

                    # Apply updates
                    updates *= step_size

                    # Check for convergence
                    if np.sum(np.abs(updates)) < threshold:
                        break

                    # Update values
                    current_values += updates

                # Calculate RMSE between true and current values
                rmse = np.sqrt(np.sum(np.power(current_values - true_values, 2)) / 5.0)

                # Add to total errors
                total_errors[episode] += rmse

        # Average errors over runs
        total_errors /= runs

        return total_errors


# region Main function
def run_experiment(episodes=100, runs=100):
    """
    Run the experiment comparing MC and TD methods

    Args:
        episodes: Number of episodes per run
        runs: Number of independent runs
    """
    # Create environment and agent
    env = RandomWalkEnvironment()
    agent = Agent(env)

    # Run batch learning for MC and TD
    mc_errors = agent.batch_learning("MC", episodes, runs=runs)
    td_errors = agent.batch_learning("TD", episodes, runs=runs)

    # Return results
    return mc_errors, td_errors


if __name__ == "__main__":
    mc_errors, td_errors = run_experiment()

    # Print results or plot them
    print("Monte Carlo final error:", mc_errors[-1])
    print("Temporal Difference final error:", td_errors[-1])

    # Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(mc_errors, label='MC')
        plt.plot(td_errors, label='TD')
        plt.xlabel('Episodes')
        plt.ylabel('RMS Error')
        plt.legend()
        plt.title('Comparison of MC and TD methods')
        plt.savefig('random_walk_comparison.png')
        plt.show()
    except ImportError:
        print("matplotlib not available for plotting")
# endregion Main function
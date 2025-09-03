import numpy as np

# region Hyper-parameters

# Number of non-terminal states
num_states = 19

# Discount factor (γ)
discount_factor = 1

# Set of non-terminal states
non_terminal_states = np.arange(1, num_states + 1)

# Terminal states:
# - Transition to left terminal gives reward = -1
# - Transition to right terminal gives reward = 1
terminal_states = [0, num_states + 1]

# True state-values of non-terminal states from the Bellman equation
true_values = np.arange(-20, 22, 2) / 20.0

# True values of terminal states are zero
true_values[0] = true_values[-1] = 0

# Starting state (middle of the chain)
start_state = 10

# endregion Hyper-parameters

# region Functions

def temporal_difference(value_estimates, n_steps, alpha):
    # region Summary
    """
    n-step TD Method
    :param value_estimates: Current estimates of state values (V)
    :param n_steps: Number of steps for TD update (n)
    :param alpha: Step-size parameter (α)
    """
    # endregion Summary

    # region Body

    # Initialize the episode at the starting state
    current_state = start_state

    # Sequence of visited states
    visited_states = [current_state]

    # Sequence of observed rewards
    observed_rewards = [0]

    # Time index
    time_step = 0

    # Episode termination time
    termination_time = float('inf')

    while True:
        # Advance time
        time_step += 1

        # If an episode is still running
        if time_step < termination_time:
            # Random action: step left or right
            if np.random.binomial(n=1, p=0.5) == 1:
                next_state = current_state + 1
            else:
                next_state = current_state - 1

            # Assign rewards based on terminal states
            if next_state == 0:               # left terminal
                reward = -1
            elif next_state == num_states:    # right terminal
                reward = 1
            else:                             # non-terminal
                reward = 0

            # Store trajectory
            visited_states.append(next_state)
            observed_rewards.append(reward)

            # If the terminal state reached → mark termination
            if next_state in terminal_states:
                termination_time = time_step

        # Time of state to update
        update_time = time_step - n_steps

        if update_time >= 0:
            estimated_return = 0.0

            # Compute n-step return (Equation 7.1)
            for k in range(update_time + 1, min(termination_time, update_time + n_steps) + 1):
                estimated_return += (discount_factor ** (k - update_time - 1)) * observed_rewards[k]

            # If an episode not finished yet → bootstrap from value estimate
            if update_time + n_steps <= termination_time:
                bootstrap_state = visited_states[update_time + n_steps]
                estimated_return += (discount_factor ** n_steps) * value_estimates[bootstrap_state]

            # State whose value needs to be updated
            state_to_update = visited_states[update_time]

            # Update only if not terminal
            if state_to_update not in terminal_states:
                value_estimates[state_to_update] += alpha * (estimated_return - value_estimates[state_to_update])

        # If all updates are done at the episode end → exit loop
        if update_time == termination_time - 1:
            break

        # Transition to next state
        current_state = next_state

    # endregion Body

# endregion Functions

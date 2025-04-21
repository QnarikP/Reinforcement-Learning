import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class BlackjackMonteCarlo:
    """
    This class implements Monte Carlo simulations for a simplified Blackjack game.
    It includes methods to set policies for the player and dealer, simulate the game,
    and evaluate state values through different sampling methods.
    """

    def __init__(self):
        """
        Initialize the simulation by setting up actions and policies for both the player and the dealer.
        """
        # Define constant actions: hit (request a new card) and stick (stop requesting cards)
        self.ACTION_HIT = 0  # Constant to represent "hit" action
        self.ACTION_STICK = 1  # Constant to represent "stick" action

        # Store actions in a list for sampling purposes
        self.available_actions = [self.ACTION_HIT, self.ACTION_STICK]

        # Initialize player's policy array for sums from 0 to 21
        # We use indices 12 to 21 as valid values since a player draws until reaching at least 12.
        self.player_policy = np.zeros(22, dtype=np.int64)
        # For sums 12 to 19, player should hit (i.e. request additional cards)
        for player_sum in range(12, 20):
            self.player_policy[player_sum] = self.ACTION_HIT
        # For sums 20 and 21, player should stick (i.e. not take additional cards)
        self.player_policy[20] = self.ACTION_STICK
        self.player_policy[21] = self.ACTION_STICK

        # Initialize dealer's policy array
        # Dealer hits when the sum is between 12 and 16, and sticks when it is 17 or higher
        self.dealer_policy = np.zeros(22, dtype=np.int64)
        for dealer_sum in range(12, 17):
            self.dealer_policy[dealer_sum] = self.ACTION_HIT
        for dealer_sum in range(17, 22):
            self.dealer_policy[dealer_sum] = self.ACTION_STICK

    def target_policy_player(self, usable_ace_player, player_sum, dealer_card):
        """
        Target policy for the player, which returns the action based solely on the player's sum.
        A "usable ace" or the dealer card is not used in this simplified policy.

        :param usable_ace_player: Boolean flag indicating whether the player has a usable ace.
        :param player_sum: Total sum of the player's cards.
        :param dealer_card: The dealer's showing card (not used in this policy).
        :return: Chosen action based on the player's sum.
        """
        # Return action based on the pre-defined player policy for the given player sum.
        return self.player_policy[player_sum]

    def behavior_policy_player(self, usable_ace_player, player_sum, dealer_card):
        """
        Behavior policy for the player, which randomly chooses an action with equal probability.

        :param usable_ace_player: Boolean flag indicating whether the player has a usable ace.
        :param player_sum: Total sum of the player's cards.
        :param dealer_card: The dealer's showing card (not used in this simple behavior policy).
        :return: Randomly chosen action (either hit or stick).
        """
        # Simulate a coin flip where 50% probability is assigned to either hit or stick.
        if np.random.binomial(n=1, p=0.5) == 1:
            return self.ACTION_STICK
        return self.ACTION_HIT

    def get_card(self):
        """
        Draw a new card from the deck. Cards are represented by numbers.
        The card drawn is between 1 and 13 (inclusive) where:
            - Ace is represented as 1
            - Face cards (Jack, Queen, King) are represented as values greater than 10,
              but here are capped to 10 for game simplification.

        :return: Integer value representing the drawn card (capped at 10).
        """
        # Draw a random integer between 1 and 13 (inclusive)
        card = np.random.randint(low=1, high=14)
        # Return the card value, ensuring face cards are represented as 10
        return min(card, 10)

    def card_value(self, card_number):
        """
        Calculate the value of a card based on its number.
        An Ace (represented by 1) is given a value of 11, while all other cards have their own face value.

        :param card_number: The numeric identifier of the card.
        :return: The value of the card (11 for Ace, or the number for other cards).
        """
        # If the card is an Ace (1), treat it as 11; otherwise use its own value.
        return 11 if card_number == 1 else card_number

    def play(self, player_policy_function, initial_state=None, initial_action=None):
        """
        Simulate a single game of simplified Blackjack.

        :param player_policy_function: Policy function to decide the player's move.
        :param initial_state: Optional initial state given as a list
                              [usable_ace_player, player_sum, dealer_card1]; otherwise, state is randomly generated.
        :param initial_action: Optional initial action to force at the beginning of the game.
        :return: (state, reward, player_trajectory)
                 state is [usable_ace_player, player_sum, dealer_card1]
                 reward: -1 if player loses, 0 if draw, 1 if player wins.
                 player_trajectory: list of ((usable_ace_player, player_sum, dealer_card1), action) tuples.
        """
        # Initialize player's sum as 0
        player_total = 0
        # Initialize an empty list to store the trajectory (state-action pairs) of the player
        trajectory = []
        # Flag to check if the player has a usable ace (ace counted as 11)
        has_usable_ace = False

        # Determine initial state if not provided
        if initial_state is None:
            # Ensure the player's total is at least 12 by drawing cards repeatedly.
            while player_total < 12:
                drawn_card = self.get_card()  # draw a card
                # Add the card's value to the player's total sum.
                player_total += self.card_value(drawn_card)
                # If the player's total goes over 21, it must have been a double ace situation.
                if player_total > 21:
                    # In this scenario, the total must equal 22 (two aces counted as 11),
                    # so convert one Ace from 11 to 1.
                    assert player_total == 22
                    player_total -= 10
                else:
                    # Update has_usable_ace flag if the drawn card is an Ace.
                    has_usable_ace |= (drawn_card == 1)
            # Initialize dealer's cards: first card is shown, second is hidden.
            dealer_card1 = self.get_card()  # Dealer's face-up card.
            dealer_card2 = self.get_card()  # Dealer's hidden card.
        else:
            # Use the provided initial state.
            has_usable_ace, player_total, dealer_card1 = initial_state
            # Generate dealer's second card if state is specified.
            dealer_card2 = self.get_card()

        # Save the starting state as a list containing the player's ace status, total sum, and dealer's showing card.
        current_state = [has_usable_ace, player_total, dealer_card1]

        # Initialize dealer's total score by combining the values of the dealer's two cards.
        dealer_total = self.card_value(dealer_card1) + self.card_value(dealer_card2)
        # Check if the dealer has a usable ace
        dealer_has_usable_ace = (dealer_card1 == 1 or dealer_card2 == 1)

        # If dealer's total exceeds 21 (which can happen if both cards are aces),
        # adjust one ace to be counted as 1.
        if dealer_total > 21:
            assert dealer_total == 22  # This ensures that the only bust here is from two aces.
            dealer_total -= 10

        # Confirm that both totals are within valid game ranges.
        assert dealer_total <= 21
        assert player_total <= 21

        # --- Player's Turn ---
        while True:
            # Use the initial_action if provided for the first move.
            if initial_action is not None:
                action = initial_action
                initial_action = None  # Reset so subsequent actions are chosen by policy.
            else:
                # Decide action based on the provided player policy function.
                action = player_policy_function(has_usable_ace, player_total, dealer_card1)

            # Record the state and action for the importance sampling trajectory.
            trajectory.append(((has_usable_ace, player_total, dealer_card1), action))

            # If the player chooses to stick, end the player's turn.
            if action == self.ACTION_STICK:
                break

            # If the player chooses to hit, draw a new card.
            drawn_card = self.get_card()
            # Count the number of aces (using has_usable_ace as a flag) the player holds.
            ace_counter = int(has_usable_ace)
            # If the drawn card is an Ace, increment the counter.
            if drawn_card == 1:
                ace_counter += 1
            # Update the player's total with the value of the drawn card.
            player_total += self.card_value(drawn_card)
            # If the total exceeds 21, but the player has aces that can be revalued, adjust the total by treating an ace as 1.
            while player_total > 21 and ace_counter:
                player_total -= 10  # Adjust one ace from 11 to 1.
                ace_counter -= 1
            # If player still exceeds 21 after adjustments, the player busts.
            if player_total > 21:
                return current_state, -1, trajectory
            # Ensure player's total is now valid.
            assert player_total <= 21
            # Update the flag indicating if the player still has a usable ace (i.e., exactly one ace used as 11).
            has_usable_ace = (ace_counter == 1)

        # --- Dealer's Turn ---
        while True:
            # Get dealer's action from the dealer policy, which depends on the current sum.
            dealer_action = self.dealer_policy[dealer_total]
            # If the policy instructs the dealer to stick, exit the loop.
            if dealer_action == self.ACTION_STICK:
                break
            # Otherwise, the dealer takes a new card.
            new_card = self.get_card()
            # Count the dealer's usable ace flag as an integer.
            ace_counter_dealer = int(dealer_has_usable_ace)
            # If the new card is an Ace, increment the count.
            if new_card == 1:
                ace_counter_dealer += 1
            # Add the new card's value to the dealer's total.
            dealer_total += self.card_value(new_card)
            # If the dealer's total exceeds 21, adjust usable aces if possible.
            while dealer_total > 21 and ace_counter_dealer:
                dealer_total -= 10  # Adjust one ace from 11 to 1.
                ace_counter_dealer -= 1
            # If the dealer still busts, return a win for the player.
            if dealer_total > 21:
                return current_state, 1, trajectory
            # Update the dealer's usable ace flag for the next iteration.
            dealer_has_usable_ace = (ace_counter_dealer == 1)

        # Final comparison of player's and dealer's sums if neither bust.
        assert player_total <= 21 and dealer_total <= 21
        if player_total > dealer_total:
            return current_state, 1, trajectory  # Player wins.
        elif player_total == dealer_total:
            return current_state, 0, trajectory  # Draw.
        else:
            return current_state, -1, trajectory  # Dealer wins.

    def monte_carlo_on_policy(self, num_episodes):
        """
        Perform on-policy Monte Carlo simulation using the player's target policy.
        Returns the estimated state-value function for states when the player has a usable ace
        and when the player does not have a usable ace.

        :param num_episodes: Number of episodes to simulate.
        :return: Tuple of (values with usable ace, values with no usable ace),
                 each being a 10x10 array where rows index player sum (12 to 21) and
                 columns index dealer's showing card (1 to 10).
        """
        # Initialize arrays to accumulate rewards for states where player has a usable ace.
        values_usable_ace = np.zeros((10, 10))
        # Count for states with a usable ace; start with ones to prevent division by zero.
        count_usable_ace = np.ones((10, 10))
        # Initialize arrays for states without a usable ace.
        values_no_usable_ace = np.zeros((10, 10))
        # Count for states without a usable ace.
        count_no_usable_ace = np.ones((10, 10))

        # Loop over the number of episodes with a progress bar.
        for _ in tqdm(range(num_episodes)):
            # Play an episode with the target policy.
            state, reward, trajectory = self.play(self.target_policy_player)
            # Process each state-action pair in the player's trajectory.
            for (has_usable_ace, player_sum, dealer_card), _ in trajectory:
                # Map the player_sum (which is between 12 and 21) to an index from 0 to 9.
                player_index = player_sum - 12
                # Map the dealer's card (which is between 1 and 10) to index from 0 to 9.
                dealer_index = dealer_card - 1
                if has_usable_ace:
                    # Update count and cumulative reward for states with a usable ace.
                    count_usable_ace[player_index, dealer_index] += 1
                    values_usable_ace[player_index, dealer_index] += reward
                else:
                    # Update count and cumulative reward for states without a usable ace.
                    count_no_usable_ace[player_index, dealer_index] += 1
                    values_no_usable_ace[player_index, dealer_index] += reward

        # Compute the average reward (value) for each state.
        return values_usable_ace / count_usable_ace, values_no_usable_ace / count_no_usable_ace

    def monte_carlo_exploring_starts(self, num_episodes):
        """
        Monte Carlo with Exploring Starts (MCES) to estimate optimal state-action values.
        Returns the average state-action values over all episodes.

        :param num_episodes: Number of episodes to simulate.
        :return: A 4D numpy array of state-action values with dimensions
                 [player_sum_index, dealer_card_index, usable_ace (0 or 1), action (0 for hit, 1 for stick)].
        """
        # Initialize a state-action value array and the count of state-action visits.
        # Dimensions: 10 (player sums 12-21) x 10 (dealer's card 1-10) x 2 (usable ace flag) x 2 (actions)
        state_action_values = np.zeros((10, 10, 2, 2))
        state_action_counts = np.ones((10, 10, 2, 2))  # start counts at 1 to avoid division by zero

        def behavior_policy(usable_ace, player_sum, dealer_card):
            """
            Greedy behavior policy that selects the action with the highest current average return
            for the given state.

            :param usable_ace: Boolean flag indicating if the player has a usable ace.
            :param player_sum: Total sum of the player's cards.
            :param dealer_card: The dealer's showing card.
            :return: Chosen action (0 for hit, 1 for stick).
            """
            # Map boolean flag to integer index: 0 for no usable ace, 1 for usable ace.
            usable_ace_index = int(usable_ace)
            # Translate player's sum and dealer's card to indices.
            player_index = player_sum - 12
            dealer_index = dealer_card - 1
            # Compute the average returns for each action at the given state.
            average_returns = state_action_values[player_index, dealer_index, usable_ace_index, :] / \
                              state_action_counts[player_index, dealer_index, usable_ace_index, :]
            # Retrieve all actions that yield the maximum average return.
            optimal_actions = [action for action, value in enumerate(average_returns) if
                               value == np.max(average_returns)]
            # Randomly choose among the optimal actions in case of a tie.
            return np.random.choice(optimal_actions)

        # Monte Carlo Exploring Starts loop over episodes.
        for episode in tqdm(range(num_episodes)):
            # Initialize a random starting state [usable_ace, player_sum, dealer_card].
            initial_state = [bool(np.random.choice([0, 1])),
                             np.random.choice(range(12, 22)),
                             np.random.choice(range(1, 11))]
            # Choose a random initial action from available actions.
            initial_action = np.random.choice(self.available_actions)
            # For the first episode use the target policy, then use the greedy behavior policy.
            current_policy = behavior_policy if episode else self.target_policy_player
            # Play the game with the current starting state and initial action.
            _, reward, trajectory = self.play(current_policy, initial_state, initial_action)
            # Maintain a set to enforce first-visit updates per state-action pair.
            first_visit_set = set()
            for (has_usable_ace, player_sum, dealer_card), action in trajectory:
                # Adjust indices accordingly.
                player_index = player_sum - 12
                dealer_index = dealer_card - 1
                usable_ace_index = int(has_usable_ace)
                # Create a tuple representing the state-action pair.
                state_action_key = (usable_ace_index, player_index, dealer_index, action)
                # Only update if this is the first visit to this state-action pair in the trajectory.
                if state_action_key in first_visit_set:
                    continue
                first_visit_set.add(state_action_key)
                # Accumulate the reward and increment the visit count.
                state_action_values[player_index, dealer_index, usable_ace_index, action] += reward
                state_action_counts[player_index, dealer_index, usable_ace_index, action] += 1

        # Return the averaged state-action values.
        return state_action_values / state_action_counts

    def monte_carlo_off_policy(self, num_episodes):
        """
        Perform off-policy Monte Carlo sampling using importance sampling.
        The behavior policy is the random policy while the target policy is the player's target policy.
        Returns the ordinary and weighted importance sampling estimates.

        :param num_episodes: Number of episodes to simulate.
        :return: Tuple containing ordinary sampling and weighted sampling estimates.
        """
        # Fixed initial state for off-policy evaluation.
        initial_state = [True, 13, 2]

        importance_ratios = []  # List to store importance sampling ratios for each episode.
        rewards_list = []  # List to store rewards for each episode.

        # Loop over each episode.
        for _ in range(num_episodes):
            # Play an episode using the behavior policy.
            _, reward, trajectory = self.play(self.behavior_policy_player, initial_state=initial_state)
            # Initialize numerator and denominator for the importance sampling ratio calculation.
            ratio_numerator = 1.0
            ratio_denominator = 1.0
            # Iterate over each state-action pair in the trajectory.
            for (has_usable_ace, player_sum, dealer_card), action_taken in trajectory:
                # Check if the action taken matches the target policy's choice.
                if action_taken == self.target_policy_player(has_usable_ace, player_sum, dealer_card):
                    # If it matches, multiply the denominator by the probability under the random behavior policy (0.5).
                    ratio_denominator *= 0.5
                else:
                    # If there is a mismatch, the likelihood ratio is zero, so break out.
                    ratio_numerator = 0.0
                    break
            # Compute the importance sampling ratio (rho)
            rho = ratio_numerator / ratio_denominator
            importance_ratios.append(rho)
            rewards_list.append(reward)

        # Convert lists to numpy arrays for vectorized operations.
        importance_ratios = np.asarray(importance_ratios)
        rewards_list = np.asarray(rewards_list)
        # Multiply rewards by their corresponding importance sampling ratio.
        weighted_rewards = importance_ratios * rewards_list

        # Compute the cumulative sum for ordinary importance sampling.
        cumulative_weighted_rewards = np.add.accumulate(weighted_rewards)
        cumulative_rhos = np.add.accumulate(importance_ratios)

        # Compute ordinary importance sampling estimate at each episode.
        ordinary_estimate = cumulative_weighted_rewards / np.arange(1, num_episodes + 1)

        # Compute weighted importance sampling estimate, handling division by zero.
        with np.errstate(divide='ignore', invalid='ignore'):
            weighted_estimate = np.where(cumulative_rhos != 0, cumulative_weighted_rewards / cumulative_rhos, 0)

        return ordinary_estimate, weighted_estimate
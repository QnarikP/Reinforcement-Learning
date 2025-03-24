# Tic-Tac-Toe RL Agent

## Overview
This project features a Tic-Tac-Toe game with reinforcement learning (RL). It includes a game manager, an RL player learning via temporal-difference methods, and a human player using keyboard input.

## Features
- **Game Manager**: Oversees turns and determines the winner.
- **RL Player**: Improves decisions with reinforcement learning.
- **Human Player**: Plays via keyboard.
- **State Management**: Tracks board state, computes unique hashes, and evaluates outcomes.
- **Learning Strategy**: Uses an epsilon-greedy approach.
- **Policy Persistence**: Saves and loads policies with pickle.

## Key Classes
### Judge
- `play()`: Manages gameplay and decides the winner.
- `reset()`: Prepares for a new game.

### RL Player
- `act()`: Selects actions using epsilon-greedy.
- `update_state_value_estimates()`: Updates values with RL.
- `save_policy() / load_policy()`: Stores and retrieves policies.

### Human Player
- `act()`: Takes keyboard input for moves.

### State
- `calculate_hash_value()`: Computes a unique board hash.
- `is_game_ended()`: Checks game status and winner.
- `get_next_state()`: Generates next board state.
- `print_state()`: Displays the board.
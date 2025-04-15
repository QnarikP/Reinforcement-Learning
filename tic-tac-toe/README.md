# Tic‑Tac‑Toe RL Agent

## Overview

This project implements a reinforcement learning agent for Tic‑Tac‑Toe. The agent uses temporal‑difference learning (TD(0)) to estimate state values and improve its gameplay over repeated training episodes. This approach follows the ideas discussed in Chapter 6 (Temporal‑Difference Learning) of Sutton & Barto’s *Reinforcement Learning*.

## Project Structure

```
tic_tac_toe_rl/
└── src/
    ├── tic_tac_toe.py   # Main script that trains, competes, or lets a human play against the RL agent.
    ├── judge.py         # Organizes gameplay by alternating moves between players.
    ├── player.py        # Contains implementations for the RL player and the human player.
    └── state.py         # Defines the board state, including hash computation and status evaluation.
```

## How to Run

To run the project from your terminal, use one of these bash commands:

```bash
# To train the RL agents:
python src/tic_tac_toe.py --train

# To run a competition between two trained RL agents:
python src/tic_tac_toe.py --compete

# To play against the RL agent:
python src/tic_tac_toe.py --play
```
# Gridworld Dynamic Programming

## Overview

This project demonstrates various dynamic programming techniques—including iterative policy evaluation and in-place policy iteration—on a Gridworld environment. The methodology follows principles from Chapter 4 (Dynamic Programming) in Sutton & Barto’s *Reinforcement Learning*.

## Project Structure

```
gridworld_dp/
├── book_images/           # Illustrative figures referenced from the book.
│   ├── Example_4.1.PNG
│   └── Figure_4.1.PNG     
├── generated_images/
│   └── [Automatically saved output plots]
├── notebooks/
│   └── grid_world.ipynb   # Notebook demonstrating how to apply DP methods.
└── src/
    ├── __init__.py
    └── grid_world.py      # Defines the GridWorld class along with DP-based methods.
```

## How to Run

Start by launching the Jupyter Notebook or executing the Python module with:

```bash
# Start the interactive notebook:
jupyter notebook notebooks/grid_world.ipynb

# Or run directly from the command line:
python src/grid_world.py
```

## Example Output

Below is an example image showing the **final state-values** after applying iterative policy evaluation on a $4 \times 4$ grid:

<img src="generated_images/final_state_values.png" width="60%">

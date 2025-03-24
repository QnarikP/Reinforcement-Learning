# **Grid World - Reinforcement Learning (Project 3)**  

## **Overview**  
This project implements **Grid World**, a classic environment used in **Reinforcement Learning (RL)** to develop and test **Value Iteration** and **Policy Evaluation**. The agent navigates a **5x5 grid**, transitioning between states while aiming to maximize rewards. Special states introduce unique transitions and rewards, making it an excellent environment for understanding **Markov Decision Processes (MDPs)**.  

## **Key Features**  
- **Grid-based Environment**: A **5x5** grid world with deterministic transitions.  
- **Dynamic Programming (DP) Algorithms**: Implements **Value Iteration** for optimal policy computation.  
- **Customizable Actions & Rewards**: Supports different movement strategies and special state handling.  
- **Visualization**: Uses `matplotlib` to plot **state-value functions** and **optimal policies** with the **Blues colormap**.  
- **Efficient Implementation**: Optimized loops and **Bellman Equations** for convergence.  

## **How It Works**  
1. **State Representation**  
   - The agent moves in a **5x5 grid**, choosing actions **left, up, right, down**.  
   - Some states (A, B) teleport the agent to new locations with additional rewards.  

2. **Value Iteration**  
   - Iterates over all states, updating values using the **Bellman optimality equation**.  
   - Continues until the state-value function converges.  

3. **Policy Extraction**  
   - Derives the best actions at each state based on the **optimal value function**.  
   - Uses arrows (`←, ↑, →, ↓`) to represent the optimal policy.  

4. **Visualization**  
   - **State-Value Function**: Displays a heatmap of values using `matplotlib` with **cmap=Blues**.  
   - **Optimal Policy**: Overlays arrows to indicate the best action at each state.  

## **Learning Objectives**  
- Understand **Value Iteration** and **Bellman Equations** in RL.  
- Explore **state transitions**, **rewards**, and **policy extraction**.  
- Visualize **state-value functions** and **optimal policies** effectively.  
- Apply **Dynamic Programming (DP)** techniques to **MDPs**.

## Optimal Policy Visualization

After running the value iteration algorithm, we obtain the optimal policy for the Grid World environment:

![Optimal Policy](generated_images/figure_35__optimal_policy.png)
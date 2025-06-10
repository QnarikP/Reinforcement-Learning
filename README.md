# Reinforcement Learning Projects

This repository contains various Reinforcement Learning (RL) projects covering different topics and approaches in the field. Each project includes implementations, experiments, and insights related to RL algorithms and their applications.

## Projects Included

- `tic-tac-toe`  
- `ten-armed-testbed`  
- `gridworld-mdp`  
- `gridworld-dp`  
- `gambler-problem`  
- `blackjack`  
- `infinite-variance`
- `random walk`

Each folder contains a self-contained implementation of a classical RL problem or experiment.

## Installation

To set up the environment, ensure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

Dependencies may vary across projects, so check each project's README for specific setup instructions.

## Acknowledgements

The experiments and code in this repository were largely inspired by [this book](http://incompleteideas.net/) – my trusty guide through the labyrinth of Reinforcement Learning. Special thanks to my lecturer, [Mr. H.S. Abgaryan](https://www.linkedin.com/in/hovhannes-abgaryan/) (a.k.a. Mr. Anthony Stark), whose lectures were as enlightening as they were entertaining (and who somehow made debugging seem almost fun).

## Mathematical Foundations

At its core, Reinforcement Learning (RL) models problems as Markov Decision Processes (MDPs) and uses value functions to estimate the long‑term benefits of states and actions. In this section, we elaborate on the main formulas and ideas that underpin our current and future projects.

### 1. Markov Decision Process (MDP)

An MDP is defined by the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ where:

- **$\mathcal{S}$ (States):** Represents the complete set of situations or configurations in which an agent can find itself. For example, in our Grid World projects, $\mathcal{S}$ might be every cell in the grid.
  
- **$\mathcal{A}$ (Actions):** Denotes the set of all possible actions available to the agent in each state. In a game like Tic‑Tac‑Toe, actions correspond to the available moves (placing a symbol on an empty board cell).
  
- **$P(s' | s, a)$ (Transition Dynamics):** This is the probability of moving to state $s'$ after taking action $a$ in state $s$. It quantifies the uncertainty in the environment. In many classical problems, these probabilities are deterministic; in others, like the k‑armed bandit problem, they represent inherent randomness.
  
- **$R(s, a, s')$ (Reward Function):** Specifies the immediate reward received after transitioning from state $s$ to state $s'$ using action $a$. Rewards guide the learning process; for instance, a win may yield a reward of +1 while a loss gives -1.
  
- **$\gamma \in [0,1]$ (Discount Factor):** Determines the present value of future rewards. A value of $\gamma=0$ makes the agent myopic (only considering immediate rewards), whereas $\gamma$ close to 1 causes the agent to value long‑term rewards almost as highly as immediate ones.

Understanding the MDP framework is critical because it formalizes the interaction between the agent and the environment. By making the Markov property—where the future is independent of the past given the present state—an MDP allows us to derive recursive relationships for evaluating a policy.

### 2. The Value Function

The **state-value function** $v_\pi(s)$ under a policy $\pi$ represents the expected return when starting from state $s$ and following $\pi$ thereafter. Mathematically, it is defined as:

$$v_\pi(s) = \mathbb{E}_\pi\left[G_t \;|\; S_t = s\right] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;|\; S_t=s\right]$$

- **$G_t$** is the return following time step $t$ (a sum of discounted rewards).
- The expectation is taken over all possible trajectories starting at state $s$ under the policy $\pi$.

This function summarizes the future reward prospects of each state and is central to many RL algorithms.

### 3. The Bellman Equation

The Bellman equation provides a recursive decomposition of the value function. For any state $s$, the value $v_\pi(s)$ can be expressed in terms of the immediate reward and the value of successor states:

$$v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[ R(s, a, s') + \gamma\, v_\pi(s') \right]$$

**Explanation:**

- The inner summation computes the expected return for a given action $a$: the immediate reward $R(s, a, s')$ plus the discounted future value $\gamma\, v_\pi(s')$.
- The outer summation averages over the actions, weighted by the policy $\pi$.
- This equation underlies methods like iterative policy evaluation, where the value function is repeatedly updated until convergence.

### 4. The Bellman Optimality Equation

When seeking the best possible behavior, we define the **optimal state-value function** $v_*(s)$ as the maximum value achievable from any state:

$$v_*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[ R(s, a, s') + \gamma\, v_*(s') \right]$$

Similarly, for action-values (the $q$-function), the optimal equation is:

$$q_*(s, a) = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \max_{a'} q_*(s', a') \right]$$

**Key Points:**

- These equations form the basis for algorithms like Value Iteration and Q‑Learning.
- The optimal value function gives a benchmark against which any policy can be compared.
- They guarantee that by selecting actions that maximize the expected return at each step, the agent follows an optimal policy.

### 5. Temporal-Difference (TD) Learning

TD learning is a foundational approach that updates the value estimate using observed rewards and the estimated value of the next state, rather than waiting until the end of an episode:

$$V(S_t) \leftarrow V(S_t) + \alpha \Bigl[ R_{t+1} + \gamma\, V(S_{t+1}) - V(S_t) \Bigr]$$

- **$\alpha$** is the learning rate.
- **TD Error ($\delta_t$)** is given by:
  
  $$\delta_t = R_{t+1} + \gamma\, V(S_{t+1}) - V(S_t)$$
  
- This incremental update rule allows an agent to learn online and continuously refine its value function using every step’s experience.
- TD learning combines aspects of Monte Carlo methods (using sample returns) and dynamic programming (bootstrapping).

### 6. Monte Carlo Methods

Monte Carlo (MC) methods approximate the value function by averaging the returns of multiple sample episodes. For a state $s$, if $N(s)$ visits have been observed, the value can be estimated as:

$$V(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G^{(i)}$$

where $G^{(i)}$ is the total discounted return following the $i$-th visit to $s$.

**Additional Points:**

- MC methods do not require a model of the environment (i.e., they do not need $P$ and $R$).
- They are particularly useful for episodic tasks, where each episode terminates after a finite sequence of states.
- Variants include first‑visit MC and every‑visit MC, which differ in how returns are averaged.

### 7. Policy Improvement and Iteration

Once the value function is estimated under a policy $\pi$, the policy can be improved by choosing actions that maximize the expected return. The policy improvement step can be expressed as:

$$\pi'(s) = \arg \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[ R(s, a, s') + \gamma\, v_\pi(s') \right]$$

By repeatedly evaluating the current policy (policy evaluation) and then improving it (policy improvement), we obtain the policy iteration algorithm, which converges to the optimal policy.

### 8. Off-Policy Evaluation and Importance Sampling

When the data is generated by a different policy $b$ (the behavior policy) than the one we wish to evaluate $\pi$ (the target policy), importance sampling is used to correct the discrepancy. The importance sampling ratio for an episode is computed as:

$$\rho = \prod_{t=0}^{T-1} \frac{\pi(A_t \mid S_t)}{b(A_t \mid S_t)}$$

This ratio re-weights the returns so that the estimator remains unbiased. Two common methods are:

- **Ordinary Importance Sampling:** Uses the ratio directly but can suffer from high variance.
  
- **Weighted Importance Sampling:** Normalizes the ratios to reduce variance, at the expense of introducing bias in finite samples.

**Why It Matters:**

- Off-policy methods are crucial when it is impractical or unsafe to deploy the target policy directly.
- The ability to learn about one policy while following another is a powerful feature in many real-world applications, such as evaluating new strategies without affecting live systems.

# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description
FrozenLake-v1 is a grid-based reinforcement learning environment provided by Gymnasium. The environment contains frozen tiles, holes, a starting state, and a goal state. The agent starts from the starting position and must learn to reach the goal while avoiding the holes. At each state, the agent can choose one of four actions: Left, Down, Right, or Up. The agent receives a reward when it successfully reaches the goal. The environment is used to train the agent through repeated episodes and learn the best action for each state using Monte Carlo Control.




## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm
1.Initialize the FrozenLake environment and obtain the number of states and actions.

2.Initialize the Q-table with zeros.

3.Set the hyperparameters: number of episodes, learning rate α, discount factor γ, and epsilon values.

4.Select actions using the epsilon-greedy policy.

5.Generate a complete episode until the agent reaches the goal, falls into a hole, or reaches the maximum number of steps.

6.Calculate the return Gt by traversing the episode backward.

7.Update the Q-value using: Q(s,a) ← Q(s,a) + α[Gt − Q(s,a)]

8.Gradually reduce epsilon to shift from exploration to exploitation.

9.Extract the greedy policy using the maximum Q-value for each state.

10.Display the Q-table, state-value function, learned policy, average reward, and learning curve.


## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
# Write your code here
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Create Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env.observation_space.n
n_actions = env.action_space.n

print("Number of states:", n_states)
print("Number of actions:", n_actions)


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 20000
gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100
# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))
episode_rewards = []


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

# Write your code here
def epsilon_greedy_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])
# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------

def generate_episode(epsilon):
    """
    Generates one episode using the current epsilon-greedy policy.
    Returns a list of (state, action, reward).
    """

    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):
        action = epsilon_greedy_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

        if terminated or truncated:
            break

    return episode
# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

# Write your code here
epsilon = epsilon_start

for episode_num in range(num_episodes):

    episode = generate_episode(epsilon)

    G = 0
    visited = set()
    total_reward = 0

    for state, action, reward in reversed(episode):

        G = gamma * G + reward
        total_reward += reward

        if (state, action) not in visited:
            visited.add((state, action))

            Q[state, action] += alpha * (G - Q[state, action])

    episode_rewards.append(total_reward)

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    if (episode_num + 1) % 2000 == 0:
        print(
            f"Episode {episode_num + 1}, "
            f"Epsilon: {epsilon:.4f}, "
            f"Average Reward: {np.mean(episode_rewards[-1000:]):.3f}"
        )

# -------------------------------------------------
# Display Results
# -------------------------------------------------

def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)
    print("Name:          ")
    print("Register Number:      ")
    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(optimal_policy)

success_rate = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", success_rate)

# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500
moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Monte Carlo Control Learning Curve")
plt.grid(True)
plt.show()

env.close()



```

---

## Output


<img width="645" height="730" alt="image" src="https://github.com/user-attachments/assets/591bde7a-5f2d-463d-b1c5-8bb654f3f02f" />
 

<img width="978" height="597" alt="image" src="https://github.com/user-attachments/assets/bbbfe82d-76e6-4923-a6c5-2e93fb2e9e75" />


---

## Result
```text

The SARSA control algorithm was successfully implemented using the Gymnasium `FrozenLake-v1` environment. The agent learned the action-value function through repeated interaction with the environment and obtained a learned policy for selecting actions that help it reach the goal while avoiding holes.




```
---

## Inference
```text

The experiment demonstrates that SARSA can learn an action-value function through trial-and-error interaction with the environment. The epsilon-greedy policy provides a balance between exploration and exploitation. Since SARSA uses the action actually selected in the next state for updating the Q-value, it learns according to the policy being followed by the agent. After sufficient training, the agent learns a suitable policy for navigating the FrozenLake environment.




```





---


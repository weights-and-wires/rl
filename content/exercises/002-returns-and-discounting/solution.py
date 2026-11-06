"""Reference solution for the returns and discounting exercise."""

import numpy as np

import rlenvs

# A 3 by 3 room with a pillar in the middle. Every step costs 1, and the
# episode ends on reaching G in the far corner.
ROOM = [
    "S..",
    ".#.",
    "..G",
]


def run_episode(mdp, pi, rng):
    s = mdp.reset(seed=int(rng.integers(2**31)))
    states, actions, rewards = [], [], []
    while True:
        a = int(rng.choice(mdp.n_actions, p=pi[s]))
        s2, r, terminated, truncated = mdp.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s2
        if terminated or truncated:
            break
    return np.array(states), np.array(actions), np.array(rewards, dtype=float), bool(terminated)


def discount_weights(gamma, n):
    return gamma ** np.arange(n, dtype=float)


def discounted_returns(rewards, gamma):
    rewards = np.asarray(rewards, dtype=float)
    returns = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def mc_value_estimate(mdp, pi, n_episodes, seed):
    rng = np.random.default_rng(seed)
    g0 = np.empty(n_episodes)
    for i in range(n_episodes):
        _, _, rewards, _ = run_episode(mdp, pi, rng)
        g0[i] = discounted_returns(rewards, mdp.gamma)[0]
    return float(g0.mean()), float(g0.std(ddof=1) / np.sqrt(n_episodes))


if __name__ == "__main__":
    print("weights on rewards 0 to 4 steps ahead:")
    for gamma in (0.5, 0.9):
        print(f"  gamma={gamma}:", np.round(discount_weights(gamma, 5), 3))

    mdp = rlenvs.gridworld(ROOM, gamma=0.9)
    pi = rlenvs.uniform_policy(mdp)
    print("returns for rewards (1, 0, 2):", discounted_returns([1.0, 0.0, 2.0], 0.9))

    exact = rlenvs.policy_evaluation(mdp, pi)[mdp.names["start"]]
    for n in (100, 400, 1600):
        estimate, se = mc_value_estimate(mdp, pi, n, seed=1)
        print(f"  {n:5d} episodes: V(start) about {estimate:.3f} +- {se:.3f}   (exact {exact:.3f})")

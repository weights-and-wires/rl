"""Reference solution for the agent-environment loop exercise."""

import numpy as np

import rlenvs

# A 3 by 3 room with a pillar in the middle. Every step costs 1, and the
# episode ends on reaching G in the far corner.
ROOM = [
    "S..",
    ".#.",
    "..G",
]


def uniform_random_policy(n_states, n_actions):
    return np.full((n_states, n_actions), 1.0 / n_actions)


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


def visit_counts(states, n_states):
    return np.bincount(np.asarray(states, dtype=int), minlength=n_states)


def mean_episode_length(mdp, pi, n_episodes, seed):
    rng = np.random.default_rng(seed)
    lengths = [len(run_episode(mdp, pi, rng)[1]) for _ in range(n_episodes)]
    return float(np.mean(lengths))


if __name__ == "__main__":
    mdp = rlenvs.gridworld(ROOM, gamma=0.9)
    pi = uniform_random_policy(mdp.n_states, mdp.n_actions)
    rng = np.random.default_rng(0)
    states, actions, rewards, terminated = run_episode(mdp, pi, rng)
    print(f"one episode of the random agent: {len(rewards)} steps, terminated={terminated}")
    print("  first five steps (state, action, reward):")
    for t in range(min(5, len(rewards))):
        print(f"    t={t}: s={states[t]}, a={actions[t]}, r={rewards[t]:.0f}")
    print("  visits per state:", visit_counts(states, mdp.n_states))
    print(f"  mean length over 500 episodes: {mean_episode_length(mdp, pi, 500, seed=1):.1f}")

"""Reference solution for the Markov decision process exercise."""

import numpy as np

import rlenvs


def hallway_mdp(gamma=0.9):
    """Four cells in a row. Cell 3 is the exit and ends the episode.

    Action 0 is left and action 1 is right. A move changes the cell by one, and
    moving left from cell 0 bumps the wall and stays in cell 0. Every move from
    cells 0, 1, and 2 pays -1. The episode starts in cell 0.
    """
    P = np.zeros((4, 2, 4))
    R = np.zeros((4, 2))
    for s in range(3):
        P[s, 0, max(s - 1, 0)] = 1.0
        P[s, 1, s + 1] = 1.0
        R[s, :] = -1.0
    P[3, :, 3] = 1.0
    return rlenvs.TabularMDP(
        P=P, R=R, gamma=gamma,
        mu0=np.array([1.0, 0.0, 0.0, 0.0]),
        terminal=np.array([False, False, False, True]),
    )


def sample_episode(mdp, pi, rng, max_steps=1000):
    s = int(rng.choice(mdp.n_states, p=mdp.mu0))
    states, actions, rewards = [s], [], []
    while not mdp.terminal[s] and len(actions) < max_steps:
        a = int(rng.choice(mdp.n_actions, p=pi[s]))
        s, r = mdp.sample(s, a, rng)
        states.append(s)
        actions.append(a)
        rewards.append(r)
    return states, actions, rewards


def policy_matrices(mdp, pi):
    live = ~mdp.terminal
    P_pi = np.einsum("sa,sat->st", pi, mdp.P) * live[:, None]
    r_pi = (pi * mdp.R).sum(axis=1) * live
    return P_pi, r_pi


if __name__ == "__main__":
    mdp = hallway_mdp()
    uniform = rlenvs.uniform_policy(mdp)
    always_right = np.tile([0.0, 1.0], (4, 1))

    states, actions, rewards = sample_episode(mdp, always_right, np.random.default_rng(0))
    print("Always right: states", states, "actions", actions, "rewards", rewards)

    rng = np.random.default_rng(0)
    for k in range(3):
        states, actions, rewards = sample_episode(mdp, uniform, rng)
        G = sum(mdp.gamma**t * r for t, r in enumerate(rewards))
        print(f"Uniform episode {k}: {len(actions)} steps, return {G:.4f}, cells {states}")

    P_pi, r_pi = policy_matrices(mdp, uniform)
    print("P_pi under the uniform policy:")
    print(P_pi)
    print("r_pi =", r_pi)

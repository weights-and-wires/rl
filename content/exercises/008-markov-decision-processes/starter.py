"""Markov decision processes: sampling trajectories and averaging out a policy.

Fill in the functions in order. An `rlenvs.TabularMDP` carries `P` of shape
(S, A, S), `R` of shape (S, A), `gamma`, `mu0` of shape (S,), and `terminal` of
shape (S,). A policy `pi` has shape (S, A), and each row sums to one.

`hallway_mdp` is the running example for Part 2. It is complete; don't change it.
"""

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
    """Follow `pi` through the model from a start state until the episode ends.

    Draw the first state from `mdp.mu0`. Then, while the current state is not
    terminal and fewer than `max_steps` steps have been taken, draw an action
    from `pi[s]` and call `mdp.sample(s, a, rng)`, which returns
    `(next_state, reward)`.

    Args:
        mdp: an `rlenvs.TabularMDP`.
        pi: the policy, shape (S, A).
        rng: a `np.random.Generator`, used for every random draw.
        max_steps: stop after this many steps even if the episode has not ended.

    Returns:
        A tuple `(states, actions, rewards)` of lists. `states` holds s_0 to
        s_T, so it has one more entry than `actions` and `rewards`.
    """
    raise NotImplementedError("sample_episode")


def policy_matrices(mdp, pi):
    """The Markov chain and expected rewards that a policy induces.

    Args:
        mdp: an `rlenvs.TabularMDP`.
        pi: the policy, shape (S, A).

    Returns:
        A tuple `(P_pi, r_pi)`. `P_pi[s, s2]` is sum_a pi(a | s) P(s2 | s, a),
        shape (S, S). `r_pi[s]` is sum_a pi(a | s) R(s, a), shape (S,). Rows of
        terminal states are all zero in both, because nothing happens after
        an episode ends.
    """
    raise NotImplementedError("policy_matrices")


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

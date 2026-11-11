"""Cooperative multi-agent RL: VDN, a QMIX mixer, and the climbing game.

Starter. Every function works on plain numpy arrays, except
`fit_qmix`, which trains a tiny monotonic mixer with torch.
"""

import itertools

import numpy as np
import torch

torch.set_num_threads(1)

# Claus and Boutilier's climbing game. Rows are agent 1's actions, columns are
# agent 2's. Both agents receive the same payoff. (0, 0) pays 11, the best
# joint action, but a miss next to it costs -30. (2, 2) pays only 5 and
# never punishes a partner who strays.
CLIMBING = np.array(
    [
        [11.0, -30.0, 0.0],
        [-30.0, 7.0, 6.0],
        [0.0, 0.0, 5.0],
    ]
)

# Two 2 by 2 team payoff tables. AND pays 1 only when both agents play action 1,
# and it is monotone in each agent's action. XOR pays 1 when the agents match,
# and no monotone mixer represents it.
AND = np.array([[0.0, 0.0], [0.0, 1.0]])
XOR = np.array([[1.0, 0.0], [0.0, 1.0]])


def greedy_actions(qs):
    """Each agent's own greedy action. `qs` has shape (n_agents, n_actions); returns an int array (n_agents,)."""
    raise NotImplementedError("write this function")


def vdn_q_tot(qs, actions):
    """VDN's team value: the sum of each agent's utility for its own action.

    `qs` has shape (n_agents, n_actions) and `actions` has shape (n_agents,).
    Returns a float.
    """
    raise NotImplementedError("write this function")


def best_joint_action(qs):
    """Search every joint action and return the one with the largest VDN Q_tot, as a tuple.

    Ties go to the first joint action in lexicographic order.
    """
    raise NotImplementedError("write this function")


def qmix_mixer(qs, state, hyper):
    """QMIX's monotonic mixing network.

    `qs` has shape (B, N) and `state` has shape (B, S). `hyper` is a dict of
    linear hypernetworks, each a pair of a matrix `*_A` and a bias `*_c`:

        w1: (S, N*H), (N*H,)    first-layer weights, reshaped to (B, N, H), then abs
        b1: (S, H),   (H,)      first-layer bias
        w2: (S, H),   (H,)      second-layer weights, then abs
        b2: (S, 1),   (1,)      final bias, no abs

    The hidden layer uses ELU. Returns Q_tot with shape (B,).
    """
    raise NotImplementedError("write this function")


def vdn_fit(table):
    """Least-squares additive fit of a 2-agent payoff table.

    Returns (q1, q2, mse) with q1[i] + q2[j] the closest sum to table[i, j],
    where mse is the mean squared error over all cells. q1 and q2 split the
    grand mean evenly.
    """
    raise NotImplementedError("write this function")


def fit_qmix(table, hidden=8, steps=1500, lr=0.05, seed=0):
    """Fit a one-state QMIX to a 2-agent payoff table by Adam on the mean squared error.

    The agent utilities are free tables `q1`, `q2`. The mixer has hidden width
    `hidden`, abs on both weight layers, and ELU. Returns the fitted table and the final mse.
    """
    raise NotImplementedError("write this function")


def expected_individual_values(table, partner):
    """Mean payoff of each of agent 1's actions when agent 2 plays the distribution `partner`.

    This is what an independent learner's Q table estimates. Returns shape (n1,).
    """
    raise NotImplementedError("write this function")


def independent_learners(table, steps=3000, alpha=0.05, epsilon=0.4, seed=0):
    """Two independent epsilon-greedy learners that each see only the team reward.

    Each keeps Q_i[a_i] and updates it by Q_i[a_i] += alpha * (r - Q_i[a_i]).
    Q starts at zero. Draw each agent's explore coin, then its random action, in
    agent order. Returns (Q1, Q2).
    """
    raise NotImplementedError("write this function")


def joint_learner(table, steps=3000, alpha=0.05, epsilon=0.4, seed=0):
    """One epsilon-greedy learner over joint actions. Returns Q with the table's shape."""
    raise NotImplementedError("write this function")


if __name__ == "__main__":
    qs = np.array([[1.0, 3.0, 0.0], [2.0, -1.0, 4.0]])
    joint = best_joint_action(qs)
    print("greedy per agent:", greedy_actions(qs), "joint argmax:", joint, "Q_tot:", vdn_q_tot(qs, joint))
    for name, table in (("AND", AND), ("XOR", XOR)):
        _, _, vdn_mse = vdn_fit(table)
        _, qmix_mse = fit_qmix(table)
        print(f"{name}: VDN mse {vdn_mse:.4f}, QMIX mse {qmix_mse:.4f}")
    for seed in range(3):
        Q1, Q2 = independent_learners(CLIMBING, seed=seed)
        J = joint_learner(CLIMBING, seed=seed)
        print(f"seed {seed}: independent plays {(int(Q1.argmax()), int(Q2.argmax()))}, "
              f"joint plays {np.unravel_index(J.argmax(), J.shape)}")

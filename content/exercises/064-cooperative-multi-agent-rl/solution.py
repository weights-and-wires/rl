"""Cooperative multi-agent RL: VDN, a QMIX mixer, and the climbing game.

Reference solution. Every function works on plain numpy arrays, except
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
    return np.argmax(qs, axis=1)


def vdn_q_tot(qs, actions):
    """VDN's team value: the sum of each agent's utility for its own action.

    `qs` has shape (n_agents, n_actions) and `actions` has shape (n_agents,).
    Returns a float.
    """
    qs = np.asarray(qs, dtype=float)
    actions = np.asarray(actions)
    return float(qs[np.arange(len(qs)), actions].sum())


def best_joint_action(qs):
    """Search every joint action and return the one with the largest VDN Q_tot, as a tuple.

    Ties go to the first joint action in lexicographic order.
    """
    qs = np.asarray(qs, dtype=float)
    n_agents, n_actions = qs.shape
    best, best_value = None, -np.inf
    for joint in itertools.product(range(n_actions), repeat=n_agents):
        value = vdn_q_tot(qs, joint)
        if value > best_value:
            best, best_value = joint, value
    return best


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
    qs = np.asarray(qs, dtype=float)
    state = np.asarray(state, dtype=float)
    batch, n_agents = qs.shape
    w1 = np.abs(state @ hyper["w1_A"] + hyper["w1_c"]).reshape(batch, n_agents, -1)
    b1 = state @ hyper["b1_A"] + hyper["b1_c"]
    w2 = np.abs(state @ hyper["w2_A"] + hyper["w2_c"])
    b2 = (state @ hyper["b2_A"] + hyper["b2_c"])[:, 0]
    pre = np.einsum("bn,bnh->bh", qs, w1) + b1
    hidden = np.where(pre > 0, pre, np.exp(np.minimum(pre, 0.0)) - 1.0)
    return (hidden * w2).sum(axis=1) + b2


def vdn_fit(table):
    """Least-squares additive fit of a 2-agent payoff table.

    Returns (q1, q2, mse) with q1[i] + q2[j] the closest sum to table[i, j],
    where mse is the mean squared error over all cells. q1 and q2 split the
    grand mean evenly.
    """
    table = np.asarray(table, dtype=float)
    grand = table.mean()
    q1 = table.mean(axis=1) - grand / 2.0
    q2 = table.mean(axis=0) - grand / 2.0
    fitted = q1[:, None] + q2[None, :]
    return q1, q2, float(((table - fitted) ** 2).mean())


def fit_qmix(table, hidden=8, steps=1500, lr=0.05, seed=0):
    """Fit a one-state QMIX to a 2-agent payoff table by Adam on the mean squared error.

    The agent utilities are free tables `q1`, `q2`. The mixer has hidden width
    `hidden`, abs on both weight layers, and ELU. Returns the fitted table and the final mse.
    """
    table_t = torch.as_tensor(np.asarray(table, dtype=np.float64))
    n1, n2 = table_t.shape
    g = torch.Generator().manual_seed(seed)
    q1 = (0.5 * torch.randn(n1, generator=g, dtype=torch.float64)).requires_grad_()
    q2 = (0.5 * torch.randn(n2, generator=g, dtype=torch.float64)).requires_grad_()
    w1 = torch.randn(2, hidden, generator=g, dtype=torch.float64).requires_grad_()
    b1 = torch.zeros(hidden, dtype=torch.float64, requires_grad=True)
    w2 = torch.randn(hidden, generator=g, dtype=torch.float64).requires_grad_()
    b2 = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([q1, q2, w1, b1, w2, b2], lr=lr)

    def table_of():
        pairs = torch.stack(
            [q1[:, None].expand(n1, n2), q2[None, :].expand(n1, n2)], dim=-1
        )
        h = torch.nn.functional.elu(pairs @ w1.abs() + b1)
        return h @ w2.abs() + b2

    for _ in range(steps):
        opt.zero_grad()
        loss = ((table_of() - table_t) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        fitted = table_of()
        return fitted.numpy(), float(((fitted - table_t) ** 2).mean())


def expected_individual_values(table, partner):
    """Mean payoff of each of agent 1's actions when agent 2 plays the distribution `partner`.

    This is what an independent learner's Q table estimates. Returns shape (n1,).
    """
    return np.asarray(table, dtype=float) @ np.asarray(partner, dtype=float)


def independent_learners(table, steps=3000, alpha=0.05, epsilon=0.4, seed=0):
    """Two independent epsilon-greedy learners that each see only the team reward.

    Each keeps Q_i[a_i] and updates it by Q_i[a_i] += alpha * (r - Q_i[a_i]).
    Q starts at zero. Draw each agent's explore coin, then its random action, in
    agent order. Returns (Q1, Q2).
    """
    rng = np.random.default_rng(seed)
    table = np.asarray(table, dtype=float)
    n1, n2 = table.shape
    qs = [np.zeros(n1), np.zeros(n2)]
    for _ in range(steps):
        acts = []
        for q in qs:
            if rng.random() < epsilon:
                acts.append(int(rng.integers(len(q))))
            else:
                acts.append(int(np.argmax(q)))
        r = table[acts[0], acts[1]]
        for q, a in zip(qs, acts):
            q[a] += alpha * (r - q[a])
    return qs[0], qs[1]


def joint_learner(table, steps=3000, alpha=0.05, epsilon=0.4, seed=0):
    """One epsilon-greedy learner over joint actions. Returns Q with the table's shape."""
    rng = np.random.default_rng(seed)
    table = np.asarray(table, dtype=float)
    Q = np.zeros_like(table)
    for _ in range(steps):
        if rng.random() < epsilon:
            a = (int(rng.integers(table.shape[0])), int(rng.integers(table.shape[1])))
        else:
            a = np.unravel_index(np.argmax(Q), Q.shape)
        Q[a] += alpha * (table[a] - Q[a])
    return Q


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

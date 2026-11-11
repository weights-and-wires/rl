"""Reference solution for the preferences-over-trajectories exercise."""

import numpy as np

import rlenvs

# ---------------------------------------------------------------------------
# The world and its helpers, identical in starter.py. You do not edit these.
# ---------------------------------------------------------------------------

LAYOUT = ["S...", ".#..", "..H.", "...G"]
GAMMA = 0.9


def make_world() -> rlenvs.TabularMDP:
    """A 4x4 grid with a wall, a hole `H`, and a goal `G`, as a task that never ends.

    Both `G` and `H` keep the agent where it is, so no state is terminal. With
    no terminal states, adding a constant to every reward shifts every Q value
    by the same number and cannot change the optimal policy. The reward is set
    later by `reward_mdp`. Actions: 0 up, 1 right, 2 down, 3 left.
    """
    g = rlenvs.gridworld(LAYOUT, gamma=GAMMA)
    P = g.P.copy()
    cells = g.names["cells"]
    goal = [k for k, (i, j) in enumerate(cells) if LAYOUT[i][j] == "G"][0]
    hole = [k for k, (i, j) in enumerate(cells) if LAYOUT[i][j] == "H"][0]
    for s in (goal, hole):
        P[s] = 0.0
        P[s, :, s] = 1.0
    S = g.n_states
    return rlenvs.TabularMDP(
        P=P,
        R=np.zeros((S, 4)),
        gamma=GAMMA,
        mu0=g.mu0,
        terminal=np.zeros(S, dtype=bool),
        names={"cells": cells, "goal": goal, "hole": hole, "start": g.names["start"]},
    )


def true_reward(mdp: rlenvs.TabularMDP) -> np.ndarray:
    """The reward of landing in each state, shape (S,): -0.15 per step of distance to the goal, +1 at G, -1 at H."""
    cells = mdp.names["cells"]
    gi, gj = cells[mdp.names["goal"]]
    r = np.array([-0.15 * (abs(i - gi) + abs(j - gj)) for (i, j) in cells])
    r[mdp.names["goal"]] = 1.0
    r[mdp.names["hole"]] = -1.0
    return r


def reward_mdp(mdp: rlenvs.TabularMDP, r: np.ndarray) -> rlenvs.TabularMDP:
    """A copy of `mdp` that pays `r[s2]` for landing in state `s2`."""
    R3 = np.broadcast_to(np.asarray(r, dtype=float)[None, None, :], mdp.P.shape).copy()
    return rlenvs.TabularMDP(
        P=mdp.P, R=(mdp.P * R3).sum(axis=2), gamma=mdp.gamma, mu0=mdp.mu0,
        terminal=mdp.terminal, R3=R3,
    )


def sample_segments(mdp: rlenvs.TabularMDP, n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """`n` random-walk segments, shape (n, length). Entry [i, t] is the state entered at step t."""
    S, A = mdp.n_states, mdp.n_actions
    segs = np.zeros((n, length), dtype=int)
    for i in range(n):
        s = int(rng.integers(S))
        for t in range(length):
            s = int(rng.choice(S, p=mdp.P[s, int(rng.integers(A))]))
            segs[i, t] = s
    return segs


def policy_regret(mdp: rlenvs.TabularMDP, r_true: np.ndarray, r_hat: np.ndarray) -> float:
    """How much value the greedy policy for `r_hat` loses under `r_true`, worst state."""
    true_mdp = reward_mdp(mdp, r_true)
    learned_mdp = reward_mdp(mdp, r_hat)
    q_hat = rlenvs.q_from_v(learned_mdp, rlenvs.value_iteration(learned_mdp))
    v_pi = rlenvs.policy_evaluation(true_mdp, rlenvs.greedy(q_hat))
    v_star = rlenvs.value_iteration(true_mdp)
    return float(np.max(v_star - v_pi))


# ---------------------------------------------------------------------------
# What you write.
# ---------------------------------------------------------------------------


def segment_return(reward_fn, segment) -> float:
    """The sum of the per-step rewards along a segment (a sum, not a mean)."""
    return float(np.sum(reward_fn(np.asarray(segment))))


def preference_prob(return_a: float, return_b: float) -> float:
    """P(segment A is preferred over B) = sigmoid(return_a - return_b), safe for gaps of +-800."""
    gap = float(return_a) - float(return_b)
    return float(0.5 * (1.0 + np.tanh(0.5 * gap)))


def bt_segment_loss(w, X, segs_a, segs_b, labels):
    """Mean Bradley-Terry loss over segment pairs, and its gradient with respect to `w`."""
    w = np.asarray(w, dtype=float)
    labels = np.asarray(labels, dtype=float)
    d = X[segs_a].sum(axis=1) - X[segs_b].sum(axis=1)      # (N, d)
    z = d @ w                                                  # (N,) return gaps
    loss = np.mean(np.logaddexp(0.0, z) - labels * z)
    p = 0.5 * (1.0 + np.tanh(0.5 * z))
    grad = d.T @ (p - labels) / len(labels)
    return float(loss), grad


def simulate_preference(true_r, seg_a, seg_b, beta, rng) -> int:
    """One noisy label from a Boltzmann labeler: 1 if A is preferred, else 0. Draws one `rng.random()`."""
    gap = np.sum(np.asarray(true_r)[np.asarray(seg_a)]) - np.sum(np.asarray(true_r)[np.asarray(seg_b)])
    return int(rng.random() < preference_prob(beta * gap, 0.0))


def fit_reward(X, segs_a, segs_b, labels, l2=0.01, steps=400, lr=0.5):
    """Gradient descent on the Bradley-Terry loss plus (l2 / 2) * |w|^2, from w = 0."""
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        _, grad = bt_segment_loss(w, X, segs_a, segs_b, labels)
        w = w - lr * (grad + l2 * w)
    return w


def disagreement_query(ws, X, segs_a, segs_b) -> int:
    """Index of the candidate pair whose predicted P(A preferred) has the largest variance across members."""
    d = X[segs_a].sum(axis=1) - X[segs_b].sum(axis=1)      # (N, d)
    probs = 0.5 * (1.0 + np.tanh(0.5 * (np.asarray(ws) @ d.T)))  # (K, N)
    return int(np.argmax(probs.var(axis=0)))


if __name__ == "__main__":
    mdp = make_world()
    r_true = true_reward(mdp)
    X = np.eye(mdp.n_states)
    rng = np.random.default_rng(0)
    segs_a = sample_segments(mdp, 300, 6, rng)
    segs_b = sample_segments(mdp, 300, 6, rng)
    labels = np.array([simulate_preference(r_true, a, b, 3.0, rng) for a, b in zip(segs_a, segs_b)])
    print("fraction of labels that prefer A:", round(float(labels.mean()), 3))

    w = fit_reward(X, segs_a, segs_b, labels)
    print("correlation of learned and true reward:", round(float(np.corrcoef(w, r_true)[0, 1]), 3))
    print("policy regret under the true reward:", round(policy_regret(mdp, r_true, w), 4))
    print("same, after adding 5 per step to the learned reward:", round(policy_regret(mdp, r_true, w + 5.0), 4))

    flipped = fit_reward(X, segs_a, segs_b, 1 - labels)
    print("policy regret when every label is flipped:", round(policy_regret(mdp, r_true, flipped), 2))

    members = np.stack([
        fit_reward(X, segs_a[idx], segs_b[idx], labels[idx])
        for idx in (rng.integers(0, 300, 300) for _ in range(5))
    ])
    cand_a = sample_segments(mdp, 50, 6, rng)
    cand_b = sample_segments(mdp, 50, 6, rng)
    print("ensemble would ask about candidate pair", disagreement_query(members, X, cand_a, cand_b))

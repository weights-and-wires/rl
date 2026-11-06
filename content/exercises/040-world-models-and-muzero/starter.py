"""World models and MuZero: returns in imagination, value equivalence, and search.

Fill in the functions below, in order. You compute Dreamer's lambda-returns
over an imagined trajectory, check when a wrong model is still good enough for
planning (value equivalence), and write MuZero's pUCT search over a model.

A deterministic model here is a function step(state, action) that returns a
tuple. In `mcts` it returns (next_state, reward, done). A prediction function
predict(state) returns (prior, value): a probability vector over actions and
a scalar value estimate.
"""

import numpy as np

import rlenvs


def lambda_returns(rewards, values, continues, gamma, lam):
    """Dreamer's lambda-returns by backward recursion.

    G_t = r_t + gamma * c_t * ((1 - lam) * v_{t+1} + lam * G_{t+1}), with G_H = v_H.

    Args:
        rewards: r_t for t = 0..H-1, shape (H,) or (H, ...).
        values: v(s_t) for t = 0..H, shape (H + 1,) or (H + 1, ...).
        continues: c_t, 1.0 if s_{t+1} is not terminal and 0.0 if it is,
            shape (H,) or (H, ...).
        gamma: the discount.
        lam: the lambda in [0, 1].

    Returns:
        G_t for t = 0..H-1, the same shape as rewards.
    """
    raise NotImplementedError("lambda_returns")


def bellman_backup(P, R, gamma, pi, v):
    """Apply the Bellman expectation operator of policy pi to value functions.

    Args:
        P: transitions, shape (S, A, S).
        R: expected rewards, shape (S, A).
        gamma: the discount.
        pi: a policy, shape (S, A), rows summing to one.
        v: one value function, shape (S,), or m of them as columns, shape (S, m).

    Returns:
        r_pi + gamma * P_pi @ v, the same shape as v, where
        r_pi[s] = sum_a pi[s, a] R[s, a] and P_pi[s, s2] = sum_a pi[s, a] P[s, a, s2].
    """
    raise NotImplementedError("bellman_backup")


def value_equivalence_loss(P_model, R_model, P, R, gamma, policies, values):
    """How far a model's Bellman backups are from the real ones.

    Args:
        P_model, R_model: the model, shapes (S, A, S) and (S, A).
        P, R: the true environment, same shapes.
        gamma: the discount.
        policies: a list of policies, each shape (S, A).
        values: value functions as columns, shape (S, m).

    Returns:
        A float: the mean, over every policy, state, and value function, of the
        squared difference between the model's backup and the true backup.
        It is zero exactly when the model is value-equivalent for these
        policies and values.
    """
    raise NotImplementedError("value_equivalence_loss")


def puct_scores(q, prior, visits, c1=1.25, c2=19652.0):
    """MuZero's pUCT score for every action at one node.

    score(a) = Q(a) + P(a) * sqrt(sum_b N(b)) / (1 + N(a))
                      * (c1 + log((sum_b N(b) + c2 + 1) / c2))

    Args:
        q: Q(a), shape (A,).
        prior: P(a), shape (A,).
        visits: N(a), shape (A,).
        c1, c2: MuZero's constants.

    Returns:
        The scores, shape (A,).
    """
    raise NotImplementedError("puct_scores")


def mcts(model_step, predict, root_state, n_actions, n_simulations, gamma, c1=1.25, c2=19652.0):
    """Monte Carlo tree search with pUCT over a deterministic model.

    Each simulation starts at the root and repeats:
      1. Pick a = argmax of puct_scores at the current node. On a tie, take the
         lowest action index (np.argmax does this).
      2. If the edge (node, a) has never been tried, call
         model_step(node_state, a) -> (next_state, reward, done), store the
         reward on the edge, and stop. The leaf value is 0 if done, else the
         value from predict(next_state), whose prior and value become the new
         node's. If the edge leads to a terminal state already found, stop with
         leaf value 0. Otherwise move to the child and repeat.
    Then back up: walking from the leaf to the root, G = reward + gamma * G
    starting from the leaf value, and at each edge add G to its total and 1 to
    its visit count.

    Q(a) at a node is the edge's total divided by its visit count. An action
    that has never been tried scores Q(a) = the node's own predicted value.

    Args:
        model_step: model_step(state, action) -> (next_state, reward, done).
        predict: predict(state) -> (prior of shape (A,), value).
        root_state: where to search from.
        n_actions: A.
        n_simulations: how many simulations to run.
        gamma: the discount.

    Returns:
        (visits, q): the root's visit counts and Q values, each shape (A,).
    """
    raise NotImplementedError("mcts")


# Helpers shared by the checks and the demo -----------------------------------


def toy_model(state, action):
    """A two-level tree. From the root, action 0 pays 0.5 and ends; action 1
    pays nothing and reaches state "B", where action 1 pays 1 and ends."""
    if state == "root":
        return ("A", 0.5, True) if action == 0 else ("B", 0.0, False)
    return ("end", float(action == 1), True)


def uniform_predict(n_actions):
    """A prediction function that knows nothing: uniform prior, zero value."""
    return lambda state: (np.full(n_actions, 1.0 / n_actions), 0.0)


def learn_deterministic_model(mdp, n_samples, rng):
    """Fit next-state, reward, and done tables from uniformly random transitions.

    Returns the model's step function, a (S, A) mask of the pairs it has seen,
    and state values from value iteration run inside the learned model.
    """
    S, A = mdp.n_states, mdp.n_actions
    nxt = np.zeros((S, A), dtype=int)
    rew = np.zeros((S, A))
    seen = np.zeros((S, A), dtype=bool)
    for _ in range(n_samples):
        s, a = int(rng.integers(S)), int(rng.integers(A))
        s2, r = mdp.sample(s, a, rng)
        nxt[s, a], rew[s, a], seen[s, a] = s2, r, True
    done = mdp.terminal[nxt]
    v = np.zeros(S)
    for _ in range(300):
        v = (rew + mdp.gamma * (~done) * v[nxt]).max(axis=1)
        v[mdp.terminal] = 0.0

    def step(s, a):
        return int(nxt[s, a]), float(rew[s, a]), bool(done[s, a])

    return step, seen, v


if __name__ == "__main__":
    rewards = np.array([0.0, 0.0, 1.0])
    values = np.array([0.5, 0.6, 0.8, 0.0])
    continues = np.array([1.0, 1.0, 0.0])
    for lam in (0.0, 0.5, 1.0):
        print(f"lambda = {lam}: {lambda_returns(rewards, values, continues, 0.9, lam)}")

    for sims in (4, 20, 200):
        visits, q = mcts(toy_model, uniform_predict(2), "root", 2, sims, gamma=0.9)
        print(f"toy tree, {sims:3d} simulations: visits {visits}, Q {np.round(q, 3)}")

    layout = ["S..#", ".#..", "...G"]
    mdp = rlenvs.gridworld(layout, gamma=0.9, step_reward=0.0, goal_reward=1.0)
    step, seen, v = learn_deterministic_model(mdp, 400, np.random.default_rng(0))
    start = mdp.names["start"]
    print(f"grid world: the learned model saw {seen.mean():.0%} of state-action pairs")
    visits, _ = mcts(step, uniform_predict(4), start, 4, 200, gamma=0.9)
    print(f"  MCTS with zero values, visits from the start: {visits}")
    visits, _ = mcts(step, lambda s: (np.full(4, 0.25), float(v[s])), start, 4, 200, gamma=0.9)
    print(f"  MCTS with values planned in the model:       {visits}")
    q_star = rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp))
    best = np.flatnonzero(np.isclose(q_star[start], q_star[start].max()))
    print(f"  the optimal first actions are {best} (1 is right, 2 is down)")

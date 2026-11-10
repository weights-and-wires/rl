"""Reference solution for the world models and MuZero exercise."""

import numpy as np

import rlenvs


def lambda_returns(rewards, values, continues, gamma, lam):
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    continues = np.asarray(continues, dtype=float)
    horizon = rewards.shape[0]
    out = np.zeros_like(rewards)
    following = values[horizon]
    for t in reversed(range(horizon)):
        blend = (1.0 - lam) * values[t + 1] + lam * following
        out[t] = rewards[t] + gamma * continues[t] * blend
        following = out[t]
    return out


def bellman_backup(P, R, gamma, pi, v):
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        return r_pi + gamma * P_pi @ v
    return r_pi[:, None] + gamma * P_pi @ v


def value_equivalence_loss(P_model, R_model, P, R, gamma, policies, values):
    values = np.asarray(values, dtype=float)
    total, count = 0.0, 0
    for pi in policies:
        gap = bellman_backup(P_model, R_model, gamma, pi, values) - bellman_backup(P, R, gamma, pi, values)
        total += float((gap**2).sum())
        count += gap.size
    return total / count


def puct_scores(q, prior, visits, c1=1.25, c2=19652.0):
    q = np.asarray(q, dtype=float)
    prior = np.asarray(prior, dtype=float)
    visits = np.asarray(visits, dtype=float)
    total = visits.sum()
    weight = c1 + np.log((total + c2 + 1.0) / c2)
    return q + prior * np.sqrt(total) / (1.0 + visits) * weight


class _Node:
    def __init__(self, state, prior, value, n_actions):
        self.state = state
        self.prior = np.asarray(prior, dtype=float)
        self.value = float(value)
        self.N = np.zeros(n_actions)
        self.W = np.zeros(n_actions)
        self.R = np.zeros(n_actions)
        self.children = {}

    @property
    def Q(self):
        # An action never tried scores the node's own predicted value.
        return np.divide(self.W, self.N, out=np.full_like(self.W, self.value), where=self.N > 0)


def mcts(model_step, predict, root_state, n_actions, n_simulations, gamma, c1=1.25, c2=19652.0):
    prior, value = predict(root_state)
    root = _Node(root_state, prior, value, n_actions)
    for _ in range(n_simulations):
        node, path = root, []
        while True:
            a = int(np.argmax(puct_scores(node.Q, node.prior, node.N, c1, c2)))
            path.append((node, a))
            if a not in node.children:
                s2, r, done = model_step(node.state, a)
                node.R[a] = r
                if done:
                    node.children[a] = None
                    value = 0.0
                else:
                    child_prior, value = predict(s2)
                    node.children[a] = _Node(s2, child_prior, value, n_actions)
                break
            if node.children[a] is None:
                value = 0.0
                break
            node = node.children[a]
        G = float(value)
        for node, a in reversed(path):
            G = node.R[a] + gamma * G
            node.W[a] += G
            node.N[a] += 1
    return root.N.copy(), root.Q.copy()


def encode(observation, weights):
    return np.asarray(weights, dtype=float) @ np.asarray(observation, dtype=float)


def latent_step(z, action, dynamics):
    z = np.asarray(z, dtype=float)
    transition = np.asarray(dynamics["W"], dtype=float)
    reward_row = np.asarray(dynamics["reward"], dtype=float)
    one_hot = np.zeros(transition.shape[1] - z.shape[0])
    one_hot[int(action)] = 1.0
    inp = np.concatenate([z, one_hot])
    return transition @ inp, float(reward_row @ inp)


def unroll_value(z, actions, dynamics, value_head, gamma):
    z = np.asarray(z, dtype=float)
    value_head = np.asarray(value_head, dtype=float)
    total = 0.0
    for step, action in enumerate(actions):
        z, reward = latent_step(z, action, dynamics)
        total += (gamma ** step) * reward
    horizon = len(actions)
    total += (gamma ** horizon) * float(value_head @ z)
    return total


def k_step_value_error(predicted, target):
    predicted = np.asarray(predicted, dtype=float)
    target = np.asarray(target, dtype=float)
    return float(np.mean((predicted - target) ** 2))


def learn_latent_model(observations, actions, rewards, next_observations, values, rank, n_actions):
    observations = np.asarray(observations, dtype=float)
    next_observations = np.asarray(next_observations, dtype=float)
    actions = np.asarray(actions, dtype=int)
    rewards = np.asarray(rewards, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    direction, *_ = np.linalg.lstsq(next_observations, values, rcond=None)
    encoder = np.zeros((rank, observations.shape[1]))
    value_head = np.zeros(rank)
    # The value is one direction. Put it in the first coordinate and read it with weight 1.
    encoder[0] = direction
    value_head[0] = 1.0
    latent = observations @ encoder.T
    latent_next = next_observations @ encoder.T
    one_hot = np.eye(n_actions)[actions]
    inputs = np.concatenate([latent, one_hot], axis=1)
    dynamics_T, *_ = np.linalg.lstsq(inputs, latent_next, rcond=None)
    reward, *_ = np.linalg.lstsq(inputs, rewards, rcond=None)
    return {
        "encoder": encoder,
        "dynamics": {
            "W": np.asarray(dynamics_T, dtype=float).T,
            "reward": np.asarray(reward, dtype=float),
        },
        "value_head": value_head,
    }


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

    observation = np.array([2.0, 5.0])
    latent = encode(observation, np.array([[1.0, 0.0]]))
    dynamics = {"W": np.array([[1.0, -1.0, 1.0]]), "reward": np.array([1.0, -1.0, 1.0])}
    head = np.array([1.0])
    print(f"latent of [2, 5]: {latent}")
    print(f"1-step unroll: {unroll_value(latent, [1], dynamics, head, 0.5)}")
    print(f"3-step unroll: {unroll_value(latent, [1, 0, 1], dynamics, head, 0.5)}")

"""Starter: returns-to-go, tokens, causal masking, and a tiny decision transformer.

Fill in the seven functions that raise NotImplementedError. The world, the dataset, the
transformer, and its training loop are written for you.
"""

import numpy as np
import torch
import torch.nn as nn

import rlenvs

torch.set_num_threads(1)

GRID = ["S....", ".....", ".....", ".....", "....G"]
HORIZON = 24
KIND_G, KIND_S, KIND_A = 0, 1, 2


def make_world():
    """A 5x5 grid. Every step pays -1, so the best return from S is -8."""
    return rlenvs.gridworld(GRID, gamma=1.0, step_reward=-1.0, goal_reward=0.0, max_steps=HORIZON)


def collect_mixed(mdp, n_episodes, seed):
    """Mixed-quality demonstrations: the optimal policy with a random eps of random moves."""
    rng = np.random.default_rng(seed)
    best = rlenvs.greedy(rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp)))
    trajs = []
    for _ in range(n_episodes):
        eps = rng.uniform(0.0, 0.2) if rng.random() < 0.5 else rng.uniform(0.8, 1.0)
        s = int(rng.choice(mdp.n_states, p=mdp.mu0))
        S, A, R = [], [], []
        for _ in range(HORIZON):
            a = int(rng.integers(mdp.n_actions)) if rng.random() < eps else int(best[s].argmax())
            s2, r = mdp.sample(s, a, rng)
            S.append(s), A.append(a), R.append(r)
            s = s2
            if mdp.terminal[s]:
                break
        trajs.append({"states": np.array(S), "actions": np.array(A), "rewards": np.array(R)})
    return trajs


def returns_to_go(rewards, gamma=1.0):
    """Return-to-go at every step: G_t = r_t + gamma * G_{t+1}, with nothing after the last step.

    Args:
        rewards: array of shape (T,), the reward at each step.
        gamma: discount factor. The decision transformer paper uses 1.0.

    Returns:
        Array of shape (T,). Entry 0 is the whole discounted return and entry T-1 is the last reward.
    """
    raise NotImplementedError("returns_to_go")


def build_tokens(rtg, states, actions):
    """Interleave (G_t, s_t, a_t) into one sequence of 3T tokens, or 3T - 1 if a_{T-1} is unknown.

    Args:
        rtg: length-T return-to-go values. states: length-T state ids. actions: length T, or T - 1
            at test time, when the last action is not chosen yet.

    Returns:
        (kinds, timesteps, values), three arrays of the same length. kinds uses KIND_G, KIND_S, and
        KIND_A; timesteps holds t for each of the three tokens of step t; values holds the token's number.
    """
    raise NotImplementedError("build_tokens")


def causal_mask(n):
    """Boolean array of shape (n, n). mask[i, j] is True when token i may attend to token j.

    Token i must see itself and earlier tokens only, so s_t sees G_t and a_{t-1} but not a_t.
    """
    raise NotImplementedError("causal_mask")


def masked_attention(x, mask, Wq, Wk, Wv):
    """One head of softmax attention over x of shape (n, d), blocked where mask is False.

    Wq, Wk, Wv have shape (d, d_head). Scores are q @ k.T / sqrt(d_head). Blocked scores get zero weight
    after the softmax. Returns an array of shape (n, d_head).
    """
    raise NotImplementedError("masked_attention")


def rollout_with_target(mdp, act, target, horizon, gamma=1.0, seed=0):
    """Run one episode, asking act(rtgs, states, actions) for each action.

    The target return starts at `target` and drops by each observed reward:
    G_{t+1} = (G_t - r_t) / gamma. Returns (episode return, the G_t used at each step).

    `act` gets lists with t + 1 targets, t + 1 states, and t actions at step t, and returns an int.
    Stop when the episode terminates or truncates, or after `horizon` steps. Use mdp.reset(seed) and mdp.step.
    """
    raise NotImplementedError("rollout_with_target")


def conditional_action_probs(actions, returns, target, n_actions):
    """P(a | G = target) from samples: action frequencies among samples whose return equals target.

    With no matching sample, return the uniform distribution. Compare returns with np.isclose.
    Returns an array of shape (n_actions,) that sums to one.
    """
    raise NotImplementedError("conditional_action_probs")


def offline_q_iteration(transitions, n_states, n_actions, gamma, iters):
    """Q-iteration on the dataset's own transitions (s, a, r, s2, done).

    Only (s, a) pairs seen in the data get a value. The backup takes the max over
    actions seen in s2. Unseen pairs stay at -inf. A done transition has no next-state term.
    Several transitions with the same (s, a) are averaged. Returns Q of shape (n_states, n_actions).
    """
    raise NotImplementedError("offline_q_iteration")


def max_return_from(trajs, start):
    """The best return among dataset trajectories that begin in `start`."""
    return max(float(np.sum(t["rewards"])) for t in trajs if t["states"][0] == start)


class TinyDT(nn.Module):
    """One masked attention layer over interleaved (G, s, a) tokens, then an MLP."""

    def __init__(self, n_states, n_actions, horizon, d=48):
        super().__init__()
        self.r = nn.Linear(1, d)
        self.s = nn.Embedding(n_states, d)
        self.a = nn.Embedding(n_actions, d)
        self.t = nn.Embedding(horizon, d)
        self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.head = nn.Linear(d, n_actions)
        self.d = d

    def forward(self, rtg, states, actions):
        """rtg (B, T) scaled floats, states and actions (B, T) ints. Returns logits (B, T, A)."""
        B, T = states.shape
        pos = self.t(torch.arange(T))[None]
        tok = torch.stack([self.r(rtg[..., None]) + pos, self.s(states) + pos, self.a(actions) + pos], dim=2)
        x = tok.reshape(B, 3 * T, self.d)
        mask = torch.as_tensor(causal_mask(3 * T))
        scores = (self.q(x) @ self.k(x).transpose(1, 2)) / self.d**0.5
        w = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        h = x + w @ self.v(x)
        h = h + self.mlp(h)
        return self.head(h[:, 1::3])


def _batch(trajs, horizon, gamma, scale):
    B = len(trajs)
    rtg, st, ac = np.zeros((B, horizon)), np.zeros((B, horizon), int), np.zeros((B, horizon), int)
    live = np.zeros((B, horizon), bool)
    for i, t in enumerate(trajs):
        n = len(t["actions"])
        rtg[i, :n] = returns_to_go(t["rewards"], gamma) / scale
        st[i, :n], ac[i, :n], live[i, :n] = t["states"], t["actions"], True
    return torch.tensor(rtg, dtype=torch.float32), torch.tensor(st), torch.tensor(ac), torch.tensor(live)


def train_dt(trajs, n_states, n_actions, steps=200, seed=0, gamma=1.0):
    """Fit a TinyDT by cross-entropy on each action, given the return-to-go before it."""
    torch.manual_seed(seed)
    model = TinyDT(n_states, n_actions, HORIZON)
    rtg, st, ac, live = _batch(trajs, HORIZON, gamma, float(HORIZON))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        logits = model(rtg, st, ac)
        loss = nn.functional.cross_entropy(logits[live], ac[live])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def dt_policy(model, seed=0):
    """An act(rtgs, states, actions) function for rollout_with_target that samples from the model."""
    rng = np.random.default_rng(seed)

    def act(rtgs, states, actions):
        T = len(states)
        a_pad = list(actions) + [0] * (T - len(actions))
        with torch.no_grad():
            logits = model(
                torch.tensor([rtgs], dtype=torch.float32) / HORIZON,
                torch.tensor([states]),
                torch.tensor([a_pad]),
            )
        p = torch.softmax(logits[0, -1], dim=-1).numpy().astype(float)
        return int(rng.choice(len(p), p=p / p.sum()))

    return act


def mean_return(mdp, act, target, n_episodes, horizon=HORIZON, seed=0):
    """The mean realized return of n episodes conditioned on `target`."""
    return float(np.mean([rollout_with_target(mdp, act, target, horizon, seed=seed + i)[0] for i in range(n_episodes)]))


if __name__ == "__main__":
    mdp = make_world()
    trajs = collect_mixed(mdp, 200, seed=0)
    returns = [float(t["rewards"].sum()) for t in trajs]
    print(f"{len(trajs)} episodes, returns from {min(returns):.0f} to {max(returns):.0f}")
    print("returns_to_go([-1, -1, -1]) =", returns_to_go([-1.0, -1.0, -1.0]))
    model = train_dt(trajs, mdp.n_states, mdp.n_actions)
    act = dt_policy(model)
    for target in (-8, -14, -20):
        print(f"target {target:>4}: mean realized return {mean_return(mdp, act, target, 20):.1f}")

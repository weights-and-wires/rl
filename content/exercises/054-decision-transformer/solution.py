"""Reference solution: returns-to-go, tokens, causal masking, and a tiny decision transformer."""

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
    """G_t = r_t + gamma * G_{t+1}: the reward at t and everything after it."""
    rewards = np.asarray(rewards, dtype=float)
    out = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def build_tokens(rtg, states, actions):
    """Interleave (G_t, s_t, a_t) into one sequence of 3T tokens, or 3T - 1 if a_{T-1} is unknown."""
    T = len(rtg)
    values, kinds, steps = [], [], []
    for t in range(T):
        group = [(KIND_G, rtg[t]), (KIND_S, states[t])]
        if t < len(actions):
            group.append((KIND_A, actions[t]))
        for kind, v in group:
            kinds.append(kind), values.append(float(v)), steps.append(t)
    return np.array(kinds), np.array(steps), np.array(values)


def causal_mask(n):
    """mask[i, j] is True when token i may attend to token j, which needs j <= i."""
    return np.tril(np.ones((n, n), dtype=bool))


def masked_attention(x, mask, Wq, Wk, Wv):
    """One head of softmax attention over x of shape (n, d), blocked where mask is False."""
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    scores = q @ k.T / np.sqrt(q.shape[1])
    scores = np.where(mask, scores, -np.inf)
    scores = scores - scores.max(axis=1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=1, keepdims=True)
    return w @ v


def rollout_with_target(mdp, act, target, horizon, gamma=1.0, seed=0):
    """Run one episode, asking act(rtgs, states, actions) for each action.

    The target return starts at `target` and drops by each observed reward:
    G_{t+1} = (G_t - r_t) / gamma. Returns (episode return, the G_t used at each step).
    """
    s = mdp.reset(seed)
    R = float(target)
    rtgs, states, actions, total = [], [], [], 0.0
    for _ in range(horizon):
        rtgs.append(R)
        states.append(s)
        a = int(act(rtgs, states, actions))
        actions.append(a)
        s, r, terminated, truncated = mdp.step(a)
        total += r
        R = (R - r) / gamma
        if terminated or truncated:
            break
    return total, rtgs[: len(actions)]


def conditional_action_probs(actions, returns, target, n_actions):
    """P(a | G = target) from samples: action frequencies among samples whose return equals target.

    With no matching sample, return the uniform distribution.
    """
    actions, returns = np.asarray(actions), np.asarray(returns, dtype=float)
    hit = np.isclose(returns, target)
    if not hit.any():
        return np.full(n_actions, 1.0 / n_actions)
    return np.bincount(actions[hit], minlength=n_actions) / hit.sum()


def offline_q_iteration(transitions, n_states, n_actions, gamma, iters):
    """Q-iteration on the dataset's own transitions (s, a, r, s2, done).

    Only (s, a) pairs seen in the data get a value. The backup takes the max over
    actions seen in s2. Unseen pairs stay at -inf.
    """
    Q = np.full((n_states, n_actions), -np.inf)
    seen = {}
    for s, a, r, s2, done in transitions:
        seen.setdefault((s, a), []).append((r, s2, done))
    for s, a in seen:
        Q[s, a] = 0.0
    for _ in range(iters):
        new = Q.copy()
        for (s, a), outs in seen.items():
            new[s, a] = np.mean([r + (0.0 if d else gamma * Q[s2].max()) for r, s2, d in outs])
        Q = new
    return Q


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

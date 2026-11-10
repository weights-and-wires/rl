"""Reference solution for the exploration exercise."""

import numpy as np
import torch

import rlenvs

torch.set_num_threads(1)


def count_bonus(counts, beta):
    counts = np.asarray(counts, dtype=float)
    return beta / np.sqrt(np.maximum(counts, 1.0))


def empirical_model(transition_counts, reward_sums):
    transition_counts = np.asarray(transition_counts, dtype=float)
    S, A, _ = transition_counts.shape
    n = transition_counts.sum(axis=2)
    P = np.zeros_like(transition_counts)
    seen = n > 0
    P[seen] = transition_counts[seen] / n[seen][:, None]
    s_idx, a_idx = np.nonzero(~seen)
    P[s_idx, a_idx, s_idx] = 1.0
    R = np.divide(reward_sums, n, out=np.zeros((S, A)), where=seen)
    return P, R, n


def plan(P, R, gamma, n_iters=500, tol=1e-8):
    Q = np.zeros(R.shape)
    for _ in range(n_iters):
        new = R + gamma * P @ Q.max(axis=1)
        if np.max(np.abs(new - Q)) < tol:
            return new
        Q = new
    return Q


def sample_posterior_mdp(transition_counts, reward_sums, rng, prior=1.0):
    transition_counts = np.asarray(transition_counts, dtype=float)
    S, A, _ = transition_counts.shape
    alpha = transition_counts + prior
    g = rng.standard_gamma(alpha)
    P = g / g.sum(axis=2, keepdims=True)
    n = transition_counts.sum(axis=2)
    mean = reward_sums / (n + 1.0)
    R = mean + rng.standard_normal((S, A)) / np.sqrt(n + 1.0)
    return P, R


def explore(mdp, method, steps, seed, beta=0.5, epsilon=0.1, replan_every=20):
    rng = np.random.default_rng(seed)
    S, A = mdp.n_states, mdp.n_actions
    counts = np.zeros((S, A, S))
    reward_sums = np.zeros((S, A))
    s = mdp.reset(seed=seed)
    total = 0.0
    Q = np.zeros((S, A))
    for t in range(steps):
        if t % replan_every == 0:
            if method == "posterior":
                P, R = sample_posterior_mdp(counts, reward_sums, rng)
                Q = plan(P, R, mdp.gamma)
            else:
                P, R, n = empirical_model(counts, reward_sums)
                if method == "optimism":
                    R = R + count_bonus(n, beta)
                Q = plan(P, R, mdp.gamma)
        if method == "epsilon" and rng.random() < epsilon:
            a = int(rng.integers(A))
        else:
            a = int(np.argmax(Q[s]))
        s2, r, terminated, truncated = mdp.step(a)
        counts[s, a, s2] += 1
        reward_sums[s, a] += r
        total += r
        s = s2
        if terminated or truncated:
            s = mdp.reset()
    P, R, _ = empirical_model(counts, reward_sums)
    return rlenvs.greedy(plan(P, R, mdp.gamma)), counts, total


def make_rnd(in_dim, out_dim, hidden, seed):
    torch.manual_seed(seed)

    def net():
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, out_dim)
        )

    target, predictor = net(), net()
    for p in target.parameters():
        p.requires_grad_(False)
    return target, predictor


def rnd_bonus(target, predictor, states):
    with torch.no_grad():
        return ((predictor(states) - target(states)) ** 2).sum(dim=1)


def rnd_train(target, predictor, states, steps, lr):
    opt = torch.optim.Adam(predictor.parameters(), lr=lr)
    with torch.no_grad():
        goal = target(states)
    for _ in range(steps):
        loss = ((predictor(states) - goal) ** 2).sum(dim=1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.item())


if __name__ == "__main__":
    mdp = rlenvs.river_swim()
    j_star = mdp.mu0 @ rlenvs.value_iteration(mdp)
    print(f"RiverSwim, 3000 steps. The optimal policy is worth {j_star:.3f} from the start.")
    for method in ("epsilon", "optimism", "posterior"):
        pi, counts, total = explore(mdp, method, steps=3000, seed=0)
        j = mdp.mu0 @ rlenvs.policy_evaluation(mdp, pi)
        far = counts[-1].sum()
        print(f"  {method:9s}: reward {total:6.1f}, visits to the far state {far:5.0f}, "
              f"final policy worth {j:.3f}")

    target, predictor = make_rnd(2, 8, 32, seed=0)
    gen = torch.Generator().manual_seed(0)
    seen = torch.rand(256, 2, generator=gen) * 0.5
    novel = torch.rand(256, 2, generator=gen) * 0.5 + 0.5
    rnd_train(target, predictor, seen, steps=300, lr=1e-2)
    print(f"RND bonus on visited states {rnd_bonus(target, predictor, seen).mean():.4f}, "
          f"on novel states {rnd_bonus(target, predictor, novel).mean():.4f}")

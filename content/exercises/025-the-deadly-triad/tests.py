"""Checks for the deadly triad exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, equal, fail, has, shape

def _stationary(P_pi):
    S = P_pi.shape[0]
    M = np.vstack([P_pi.T - np.eye(S), np.ones((1, S))])
    rhs = np.zeros(S + 1)
    rhs[-1] = 1.0
    mu = np.linalg.lstsq(M, rhs, rcond=None)[0]
    return mu / mu.sum()


def _problem(seed, n_states=6, n_features=3, gamma=0.9):
    """A random MDP, a random policy, its chain, and random linear features."""
    mdp = rlenvs.random_mdp(n_states=n_states, n_actions=2, gamma=gamma, seed=seed)
    rng = np.random.default_rng(seed + 100)
    pi = rng.dirichlet(np.ones(2), size=n_states)
    P_pi = np.einsum("sa,sat->st", pi, mdp.P)
    r_pi = (pi * mdp.R).sum(axis=1)
    d = _stationary(P_pi)
    Phi = rng.standard_normal((n_states, n_features))
    return mdp, pi, P_pi, r_pi, d, Phi


def _trajectory(mdp, pi, steps, seed):
    rng = np.random.default_rng(seed)
    s = 0
    S = np.zeros(steps, dtype=int)
    R = np.zeros(steps)
    S2 = np.zeros(steps, dtype=int)
    for t in range(steps):
        a = rng.choice(pi.shape[1], p=pi[s])
        s2, r = mdp.sample(s, a, rng)
        S[t], R[t], S2[t] = s, r, s2
        s = s2
    return S, R, S2


@check("td_fixed_point recovers V^pi exactly with one-hot features, and solves the projected Bellman equation")
def test_td_fixed_point_exact(sub):
    td_fixed_point = has(sub, "td_fixed_point")
    mdp, pi, P_pi, r_pi, d, Phi = _problem(seed=0)
    v = rlenvs.policy_evaluation(mdp, pi)
    w = td_fixed_point(np.eye(6), P_pi, r_pi, d, mdp.gamma)
    close(w, v, atol=1e-9, label="one-hot features: w should equal V^pi")
    w = np.asarray(td_fixed_point(Phi, P_pi, r_pi, d, mdp.gamma), dtype=float)
    shape(w, (3,), label="td_fixed_point weights")
    # At the TD fixed point, the Bellman residual is D-orthogonal to every feature.
    residual = r_pi + mdp.gamma * P_pi @ Phi @ w - Phi @ w
    close(Phi.T @ (d * residual), np.zeros(3), atol=1e-10,
          label="Phi^T D (Bellman residual) at the fixed point")


@check("The TD fixed point's error is within 1/(1 - gamma) of the best linear fit on 40 random problems")
def test_td_bound(sub):
    td_fixed_point = has(sub, "td_fixed_point")
    for seed in range(40):
        mdp, pi, P_pi, r_pi, d, Phi = _problem(seed=seed)
        v = rlenvs.policy_evaluation(mdp, pi)
        w = td_fixed_point(Phi, P_pi, r_pi, d, mdp.gamma)
        ve_td = float(d @ (Phi @ w - v) ** 2)
        D = np.diag(d)
        w_best = np.linalg.solve(Phi.T @ D @ Phi, Phi.T @ D @ v)
        ve_best = float(d @ (Phi @ w_best - v) ** 2)
        if ve_td < ve_best - 1e-10:
            fail(f"seed {seed}: VE at your fixed point, {ve_td:.5f}, is below the best "
                 f"possible {ve_best:.5f}, so it is not a least-squares-consistent answer")
        if ve_td > ve_best / (1 - mdp.gamma) + 1e-10:
            fail(f"seed {seed}: VE at your fixed point, {ve_td:.5f}, breaks the bound "
                 f"{ve_best / (1 - mdp.gamma):.5f} = best VE / (1 - gamma)")


@check("lstd with one-hot features equals the value of the empirical model, with terminal masking")
def test_lstd_tabular(sub):
    lstd = has(sub, "lstd")
    # Five transitions over three states. State 2's transition terminates.
    S = np.array([0, 0, 1, 1, 2])
    R = np.array([1.0, 0.0, 2.0, -1.0, 3.0])
    S2 = np.array([1, 0, 2, 0, 0])
    term = np.array([False, False, False, False, True])
    gamma = 0.5
    X, X2 = np.eye(3)[S], np.eye(3)[S2]
    w = lstd(X, R, X2, term, gamma)
    # Empirical model: P_hat from counts (terminal transitions lead nowhere), r_hat means.
    P_hat = np.zeros((3, 3))
    r_hat = np.zeros(3)
    for s, r, s2, t in zip(S, R, S2, term):
        n = np.sum(S == s)
        r_hat[s] += r / n
        if not t:
            P_hat[s, s2] += 1.0 / n
    want = np.linalg.solve(np.eye(3) - gamma * P_hat, r_hat)
    close(w, want, atol=1e-6, label="LSTD weights against (I - gamma P_hat)^-1 r_hat")


@check("lstd on a 20,000-step trajectory lands near the exact TD fixed point")
def test_lstd_converges(sub):
    lstd = has(sub, "lstd")
    td_fixed_point = has(sub, "td_fixed_point")
    mdp, pi, P_pi, r_pi, d, Phi = _problem(seed=3)
    S, R, S2 = _trajectory(mdp, pi, 20000, seed=0)
    w = lstd(Phi[S], R, Phi[S2], np.zeros(len(S), dtype=bool), mdp.gamma)
    w_td = td_fixed_point(Phi, P_pi, r_pi, d, mdp.gamma)
    gap = float(np.max(np.abs(Phi @ (np.asarray(w) - w_td))))
    if gap > 0.2:
        fail(f"LSTD's values should be within 0.2 of the TD fixed point's in every state "
             f"after 20,000 steps; the largest gap is {gap:.3f}")


@check("baird_features builds the 7-by-8 feature matrix of Baird's counterexample")
def test_baird_features(sub):
    baird_features = has(sub, "baird_features")
    Phi = np.asarray(baird_features(), dtype=float)
    shape(Phi, (7, 8), label="baird_features()")
    want = np.zeros((7, 8))
    want[np.arange(6), np.arange(6)] = 2.0
    want[:6, 7] = 1.0
    want[6, 6], want[6, 7] = 1.0, 2.0
    close(Phi, want, atol=0, label="baird_features()")


@check("baird_expected_update diverges with uniform weights and follows the exact decay on-policy")
def test_baird_divergence(sub):
    update = has(sub, "baird_expected_update")
    Phi = np.zeros((7, 8))
    Phi[np.arange(6), np.arange(6)] = 2.0
    Phi[:6, 7] = 1.0
    Phi[6, 6], Phi[6, 7] = 1.0, 2.0
    alpha, gamma = 0.01, 0.99
    uniform = np.full(7, 1 / 7)
    # One update against the linear map w + alpha Phi^T D (gamma 1 x_7^T - Phi) w.
    rng = np.random.default_rng(2)
    w = rng.standard_normal(8)
    target = gamma * np.tile(Phi[6], (7, 1))
    want = w + alpha * Phi.T @ np.diag(uniform) @ (target - Phi) @ w
    close(update(w.copy(), alpha, gamma, uniform), want, atol=1e-12, label="one uniform update")

    w0 = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=float)
    w = w0.copy()
    for _ in range(1000):
        w = update(w, alpha, gamma, uniform)
    if not np.linalg.norm(w) > 10 * np.linalg.norm(w0):
        fail(f"with uniform weights |w| should grow past ten times its start, "
             f"{10 * np.linalg.norm(w0):.1f}, in 1000 sweeps; got {np.linalg.norm(w):.1f}")

    on_policy = np.eye(7)[6]
    w = w0.copy()
    for _ in range(1000):
        w = update(w, alpha, gamma, on_policy)
    # Only state 6 is updated. Its value shrinks by 1 - alpha (1 - gamma) |x_7|^2 each sweep.
    want_v6 = 12.0 * (1 - alpha * (1 - gamma) * 5.0) ** 1000
    almost(Phi[6] @ w, want_v6, 1e-8, label="on-policy value of state 6 after 1000 sweeps")

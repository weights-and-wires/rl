"""Checks for the generalised advantage estimation exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, fail, finite, has, shape

R3 = np.array([1.0, 0.0, 2.0])
V3 = np.array([0.5, 1.0, 1.5])
NEXT3 = np.array([1.0, 1.5, 1.0])  # V(s_1), V(s_2), and V(s_3) = 1.0 if the episode is cut off
NO3 = np.zeros(3, dtype=bool)
END3 = np.array([False, False, True])


def _rollout():
    """Two episodes in one rollout of 7 steps.

    The first episode terminates at t = 3. The second runs t = 4..6 and is cut
    off by the end of the rollout, which bootstraps from last_value.
    """
    rewards = np.array([1.0, -0.5, 2.0, 0.3, 0.0, 1.5, -1.0])
    values = np.array([0.2, 0.9, -0.4, 1.1, 0.6, -0.3, 0.8])
    last_value = 0.7
    terminated = np.array([False, False, False, True, False, False, False])
    truncated = np.zeros(7, dtype=bool)
    next_values = np.append(values[1:], last_value)
    return rewards, values, next_values, terminated, truncated


def _mc_minus_v(rewards, values, next_values, terminated, truncated, gamma):
    """Discounted return to each episode's end, bootstrapped when cut off, minus V."""
    T = len(rewards)
    out = np.zeros(T)
    for t in range(T):
        g, disc = 0.0, 1.0
        for u in range(t, T):
            g += disc * rewards[u]
            disc *= gamma
            if terminated[u]:
                break
            if truncated[u] or u == T - 1:
                g += disc * next_values[u]
                break
        out[t] = g - values[t]
    return out


def _weighted_k_step(rewards, values, next_values, terminated, gamma, lam):
    """(1 - lam) * sum_k lam^(k-1) A^(k), tail mass on the longest estimator.

    One episode, ended either by termination or by the end of the arrays.
    """
    T = len(rewards)
    out = np.zeros(T)
    for t in range(T):
        n = T - t
        ks = []
        for k in range(1, n + 1):
            g = sum(gamma**l * rewards[t + l] for l in range(k))
            end = t + k - 1
            boot = 0.0 if terminated[end] else next_values[end]
            ks.append(g + gamma**k * boot - values[t])
        w = [(1 - lam) * lam ** (k - 1) for k in range(1, n)] + [lam ** (n - 1)]
        out[t] = float(np.dot(w, ks))
    return out


@check("td_errors computes r + gamma * V(s_{t+1}) - V(s_t), with no bootstrap after termination")
def test_td_errors(sub):
    td_errors = has(sub, "td_errors")
    got = td_errors(R3, V3, NEXT3, END3, 0.9)
    shape(got, (3,), label="td_errors output")
    close(got, np.array([1.4, 0.35, 0.5]), atol=1e-12, label="terminated at the third step")
    close(td_errors(R3, V3, NEXT3, NO3, 0.9), np.array([1.4, 0.35, 1.4]), atol=1e-12,
          label="not terminated: the third step bootstraps from 1.0")


@check("next_state_values uses the final observation's value at episode ends, not the reset state's")
def test_next_state_values(sub):
    next_state_values = has(sub, "next_state_values")
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    done = np.array([False, True, False, False, True])
    final_values = np.array([0.0, 9.0, 0.0, 0.0, 7.0])
    got = next_state_values(values, 6.0, done, final_values)
    shape(got, (5,), label="next_state_values output")
    close(got, np.array([2.0, 9.0, 4.0, 5.0, 7.0]), atol=1e-12, label="episode ends at t = 1 and t = 4")
    got = next_state_values(values, 6.0, np.zeros(5, dtype=bool), final_values)
    close(got, np.array([2.0, 3.0, 4.0, 5.0, 6.0]), atol=1e-12, label="no episode ends: shift, then last_value")


@check("gae reproduces the lesson's worked example, terminated and truncated")
def test_gae_worked_example(sub):
    gae = has(sub, "gae")
    got = gae(R3, V3, NEXT3, END3, NO3, 0.9, 0.8)
    shape(got, (3,), label="gae output")
    close(got, np.array([1.9112, 0.71, 0.5]), atol=1e-12, label="terminated at the third step")
    got = gae(R3, V3, NEXT3, NO3, END3, 0.9, 0.8)
    if np.allclose(got, [1.9112, 0.71, 0.5], atol=1e-9):
        fail("a truncated step was treated as a termination: it must bootstrap from next_values")
    close(got, np.array([2.37776, 1.358, 1.4]), atol=1e-12, label="truncated at the third step")


@check("gae with lambda = 0 equals the TD errors, and with lambda = 1 the Monte Carlo return minus V")
def test_gae_limits(sub):
    gae = has(sub, "gae")
    td_errors = has(sub, "td_errors")
    rewards, values, next_values, terminated, truncated = _rollout()
    close(gae(rewards, values, next_values, terminated, truncated, 0.9, 0.0),
          td_errors(rewards, values, next_values, terminated, 0.9), atol=1e-12, label="lambda = 0")
    want = _mc_minus_v(rewards, values, next_values, terminated, truncated, 0.9)
    got = gae(rewards, values, next_values, terminated, truncated, 0.9, 1.0)
    close(got, want, atol=1e-10, label="lambda = 1 across a termination at t = 3")


@check("gae never carries advantages across an episode boundary")
def test_gae_boundary(sub):
    gae = has(sub, "gae")
    rewards, values, next_values, terminated, truncated = _rollout()
    base = gae(rewards, values, next_values, terminated, truncated, 0.99, 0.95)
    changed = rewards.copy()
    changed[4:] += 10.0
    after = gae(changed, values, next_values, terminated, truncated, 0.99, 0.95)
    close(after[:4], base[:4], atol=1e-12,
          label="first episode's advantages after changing only the second episode's rewards")
    trunc = truncated.copy()
    trunc[1] = True
    cut = gae(rewards, values, next_values, terminated, trunc, 0.99, 0.95)
    close(cut[:2], gae(rewards[:2], values[:2], next_values[:2], terminated[:2], trunc[:2], 0.99, 0.95),
          atol=1e-12, label="advantages before a truncation at t = 1 depend only on steps 0 and 1")


@check("gae with 0 < lambda < 1 equals the weighted average of k-step advantages")
def test_gae_weighted_average(sub):
    gae = has(sub, "gae")
    rng = np.random.default_rng(0)
    T = 6
    rewards, values = rng.normal(size=T), rng.normal(size=T)
    next_values = np.append(values[1:], rng.normal())
    for ends in (np.zeros(T, dtype=bool), np.eye(T, dtype=bool)[-1]):
        for lam in (0.3, 0.8):
            want = _weighted_k_step(rewards, values, next_values, ends, 0.95, lam)
            got = gae(rewards, values, next_values, ends, np.zeros(T, dtype=bool), 0.95, lam)
            label = "terminated at the end" if ends.any() else "cut off at the end"
            close(got, want, atol=1e-10, label=f"lambda = {lam}, {label}")


@check("gae and lambda_returns work on a (T, N) batch, one environment per column")
def test_gae_batch(sub):
    gae = has(sub, "gae")
    lambda_returns = has(sub, "lambda_returns")
    rng = np.random.default_rng(1)
    T, N = 8, 3
    rewards, values, next_values = rng.normal(size=(T, N)), rng.normal(size=(T, N)), rng.normal(size=(T, N))
    terminated = rng.random((T, N)) < 0.2
    truncated = (rng.random((T, N)) < 0.2) & ~terminated
    got = gae(rewards, values, next_values, terminated, truncated, 0.9, 0.7)
    shape(got, (T, N), label="batched gae")
    for n in range(N):
        col = gae(rewards[:, n], values[:, n], next_values[:, n], terminated[:, n], truncated[:, n], 0.9, 0.7)
        close(got[:, n], col, atol=1e-12, label=f"column {n} against a single-environment call")
    ret = lambda_returns(rewards, values, next_values, terminated, truncated, 0.9, 0.7)
    close(ret, got + values, atol=1e-12, label="lambda_returns = gae + values")


@check("With the true V^pi as critic, GAE averages to A^pi(s, a) for every lambda, including a truncated end")
def test_gae_unbiased_with_true_values(sub):
    gae = has(sub, "gae")
    mdp = rlenvs.random_mdp(n_states=4, n_actions=2, gamma=0.9, seed=0)
    pi = rlenvs.uniform_policy(mdp)
    v = rlenvs.policy_evaluation(mdp, pi)
    adv = rlenvs.q_from_v(mdp, v) - v[:, None]

    # Vectorised rollouts of H steps from (s0, a0), then following pi, cut off
    # by a time limit after the last step.
    rng = np.random.default_rng(0)
    H, N, s0, a0 = 4, 20000, 2, 0
    S = mdp.n_states
    s = np.full(N, s0)
    a = np.full(N, a0)
    rewards, values, next_values = np.zeros((H, N)), np.zeros((H, N)), np.zeros((H, N))
    for t in range(H):
        rewards[t], values[t] = mdp.R[s, a], v[s]
        s2 = np.minimum((rng.random(N)[:, None] > np.cumsum(mdp.P[s, a], axis=1)).sum(axis=1), S - 1)
        next_values[t] = v[s2]
        s, a = s2, (rng.random(N) < pi[s2, 1]).astype(int)
    terminated = np.zeros((H, N), dtype=bool)
    truncated = np.zeros((H, N), dtype=bool)
    truncated[-1] = True
    for lam in (0.0, 0.5, 1.0):
        first = np.asarray(gae(rewards, values, next_values, terminated, truncated, 0.9, lam))[0]
        finite(first, label=f"GAE at t = 0, lambda = {lam}")
        mean, se = first.mean(), first.std(ddof=1) / np.sqrt(N)
        if not abs(mean - adv[s0, a0]) < 4 * se:
            fail(f"lambda = {lam}: mean GAE at the first step is {mean:.4f}, but A^pi = "
                 f"{adv[s0, a0]:.4f} (standard error {se:.4f}). A biased mean "
                 "usually means the truncated last step did not bootstrap.")

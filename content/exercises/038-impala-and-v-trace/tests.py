"""Checks for the V-trace exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, fail, has, shape

# The worked example in the lesson: three steps, gamma = 0.9, and importance
# ratios pi / mu of 0.5, 1.5, and 2.0.
R3 = np.array([1.0, 0.0, 1.0])
V3 = np.array([0.5, 0.4, 0.3])
BOOT3 = 0.2
RATIO3 = np.array([0.5, 1.5, 2.0])
NO3 = np.zeros(3, dtype=bool)


def _oracle(rewards, values, bootstrap, terminated, log_rhos, gamma, rho_bar, c_bar):
    """The V-trace definition as a double sum, with no recursion.

    v_s = V(x_s) + sum_{t >= s} (prod_{i=s}^{t-1} d_i c_i) rho_t * (r_t + d_t V(x_{t+1}) - V(x_t)),
    where d_t = gamma * (1 - terminated_t).
    """
    T = len(rewards)
    ratios = np.exp(log_rhos)
    rho = np.minimum(rho_bar, ratios)
    c = np.minimum(c_bar, ratios)
    d = gamma * (1.0 - terminated.astype(float))
    nxt = np.append(values[1:], bootstrap)
    delta = rho * (rewards + d * nxt - values)
    out = np.zeros(T)
    for s in range(T):
        total, trace = 0.0, 1.0
        for t in range(s, T):
            total += trace * delta[t]
            trace *= d[t] * c[t]
        out[s] = values[s] + total
    return out


def _nstep_returns(rewards, bootstrap, terminated, gamma):
    """Discounted reward to the end of the episode or the rollout, bootstrapped at the rollout's end."""
    T = len(rewards)
    out = np.zeros(T)
    for s in range(T):
        g, disc = 0.0, 1.0
        for t in range(s, T):
            g += disc * rewards[t]
            if terminated[t]:
                disc = 0.0
                break
            disc *= gamma
        out[s] = g + disc * bootstrap
    return out


def _random_rollout(seed, T=7):
    rng = np.random.default_rng(seed)
    rewards = rng.standard_normal(T)
    values = rng.standard_normal(T)
    bootstrap = float(rng.standard_normal())
    log_rhos = 0.7 * rng.standard_normal(T)
    return rewards, values, bootstrap, log_rhos


@check("clipped_ratios truncates rho at rho_bar and c at c_bar, each on its own")
def test_clipped_ratios(sub):
    clipped_ratios = has(sub, "clipped_ratios")
    rho, c = clipped_ratios(np.log(np.array([0.5, 1.0, 3.0])), rho_bar=1.0, c_bar=2.0)
    close(rho, np.array([0.5, 1.0, 1.0]), atol=1e-12, label="rho")
    close(c, np.array([0.5, 1.0, 2.0]), atol=1e-12, label="c")


@check("vtrace_targets matches the worked example: 1.2279, 1.0620, 1.1800")
def test_worked_example(sub):
    vtrace_targets = has(sub, "vtrace_targets")
    vs = np.asarray(vtrace_targets(R3, V3, BOOT3, NO3, np.log(RATIO3), 0.9, 1.0, 1.0))
    shape(vs, (3,), label="targets")
    close(vs, np.array([1.2279, 1.062, 1.18]), atol=1e-9, label="targets")


@check("on-policy (rho = c = 1) the targets equal the n-step return, whatever the values are")
def test_on_policy_is_n_step(sub):
    vtrace_targets = has(sub, "vtrace_targets")
    for seed in range(3):
        rewards, values, bootstrap, _ = _random_rollout(seed)
        zeros = np.zeros(len(rewards))
        no = np.zeros(len(rewards), dtype=bool)
        want = _nstep_returns(rewards, bootstrap, no, 0.95)
        for bars in [(1.0, 1.0), (5.0, 3.0)]:
            got = vtrace_targets(rewards, values, bootstrap, no, zeros, 0.95, *bars)
            close(got, want, atol=1e-10, label=f"seed {seed}, bars {bars}")
        got = vtrace_targets(rewards, 10 * values - 3, bootstrap, no, zeros, 0.95)
        close(got, want, atol=1e-10, label=f"seed {seed}, values changed")


@check("with c_bar = 0 each target is the importance-weighted one-step TD target")
def test_td0_limit(sub):
    vtrace_targets = has(sub, "vtrace_targets")
    got = vtrace_targets(R3, V3, BOOT3, NO3, np.log(RATIO3), 0.9, 1.0, 0.0)
    nxt = np.append(V3[1:], BOOT3)
    rho = np.minimum(1.0, RATIO3)
    want = V3 + rho * (R3 + 0.9 * nxt - V3)
    close(got, want, atol=1e-10, label="V + rho * delta")


@check("rho_bar and c_bar play different roles: targets match the definition when they differ")
def test_separate_clips(sub):
    # Wrong implementation caught here: multiplying the correction by the
    # clipped rho instead of the clipped c in the recursion.
    vtrace_targets = has(sub, "vtrace_targets")
    for seed in range(4):
        rewards, values, bootstrap, log_rhos = _random_rollout(seed)
        no = np.zeros(len(rewards), dtype=bool)
        for bars in [(2.0, 0.5), (0.8, 1.0), (1.0, 1.0)]:
            want = _oracle(rewards, values, bootstrap, no, log_rhos, 0.9, *bars)
            got = vtrace_targets(rewards, values, bootstrap, no, log_rhos, 0.9, *bars)
            close(got, want, atol=1e-10, label=f"seed {seed}, (rho_bar, c_bar) = {bars}")


@check("an episode end stops the correction, and the next episode's values never leak back")
def test_episode_end(sub):
    # Wrong implementation caught here: discounting by gamma at a terminal step
    # instead of zero, so values from the next episode flow into earlier targets.
    vtrace_targets = has(sub, "vtrace_targets")
    rewards, values, bootstrap, log_rhos = _random_rollout(11)
    term = np.zeros(7, dtype=bool)
    term[2] = True
    got = vtrace_targets(rewards, values, bootstrap, term, log_rhos, 0.9, 1.0, 1.0)
    want = _oracle(rewards, values, bootstrap, term, log_rhos, 0.9, 1.0, 1.0)
    close(got, want, atol=1e-10, label="targets with a terminal step at t = 2")
    rho = min(1.0, float(np.exp(log_rhos[2])))
    close(got[2], values[2] + rho * (rewards[2] - values[2]), atol=1e-10,
          label="target at the terminal step")
    changed = values.copy()
    changed[3:] += 5.0
    again = vtrace_targets(rewards, changed, bootstrap + 5.0, term, log_rhos, 0.9, 1.0, 1.0)
    close(again[:3], got[:3], atol=1e-10, label="targets before the terminal step after changing later values")


@check("vtrace_advantages uses the next target: worked example, and on-policy equals return minus value")
def test_advantages(sub):
    vtrace_targets = has(sub, "vtrace_targets")
    vtrace_advantages = has(sub, "vtrace_advantages")
    log_rhos = np.log(RATIO3)
    vs = vtrace_targets(R3, V3, BOOT3, NO3, log_rhos, 0.9)
    adv = vtrace_advantages(R3, V3, vs, BOOT3, NO3, log_rhos, 0.9, 1.0)
    close(adv, np.array([0.7279, 0.662, 0.88]), atol=1e-9, label="advantages")
    rewards, values, bootstrap, _ = _random_rollout(5)
    zeros = np.zeros(7)
    term = np.zeros(7, dtype=bool)
    term[3] = True
    vs = vtrace_targets(rewards, values, bootstrap, term, zeros, 0.9, 1.0, 1.0)
    adv = vtrace_advantages(rewards, values, vs, bootstrap, term, zeros, 0.9, 1.0)
    want = _nstep_returns(rewards, bootstrap, term, 0.9) - values
    close(adv, want, atol=1e-10, label="on-policy advantage, G_t - V(x_t)")


@check("interpolated_policy is min(rho_bar * mu, pi), renormalized")
def test_interpolated_policy(sub):
    interpolated_policy = has(sub, "interpolated_policy")
    pi = np.array([[0.6, 0.3, 0.1]])
    mu = np.array([[0.2, 0.3, 0.5]])
    close(interpolated_policy(pi, mu, 1.0), np.array([[1 / 3, 0.5, 1 / 6]]), atol=1e-12,
          label="rho_bar = 1")
    close(interpolated_policy(pi, mu, 100.0), pi, atol=1e-12, label="rho_bar = 100 leaves pi alone")
    got = interpolated_policy(pi, mu, 0.5)
    close(got.sum(axis=1), np.array([1.0]), atol=1e-12, label="row sums")
    close(got, np.array([[0.1, 0.15, 0.1]]) / 0.35, atol=1e-12, label="rho_bar = 0.5")


def _mdp_and_policies(seed=0):
    mdp = rlenvs.random_mdp(n_states=5, n_actions=3, gamma=0.9, seed=seed)
    rng = np.random.default_rng(seed + 100)
    pi = rng.dirichlet(np.ones(3), size=5)
    mu = rng.dirichlet(np.ones(3), size=5)
    return mdp, pi, mu


@check("vtrace_operator: on-policy it jumps to V^pi, and with c_bar = 0 it equals the one-step expectation of vtrace_targets")
def test_operator_limits(sub):
    vtrace_operator = has(sub, "vtrace_operator")
    vtrace_targets = has(sub, "vtrace_targets")
    mdp, pi, mu = _mdp_and_policies(0)
    V = np.random.default_rng(3).standard_normal(5)
    got = vtrace_operator(mdp, pi, pi, V, 1.0, 1.0)
    close(got, rlenvs.policy_evaluation(mdp, pi), atol=1e-9, label="on-policy operator output")
    # With c_bar = 0 the operator is V + E_mu[rho * delta], one step. Average
    # vtrace_targets over every (a, s') of a one-step rollout and compare.
    want = np.zeros(5)
    for s in range(5):
        for a in range(3):
            for s2 in range(5):
                p = mu[s, a] * mdp.P[s, a, s2]
                v = vtrace_targets(np.array([mdp.R[s, a]]), V[s:s + 1], V[s2],
                                   np.zeros(1, dtype=bool), np.log(pi[s:s + 1, a] / mu[s:s + 1, a]),
                                   0.9, 1.0, 1.0)
                want[s] += p * v[0]
    got0 = vtrace_operator(mdp, pi, mu, V, 1.0, 0.0)
    close(got0, want, atol=1e-10, label="c_bar = 0 operator vs enumerated expectation")


@check("the fixed point of the V-trace operator is the value of pi_rho_bar, for every c_bar")
def test_fixed_point(sub):
    vtrace_operator = has(sub, "vtrace_operator")
    interpolated_policy = has(sub, "interpolated_policy")
    for seed in range(3):
        mdp, pi, mu = _mdp_and_policies(seed)
        for rho_bar in [0.5, 1.0, 2.0]:
            target = rlenvs.policy_evaluation(mdp, interpolated_policy(pi, mu, rho_bar))
            for c_bar in [1.0, 0.3]:
                label = f"seed {seed}, rho_bar {rho_bar}, c_bar {c_bar}"
                once = vtrace_operator(mdp, pi, mu, target, rho_bar, c_bar)
                close(once, target, atol=1e-9, label=f"V^pi_rho_bar is unchanged by the operator ({label})")
    mdp, pi, mu = _mdp_and_policies(0)
    V = np.zeros(5)
    for _ in range(400):
        V = vtrace_operator(mdp, pi, mu, V, 1.0, 1.0)
    close(V, rlenvs.policy_evaluation(mdp, interpolated_policy(pi, mu, 1.0)), atol=1e-7,
          label="iterating the operator from zero")
    if np.max(np.abs(V - rlenvs.policy_evaluation(mdp, pi))) < 0.1:
        fail("with rho_bar = 1 the fixed point should differ from V^pi on this MDP, but it matches")

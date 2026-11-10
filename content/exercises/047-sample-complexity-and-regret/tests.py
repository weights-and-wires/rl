"""Checks for the sample-complexity and regret exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, fail, has, shape

SEEDS = (0, 1, 2, 3, 4)


def _two_state():
    """One action, two states: state 0 pays 1 and mostly stays, state 1 pays 0 forever."""
    P = np.array([[[0.8, 0.2]], [[0.0, 1.0]]])
    P_hat = np.array([[[0.6, 0.4]], [[0.0, 1.0]]])
    R = np.array([[1.0], [0.0]])
    pi = np.ones((2, 1))
    return P, P_hat, R, pi


def _chain():
    """Two states, two actions. State 1 pays 1 for staying (action 0)."""
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = 1.0
    P[0, 1, 1] = 1.0
    P[1, 0, 1] = 1.0
    P[1, 1, 0] = 1.0
    R = np.array([[0.0, 0.0], [1.0, 0.0]])
    return P, R


def _optimal_q(P, R, horizon):
    """Reference backward induction, written independently of the reader's code."""
    Q = np.zeros((horizon,) + R.shape)
    v = np.zeros(R.shape[0])
    for h in reversed(range(horizon)):
        Q[h] = R + np.einsum("sat,t->sa", P, v)
        v = Q[h].max(axis=1)
    return Q


def _pair(i, gamma=0.9):
    """A true random MDP and a perturbed copy of its transitions."""
    rng = np.random.default_rng(i)
    mdp = rlenvs.random_mdp(5, 3, gamma, seed=i)
    other = rlenvs.random_mdp(5, 3, gamma, seed=1000 + i)
    w = rng.uniform(0.02, 0.5)
    P_hat = (1 - w) * mdp.P + w * other.P
    pi = rng.dirichlet(np.ones(3), size=5)
    return mdp, P_hat, pi


@check("simulation_lemma_bound is gamma / (1 - gamma)^2 * eps * Rmax, with eps the largest row L1 distance")
def test_bound_value(sub):
    simulation_lemma_bound = has(sub, "simulation_lemma_bound")
    P, P_hat, R, _ = _two_state()
    # eps = |0.6 - 0.8| + |0.4 - 0.2| = 0.4, Rmax = 1, gamma = 0.5: 0.5 / 0.25 * 0.4 * 1 = 0.8.
    # Wrong implementation named in the task: gamma / (1 - gamma) * eps * Rmax, which gives 0.4.
    almost(simulation_lemma_bound(P_hat, P, R, 0.5), 0.8, 1e-12, label="bound for the two-state example")
    almost(simulation_lemma_bound(P, P, R, 0.5), 0.0, 1e-12, label="bound when the models agree")
    almost(simulation_lemma_bound(P_hat, P, 3.0 * R, 0.5), 2.4, 1e-12, label="bound with rewards scaled by 3")


@check("policy_q solves for the exact Q^pi and matches rlenvs.policy_evaluation")
def test_policy_q(sub):
    policy_q = has(sub, "policy_q")
    for seed in (0, 1, 2):
        mdp = rlenvs.random_mdp(5, 3, 0.9, seed=seed)
        pi = np.random.default_rng(seed).dirichlet(np.ones(3), size=5)
        got = policy_q(mdp.P, mdp.R, mdp.gamma, pi)
        shape(got, (5, 3), label="Q^pi")
        want = rlenvs.q_from_v(mdp, rlenvs.policy_evaluation(mdp, pi))
        close(got, want, atol=1e-9, label=f"Q^pi on random MDP {seed}")
    P, _, R, pi = _two_state()
    close(policy_q(P, R, 0.5, pi), [[1.0 + 0.5 * 0.8 * 1.0 / 0.6], [0.0]], atol=1e-9,
          label="Q^pi of the two-state example (V0 = 1 / 0.6)")


@check("simulation_gap equals Q^pi - Q_hat^pi exactly (the identity, not the bound)")
def test_simulation_gap(sub):
    simulation_gap = has(sub, "simulation_gap")
    policy_q = has(sub, "policy_q")
    P, P_hat, R, pi = _two_state()
    # V = [5/3, 0], V_hat = [1/0.7, 0]. The gap in state 0 is 5/3 - 10/7 = 5/21 = 0.2381.
    got = simulation_gap(P_hat, P, R, 0.5, pi)
    close(got, [[5.0 / 21.0], [0.0]], atol=1e-9, label="gap on the two-state example")
    # Wrong implementation named in the task: invert (I - gamma P^pi) built from the true P
    # while still multiplying V^pi. That is the identity with the roles half swapped, and it
    # gives a different answer whenever P and P_hat differ.
    for i in (0, 1, 2, 3):
        mdp, P_hat, pi = _pair(i)
        want = policy_q(mdp.P, mdp.R, mdp.gamma, pi) - policy_q(P_hat, mdp.R, mdp.gamma, pi)
        close(simulation_gap(P_hat, mdp.P, mdp.R, mdp.gamma, pi), want, atol=1e-9,
              label=f"Q^pi - Q_hat^pi on random pair {i}")


@check("the simulation-lemma bound holds on 200 random MDP pairs, for a random policy each")
def test_bound_holds(sub):
    simulation_lemma_bound = has(sub, "simulation_lemma_bound")
    policy_q = has(sub, "policy_q")
    worst = 0.0
    for i in range(200):
        mdp, P_hat, pi = _pair(i)
        gap = np.abs(policy_q(mdp.P, mdp.R, mdp.gamma, pi) - policy_q(P_hat, mdp.R, mdp.gamma, pi)).max()
        bound = simulation_lemma_bound(P_hat, mdp.P, mdp.R, mdp.gamma)
        if gap > bound + 1e-12:
            fail(f"pair {i}: max |Q^pi - Q_hat^pi| = {gap:.4f} is above the bound {bound:.4f}")
        worst = max(worst, gap / bound)
    if worst < 0.001:
        fail(f"the bound is vacuous: the worst gap is only {worst:.5f} of it, expected a ratio above 0.001")


@check("hoeffding_bonus is H * sqrt(log(2 S A H / delta) / (2 max(N, 1)))")
def test_hoeffding_bonus(sub):
    hoeffding_bonus = has(sub, "hoeffding_bonus")
    N = np.array([[0, 1], [4, 100]])
    got = hoeffding_bonus(N, 10, 0.1)
    shape(got, (2, 2), label="bonus")
    # log(2 * 4 * 10 / 0.1) = log 800 = 6.6846, so the N = 1 bonus is 10 * sqrt(3.3423) = 18.282.
    close(got, [[18.28197436, 18.28197436], [9.14098718, 1.82819744]], atol=1e-6,
          label="bonus for N = [[0, 1], [4, 100]], H = 10, delta = 0.1")


@check("backward_induction gives the optimal finite-horizon Q, and clip caps step h at horizon - h")
def test_backward_induction(sub):
    backward_induction = has(sub, "backward_induction")
    P, R = _chain()
    Q = backward_induction(P, R, 3)
    shape(Q, (3, 2, 2), label="Q")
    want = np.array([[[1, 2], [3, 1]], [[0, 1], [2, 0]], [[0, 0], [1, 0]]], dtype=float)
    close(Q, want, atol=1e-12, label="Q for the two-state chain, horizon 3")
    capped = backward_induction(P, R + 5.0, 3, clip=True)
    close(capped, np.array([3.0, 2.0, 1.0])[:, None, None] * np.ones((3, 2, 2)), atol=1e-12,
          label="clipped Q when every reward is inflated by 5")
    mdp = rlenvs.river_swim()
    close(backward_induction(mdp.P, mdp.R, 12), _optimal_q(mdp.P, mdp.R, 12), atol=1e-9,
          label="Q on RiverSwim, horizon 12")


@check("finite_horizon_value matches a brute-force sum over every action sequence")
def test_finite_horizon_value(sub):
    finite_horizon_value = has(sub, "finite_horizon_value")
    P, R = _chain()
    always_stay = np.zeros((3, 2, 2))
    always_stay[:, :, 0] = 1.0
    close(finite_horizon_value(P, R, always_stay), [0.0, 3.0], atol=1e-12, label="always action 0, horizon 3")
    rng = np.random.default_rng(0)
    pi = rng.dirichlet(np.ones(2), size=(3, 2))
    want = np.zeros(2)
    for s0 in range(2):
        for a0 in range(2):
            for s1 in range(2):
                for a1 in range(2):
                    for s2 in range(2):
                        for a2 in range(2):
                            p = pi[0, s0, a0] * P[s0, a0, s1] * pi[1, s1, a1] * P[s1, a1, s2] * pi[2, s2, a2]
                            want[s0] += p * (R[s0, a0] + R[s1, a1] + R[s2, a2])
    close(finite_horizon_value(P, R, pi), want, atol=1e-12, label="random step-dependent policy, horizon 3")


@check("ucbvi is optimistic: Q >= Q* at every step and state-action, in every episode, on five seeds")
def test_ucbvi_optimism(sub):
    ucbvi = has(sub, "ucbvi")
    mdp = rlenvs.river_swim()
    horizon, episodes = 5, 1000
    Q_star = _optimal_q(mdp.P, mdp.R, horizon)
    cap = (horizon - np.arange(horizon))[:, None, None] * np.ones((1,) + mdp.R.shape)
    for seed in SEEDS:
        Qs, N, regret = ucbvi(mdp, episodes, horizon, 0.1, seed)
        shape(Qs, (episodes, horizon, 6, 2), label="Qs")
        shape(N, (6, 2), label="N")
        shape(regret, (episodes,), label="regret")
        if not np.isclose(N.sum(), episodes * horizon):
            fail(f"seed {seed}: the counts should add up to {episodes * horizon} steps, got {N.sum():g}")
        slack = (Qs - Q_star[None]).min()
        if slack < -1e-9:
            k, h, s, a = np.unravel_index((Qs - Q_star[None]).argmin(), Qs.shape)
            fail(f"seed {seed}: Q is below Q* by {-slack:.4f} at episode {k}, step {h}, state {s}, action {a}: "
                 f"got {Qs[k, h, s, a]:.4f}, expected at least {Q_star[h, s, a]:.4f}")
        if (Qs > cap[None] + 1e-9).any():
            fail(f"seed {seed}: some Q exceeds the cap horizon - h; the clip is missing")
        # Optimism must not be bought by sitting at the cap everywhere.
        open_entries = int((Qs[-1] < cap - 1e-9).sum())
        if open_entries < 30:
            fail(f"seed {seed}: only {open_entries} of 60 final Q entries are below the cap, expected at least 30")


@check("on RiverSwim, UCB-VI has under a quarter of epsilon-greedy's regret, on five seeds")
def test_regret(sub):
    ucbvi = has(sub, "ucbvi")
    mdp = rlenvs.river_swim()
    horizon, episodes = 20, 300
    rows = []
    for seed in SEEDS:
        _, N, reg_opt = ucbvi(mdp, episodes, horizon, 0.1, seed, bonus_scale=0.1)
        _, _, reg_eps = ucbvi(mdp, episodes, horizon, 0.1, seed, bonus_scale=0.0, epsilon=0.1)
        rows.append((seed, reg_opt.sum(), reg_eps.sum(), reg_opt[:50].mean(), reg_opt[-50:].mean()))
        if reg_opt.sum() > 0.25 * reg_eps.sum():
            fail(f"seed {seed}: UCB-VI regret {reg_opt.sum():.1f} should be under a quarter of "
                 f"epsilon-greedy's {reg_eps.sum():.1f}")
        if reg_opt[-50:].mean() > 0.25 * reg_opt[:50].mean():
            fail(f"seed {seed}: UCB-VI regret per episode should fall: first 50 episodes {reg_opt[:50].mean():.2f}, "
                 f"last 50 episodes {reg_opt[-50:].mean():.2f}")


@check("Robbins-Monro sums: 1/n has a growing sum and a bounded sum of squares, constant alpha does not, and the variance is exact")
def test_step_sizes(sub):
    step_size_sums = has(sub, "step_size_sums")
    estimator_variance = has(sub, "estimator_variance")
    n = np.arange(1, 1001)
    total, squares = step_size_sums(1.0 / n)
    almost(total, 7.485470860550345, 1e-9, label="sum of 1/n for n up to 1000 (the harmonic number)")
    almost(squares, 1.6439345666815601, 1e-9, label="sum of 1/n^2 for n up to 1000 (close to pi^2 / 6 = 1.6449)")
    total, squares = step_size_sums(np.full(1000, 0.1))
    almost(total, 100.0, 1e-9, label="sum of constant alpha = 0.1 over 1000 steps")
    almost(squares, 10.0, 1e-9, label="sum of squares of constant alpha = 0.1 over 1000 steps (it grows with n)")
    # Wrong implementation named in the task: ignoring the decay prod (1 - alpha_j), so the
    # variance is sigma2 * sum(alpha_k^2). For alpha = 1/n that gives 1.64, not sigma2 / n.
    almost(estimator_variance(1.0 / n, 1.0), 0.001, 1e-12, label="variance of the 1/n estimate after 1000 samples")
    almost(estimator_variance(1.0 / n, 4.0), 0.004, 1e-12, label="variance with sigma2 = 4")
    almost(estimator_variance(np.full(1000, 0.1), 1.0), 0.1 / 1.9, 1e-9,
           label="variance of the alpha = 0.1 estimate, which settles at alpha / (2 - alpha)")

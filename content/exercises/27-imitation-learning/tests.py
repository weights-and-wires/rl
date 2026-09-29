"""Checks for the imitation learning exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, equal, fail, has, shape

EPS = 0.1
LEDGE = ["S........G", "..........", "HHHHHHHHHH"]


def _ledge():
    return rlenvs.gridworld(LEDGE, gamma=0.99, slip=0.0, step_reward=0.0, goal_reward=1.0,
                            hole_reward=-1.0, max_steps=200)


def _expert(mdp):
    return rlenvs.greedy(rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp)))


def _tightrope(eps, recover):
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = P[0, 1, 1] = P[1, 0, 0] = P[1, 1, 1] = 1.0
    R = np.array([[0.0, 0.0], [-1.0, -1.0]])
    mdp = rlenvs.TabularMDP(P=P, R=R, gamma=0.99, mu0=np.array([1.0, 0.0]))
    pi = np.array([[1.0 - eps, eps], [1.0, 0.0] if recover else [0.0, 1.0]])
    return mdp, pi


def _noisy(pi, eps):
    return (1.0 - eps) * pi + eps / pi.shape[1]


@check("fit_bc turns action counts into probabilities and is uniform on unseen states")
def test_fit_bc(sub):
    fit_bc = has(sub, "fit_bc")
    pi = fit_bc(np.array([0, 0, 0, 2, 0]), np.array([1, 1, 2, 0, 1]), 4, 3)
    shape(pi, (4, 3), label="policy")
    close(pi, [[0, 0.75, 0.25], [1 / 3, 1 / 3, 1 / 3], [1, 0, 0], [1 / 3, 1 / 3, 1 / 3]], atol=1e-12,
          label="BC policy from 4 labels in state 0 and 1 in state 2")


@check("add_mistakes mixes eps of uniform randomness into a policy")
def test_add_mistakes(sub):
    add_mistakes = has(sub, "add_mistakes")
    got = add_mistakes(np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]), 0.2)
    close(got, [[0.85, 0.05, 0.05, 0.05], [0.45, 0.45, 0.05, 0.05]], atol=1e-12, label="policy with eps = 0.2")


@check("finite_horizon_return matches the tightrope closed forms: eps T^2 / 2 growth without recovery, eps T with it")
def test_tightrope(sub):
    fhr = has(sub, "finite_horizon_return")
    eps = 0.01
    for T in (10, 100, 1000):
        mdp, stuck = _tightrope(eps, recover=False)
        want = T - (1 - (1 - eps) ** T) / eps
        almost(-fhr(mdp, stuck, T), want, 1e-8 * max(1.0, want), label=f"cost without recovery, T = {T}")
        mdp, back = _tightrope(eps, recover=True)
        want = eps / (1 + eps) * (T - (1 - (-eps) ** T) / (1 + eps))
        almost(-fhr(mdp, back, T), want, 1e-9 * max(1.0, want), label=f"cost with recovery, T = {T}")
    almost(-fhr(mdp, stuck, 100), 36.6032, 1e-4, label="the worked example, eps = 0.01 and T = 100")


@check("finite_horizon_return stops counting once an episode ends in a terminal state")
def test_fhr_terminal(sub):
    fhr = has(sub, "finite_horizon_return")
    mdp = _ledge()
    expert = _expert(mdp)
    # The expert walks 9 steps right; only the 9th pays the goal reward of 1.
    almost(fhr(mdp, expert, 8), 0.0, 1e-12, label="8 steps, still on the ledge")
    almost(fhr(mdp, expert, 9), 1.0, 1e-12, label="9 steps, the goal reached")
    almost(fhr(mdp, expert, 50), 1.0, 1e-12, label="50 steps, nothing after the goal")


@check("collect records non-terminal states only, and visits them at the expected rates")
def test_collect(sub):
    collect = has(sub, "collect")
    mdp = _ledge()
    expert = _expert(mdp)
    got = np.asarray(collect(mdp, expert, 3, 200, np.random.default_rng(0)))
    start = mdp.names["start"]
    path = np.arange(start, start + 9)
    equal(np.sort(got), np.sort(np.tile(path, 3)), label="states from 3 expert episodes on the ledge")

    small = rlenvs.random_mdp(4, 2, gamma=0.9, seed=5)
    pi = np.random.default_rng(5).dirichlet(np.ones(2), size=4)
    n, steps = 8000, 5
    visits = np.bincount(np.asarray(collect(small, pi, n, steps, np.random.default_rng(1))), minlength=4) / n
    P_pi = np.einsum("sa,sat->st", pi, small.P)
    d, want = small.mu0.copy(), np.zeros(4)
    for _ in range(steps):
        want += d
        d = d @ P_pi
    close(visits, want, atol=0.08, label="mean visits per episode against the exact expectation")


@check("disagreement matches the tightrope's closed form, eps d(on) + d(off)")
def test_disagreement(sub):
    disagreement = has(sub, "disagreement")
    eps = 0.1
    mdp, learner = _tightrope(eps, recover=False)
    expert = np.array([[1.0, 0.0], [1.0, 0.0]])
    d_on = (1 - 0.99) / (1 - 0.99 * (1 - eps))
    almost(disagreement(mdp, learner, expert, learner), eps * d_on + (1 - d_on), 1e-10,
           label="disagreement under the learner's own states")
    almost(disagreement(mdp, learner, expert, expert), eps, 1e-10,
           label="disagreement under the expert's states, which never leave the rope")


@check("BC matches the expert on every expert state, yet loses about 0.2 of value to compounding mistakes")
def test_bc_gap(sub):
    fit_bc = has(sub, "fit_bc")
    collect = has(sub, "collect")
    disagreement = has(sub, "disagreement")
    mdp = _ledge()
    expert = _expert(mdp)
    labels = expert.argmax(axis=1)
    start = mdp.names["start"]
    j_ref = rlenvs.policy_evaluation(mdp, _noisy(expert, EPS))[start]
    for n_demos in (1, 50):
        states = np.asarray(collect(mdp, expert, n_demos, 200, np.random.default_rng(n_demos)))
        bc = fit_bc(states, labels[states], mdp.n_states, 4)
        almost(disagreement(mdp, bc, expert, expert), 0.0, 1e-12,
               label=f"BC's error on the expert's own states, {n_demos} demos")
        own = disagreement(mdp, bc, expert, _noisy(bc, EPS))
        j_bc = rlenvs.policy_evaluation(mdp, _noisy(bc, EPS))[start]
        if j_ref - j_bc < 0.15 or own < 0.02:
            fail(f"with {n_demos} demos, expected BC to lose over 0.15 of value against the expert making "
                 f"the same mistakes ({j_ref:.4f}), got {j_bc:.4f}, and to disagree over 0.02 on its "
                 f"own states, got {own:.4f}")


@check("DAgger closes the gap: within 0.005 of the expert's value on three seeds, with almost no disagreement")
def test_dagger(sub):
    dagger = has(sub, "dagger")
    disagreement = has(sub, "disagreement")
    mdp = _ledge()
    expert = _expert(mdp)
    start = mdp.names["start"]
    j_ref = rlenvs.policy_evaluation(mdp, _noisy(expert, EPS))[start]
    for seed in (0, 1, 2):
        policies = dagger(mdp, expert, EPS, n_iters=15, n_episodes=20, max_steps=200, seed=seed)
        if len(policies) != 15:
            fail(f"dagger should return one policy per iteration, 15, got {len(policies)}")
        final = np.asarray(policies[-1])
        j = rlenvs.policy_evaluation(mdp, _noisy(final, EPS))[start]
        own = disagreement(mdp, final, expert, _noisy(final, EPS))
        if j_ref - j > 0.005 or own > 0.002:
            fail(f"seed {seed}: expected the final DAgger policy within 0.005 of {j_ref:.4f} and "
                 f"disagreement under 0.002 on its own states, got {j:.4f} and {own:.4f}")
        j_first = rlenvs.policy_evaluation(mdp, _noisy(np.asarray(policies[0]), EPS))[start]
        if j_ref - j_first < 0.15:
            fail(f"seed {seed}: the first DAgger policy is BC on expert data and should still show the "
                 f"gap, but it is worth {j_first:.4f} against {j_ref:.4f}")

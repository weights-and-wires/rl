"""Checks for the world models and MuZero exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, equal, fail, has, shape

GRID = ["S..#", ".#..", "...G"]


def _grid():
    return rlenvs.gridworld(GRID, gamma=0.9, step_reward=0.0, goal_reward=1.0)


def _deterministic_step(mdp):
    nxt = mdp.P.argmax(axis=2)

    def step(s, a):
        s2 = int(nxt[s, a])
        return s2, float(mdp.R[s, a]), not bool(mdp.terminal[s2])

    return step


def _toy_model(state, action):
    if state == "root":
        return ("A", 0.5, True) if action == 0 else ("B", 0.0, False)
    return ("end", float(action == 1), True)


def _uniform(n_actions):
    return lambda state: (np.full(n_actions, 1.0 / n_actions), 0.0)


def _learned_grid_model(mdp, n_samples, seed):
    rng = np.random.default_rng(seed)
    S, A = mdp.n_states, mdp.n_actions
    nxt = np.zeros((S, A), dtype=int)
    rew = np.zeros((S, A))
    for _ in range(n_samples):
        s, a = int(rng.integers(S)), int(rng.integers(A))
        s2, r = mdp.sample(s, a, rng)
        nxt[s, a], rew[s, a] = s2, r
    done = mdp.terminal[nxt]
    # Value iteration inside the learned model.
    v = np.zeros(S)
    for _ in range(300):
        v = (rew + mdp.gamma * (~done) * v[nxt]).max(axis=1)
        v[mdp.terminal] = 0.0

    def step(s, a):
        return int(nxt[s, a]), float(rew[s, a]), bool(done[s, a])

    return step, v


@check("lambda_returns matches the worked example for lambda = 0, 0.5, and 1")
def test_lambda_hand(sub):
    lambda_returns = has(sub, "lambda_returns")
    r = np.array([0.0, 0.0, 1.0])
    v = np.array([0.5, 0.6, 0.8, 0.0])
    c = np.array([1.0, 1.0, 0.0])
    got0 = lambda_returns(r, v, c, 0.9, 0.0)
    shape(got0, (3,), label="lambda_returns for H=3")
    close(got0, [0.54, 0.72, 1.0], atol=1e-12, label="lambda = 0, one-step targets")
    close(lambda_returns(r, v, c, 0.9, 0.5), [0.6345, 0.81, 1.0], atol=1e-12, label="lambda = 0.5")
    close(lambda_returns(r, v, c, 0.9, 1.0), [0.81, 0.9, 1.0], atol=1e-12, label="lambda = 1, Monte Carlo")


@check("lambda_returns with lambda = 1 is the discounted reward sum plus gamma^(H-t) v_H, whatever the middle values")
def test_lambda_limits(sub):
    lambda_returns = has(sub, "lambda_returns")
    rng = np.random.default_rng(0)
    H, gamma = 6, 0.8
    r = rng.standard_normal(H)
    v = rng.standard_normal(H + 1)
    c = np.ones(H)
    want = np.array([sum(gamma ** (k - t) * r[k] for k in range(t, H)) + gamma ** (H - t) * v[H]
                     for t in range(H)])
    close(lambda_returns(r, v, c, gamma, 1.0), want, atol=1e-10, label="lambda = 1")
    close(lambda_returns(r, v, c, gamma, 0.0), r + gamma * v[1:], atol=1e-12, label="lambda = 0")
    c[2] = 0.0
    got = lambda_returns(r, v, c, gamma, 0.7)
    almost(got[2], r[2], 1e-12, label="the return at a step whose next state is terminal")
    almost(got[1], r[1] + gamma * (0.3 * v[2] + 0.7 * r[2]), 1e-12, label="the step before it")


@check("lambda_returns minus the values equals GAE advantages")
def test_lambda_gae(sub):
    lambda_returns = has(sub, "lambda_returns")
    rng = np.random.default_rng(1)
    H, gamma, lam = 8, 0.95, 0.9
    r = rng.standard_normal(H)
    v = rng.standard_normal(H + 1)
    c = (rng.random(H) > 0.2).astype(float)
    delta = r + gamma * c * v[1:] - v[:-1]
    adv = np.zeros(H)
    running = 0.0
    for t in reversed(range(H)):
        running = delta[t] + gamma * lam * c[t] * running
        adv[t] = running
    close(lambda_returns(r, v, c, gamma, lam) - v[:-1], adv, atol=1e-10, label="lambda-return minus value")


@check("imagine with an exact model and V^pi gives lambda-returns equal to V^pi at every step, for every lambda")
def test_imagine(sub):
    imagine = has(sub, "imagine")
    lambda_returns = has(sub, "lambda_returns")
    mdp = _grid()
    pi = rlenvs.greedy(rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp)))
    v_pi = rlenvs.policy_evaluation(mdp, pi)
    start = mdp.names["start"]
    rewards, values, continues = imagine(_deterministic_step(mdp), lambda s: int(pi[s].argmax()),
                                         lambda s: float(v_pi[s]), start, 8)
    shape(rewards, (8,), label="rewards")
    shape(values, (9,), label="values")
    shape(continues, (8,), label="continues")
    almost(values[0], v_pi[start], 1e-12, label="values[0], the start state's value")
    # The greedy path takes 5 steps to reach G, so continues is 1 four times, then 0.
    equal(np.asarray(continues, dtype=float), [1, 1, 1, 1, 0, 0, 0, 0], label="continues")
    for lam in (0.0, 0.5, 0.95, 1.0):
        got = lambda_returns(rewards, values, continues, mdp.gamma, lam)
        close(got, values[:-1], atol=1e-10, label=f"lambda-returns at lambda = {lam}")


@check("bellman_backup leaves V^pi unchanged and handles a batch of value functions")
def test_bellman(sub):
    bellman_backup = has(sub, "bellman_backup")
    mdp = rlenvs.random_mdp(5, 3, gamma=0.9, seed=3)
    rng = np.random.default_rng(3)
    pi = rng.dirichlet(np.ones(3), size=5)
    v_pi = rlenvs.policy_evaluation(mdp, pi)
    close(bellman_backup(mdp.P, mdp.R, mdp.gamma, pi, v_pi), v_pi, atol=1e-10, label="T_pi v_pi")
    V = rng.standard_normal((5, 4))
    got = bellman_backup(mdp.P, mdp.R, mdp.gamma, pi, V)
    shape(got, (5, 4), label="backup of 4 value functions")
    P_pi = np.einsum("sa,sat->st", pi, mdp.P)
    want = (pi * mdp.R).sum(axis=1)[:, None] + mdp.gamma * P_pi @ V
    close(got, want, atol=1e-12, label="r_pi + gamma P_pi V")


@check("value_equivalence_loss is zero for a wrong model that agrees on the chosen values, and positive otherwise")
def test_value_equivalence(sub):
    loss = has(sub, "value_equivalence_loss")
    mdp = rlenvs.random_mdp(6, 2, gamma=0.9, seed=4)
    rng = np.random.default_rng(4)
    policies = [rng.dirichlet(np.ones(2), size=6) for _ in range(2)]
    values = np.stack([rlenvs.policy_evaluation(mdp, pi) for pi in policies], axis=1)

    # Perturb every row of P in a direction orthogonal to 1 and to both values.
    basis, _ = np.linalg.qr(np.concatenate([np.ones((6, 1)), values], axis=1))
    noise = rng.standard_normal(mdp.P.shape)
    delta = noise - np.einsum("sak,jk->saj", noise @ basis, basis)
    scale = 0.5 * np.min(mdp.P[delta < 0] / -delta[delta < 0])
    P_model = mdp.P + scale * delta

    almost(loss(mdp.P, mdp.R, mdp.P, mdp.R, 0.9, policies, values), 0.0, 1e-20, label="the true model")
    gap = np.abs(P_model - mdp.P).max()
    if gap < 1e-3:
        fail(f"test setup: the value-equivalent model should differ from P, max gap {gap:.2e}")
    almost(loss(P_model, mdp.R, mdp.P, mdp.R, 0.9, policies, values), 0.0, 1e-20,
           label=f"a model whose P differs by up to {gap:.3f} but agrees on both value functions")

    wrong = mdp.P + scale * (noise - noise.mean(axis=2, keepdims=True))
    wrong = np.clip(wrong, 0.0, None)
    wrong /= wrong.sum(axis=2, keepdims=True)
    got = loss(wrong, mdp.R, mdp.P, mdp.R, 0.9, policies, values)
    P1 = np.einsum("sa,sat->st", policies[0], wrong - mdp.P)
    P2 = np.einsum("sa,sat->st", policies[1], wrong - mdp.P)
    want = (np.sum((0.9 * P1 @ values) ** 2) + np.sum((0.9 * P2 @ values) ** 2)) / (2 * 6 * 2)
    almost(got, want, 1e-12, label="the mean squared backup gap for a model that is not value-equivalent")
    if got <= 0:
        fail("a model that is not value-equivalent must have a positive loss")
    # Theory, confirmed: evaluating policy 0 inside the equivalent model returns its true value.
    model_mdp = rlenvs.TabularMDP(P=P_model, R=mdp.R, gamma=0.9, mu0=mdp.mu0)
    close(rlenvs.policy_evaluation(model_mdp, policies[0]), values[:, 0], atol=1e-9,
          label="V^pi computed inside the value-equivalent model")


@check("puct_scores matches the pUCT rule by hand")
def test_puct(sub):
    puct_scores = has(sub, "puct_scores")
    q = np.array([0.5, 0.0])
    prior = np.array([0.5, 0.5])
    got = puct_scores(q, prior, np.array([2.0, 0.0]), 1.25, 19652.0)
    shape(got, (2,), label="scores")
    weight = 1.25 + np.log((2 + 19652 + 1) / 19652)
    close(got, [0.5 + 0.5 * np.sqrt(2) / 3 * weight, 0.5 * np.sqrt(2) * weight], atol=1e-12,
          label="scores after two visits to action 0")
    close(puct_scores(q, prior, np.zeros(2), 1.25, 19652.0), q, atol=1e-12,
          label="with no visits the exploration term is zero")


@check("mcts reproduces the hand trace on the toy tree after 4 simulations")
def test_mcts_trace(sub):
    mcts = has(sub, "mcts")
    visits, q = mcts(_toy_model, _uniform(2), "root", 2, 4, 0.9)
    close(visits, [3.0, 1.0], atol=0, label="root visit counts")
    close(q, [0.5, 0.0], atol=1e-12, label="root Q")


@check("mcts on the toy tree finds the delayed reward: most of 200 visits go to action 1, with Q near 0.9")
def test_mcts_toy(sub):
    mcts = has(sub, "mcts")
    visits, q = mcts(_toy_model, _uniform(2), "root", 2, 200, 0.9)
    almost(np.sum(visits), 200, 0, label="total root visits")
    if visits[1] < 150:
        fail(f"expected at least 150 of 200 visits on action 1, got visits {visits}")
    if not (0.8 < q[1] <= 0.9 + 1e-12):
        fail(f"expected Q[1] between 0.8 and 0.9 (the optimum is 0.9), got {q[1]:.4f}")


@check("mcts over a model learned from 400 random transitions, with values planned inside it, picks an optimal action from every state")
def test_mcts_grid(sub):
    mcts = has(sub, "mcts")
    mdp = _grid()
    q_star = rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp))
    step, v = _learned_grid_model(mdp, 400, seed=0)
    blind, _ = mcts(step, _uniform(4), mdp.names["start"], 4, 200, 0.9)
    if blind.max() - blind.min() > 0:
        fail(f"with zero values and a reward 5 steps away, 200 simulations should see no reward "
             f"and spread visits evenly, got {blind}")
    for s in range(mdp.n_states):
        if mdp.terminal[s]:
            continue
        visits, _ = mcts(step, lambda x: (np.full(4, 0.25), float(v[x])), s, 4, 200, 0.9)
        a = int(np.argmax(visits))
        if not np.isclose(q_star[s, a], q_star[s].max()):
            fail(f"from cell {mdp.names['cells'][s]} MCTS chose action {a} with visits {visits}; "
                 f"optimal Q values are {np.round(q_star[s], 3)}")

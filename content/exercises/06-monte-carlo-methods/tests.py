"""Checks for the Monte Carlo methods exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, equal, fail, has, shape


def _random_walk(n=5):
    S = n + 2
    P = np.zeros((S, 1, S))
    R3 = np.zeros((S, 1, S))
    for s in range(1, n + 1):
        P[s, 0, s - 1] = 0.5
        P[s, 0, s + 1] = 0.5
    P[0, 0, 0] = 1.0
    P[S - 1, 0, S - 1] = 1.0
    R3[n, 0, n + 1] = 1.0
    terminal = np.zeros(S, dtype=bool)
    terminal[[0, S - 1]] = True
    mu0 = np.zeros(S)
    mu0[(n + 1) // 2] = 1.0
    R = (P * R3).sum(axis=2)
    return rlenvs.TabularMDP(P=P, R=R, gamma=1.0, mu0=mu0, terminal=terminal, R3=R3)


def _episode(mdp, pi, rng, max_steps=1000):
    states, actions, rewards = [], [], []
    s = int(rng.choice(mdp.n_states, p=mdp.mu0))
    for _ in range(max_steps):
        if mdp.terminal[s]:
            break
        a = int(rng.choice(mdp.n_actions, p=pi[s]))
        s2, r = mdp.sample(s, a, rng)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s2
    return np.array(states), np.array(actions), np.array(rewards, dtype=float)


def _one_step_problem():
    # State 0 has two actions and both end the episode in state 1.
    # Action 0 ("L") pays 1 and action 1 ("R") pays 0.
    P = np.zeros((2, 2, 2))
    P[:, :, 1] = 1.0
    R = np.array([[1.0, 0.0], [0.0, 0.0]])
    return rlenvs.TabularMDP(
        P=P, R=R, gamma=1.0, mu0=np.array([1.0, 0.0]), terminal=np.array([False, True])
    )


def _soft_mdp(mdp, epsilon):
    # Forcing an epsilon-soft policy is the same as playing an MDP whose every
    # action is replaced, with probability epsilon, by a uniform random one.
    P = (1 - epsilon) * mdp.P + epsilon * mdp.P.mean(axis=1, keepdims=True)
    R = (1 - epsilon) * mdp.R + epsilon * mdp.R.mean(axis=1, keepdims=True)
    return rlenvs.TabularMDP(P=P, R=R, gamma=mdp.gamma, mu0=mdp.mu0, terminal=mdp.terminal)


@check("discounted_returns computes G_t for every step by a backward pass")
def test_discounted_returns(sub):
    f = has(sub, "discounted_returns")
    close(f(np.array([1.0, 0.0, 2.0]), 1.0), np.array([3.0, 2.0, 2.0]), atol=1e-12,
          label="rewards (1, 0, 2) with gamma = 1")
    # With gamma = 0.9: G_2 = 2, G_1 = 0 + 0.9 * 2 = 1.8, G_0 = 1 + 0.9 * 1.8 = 2.62.
    close(f(np.array([1.0, 0.0, 2.0]), 0.9), np.array([2.62, 1.8, 2.0]), atol=1e-12,
          label="rewards (1, 0, 2) with gamma = 0.9")


@check("mc_prediction gives 3 for A on first visit and 2.5 on every visit in the A, B, A episode")
def test_first_and_every_visit(sub):
    f = has(sub, "mc_prediction")
    # States A = 0 and B = 1; the episode visits A, B, A with rewards 1, 0, 2.
    episode = (np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([1.0, 0.0, 2.0]))
    first = f([episode], n_states=3, gamma=1.0, first_visit=True)
    every = f([episode], n_states=3, gamma=1.0, first_visit=False)
    shape(first, (3,), label="mc_prediction output")
    close(first, np.array([3.0, 2.0, 0.0]), atol=1e-12, label="first-visit values (A, B, unvisited)")
    close(every, np.array([2.5, 2.0, 0.0]), atol=1e-12, label="every-visit values (A, B, unvisited)")


@check("First-visit mc_prediction on the random walk lands within four standard errors of k / 6")
def test_random_walk(sub):
    f = has(sub, "mc_prediction")
    mdp = _random_walk()
    rng = np.random.default_rng(0)
    pi = np.ones((mdp.n_states, 1))
    episodes = [_episode(mdp, pi, rng) for _ in range(2000)]
    V = np.asarray(f(episodes, n_states=mdp.n_states, gamma=1.0, first_visit=True), dtype=float)
    truth = np.arange(1, 6) / 6
    # Each return is 0 or 1, so a first-visit average over n visits has
    # standard error sqrt(v (1 - v) / n).
    visits = np.zeros(mdp.n_states)
    for states, _, _ in episodes:
        visits[np.unique(states)] += 1
    se = np.sqrt(truth * (1 - truth) / visits[1:6])
    worst = np.max(np.abs(V[1:6] - truth) / se)
    if worst > 4.0:
        fail(
            f"expected values near {np.round(truth, 3)}, got {np.round(V[1:6], 3)}: "
            f"{worst:.1f} standard errors off at worst"
        )
    close(V[[0, 6]], np.zeros(2), atol=1e-12, label="the two terminal states")


@check("epsilon_soft puts 1 - epsilon + epsilon / A on the first best action and epsilon / A elsewhere")
def test_epsilon_soft(sub):
    f = has(sub, "epsilon_soft")
    Q = np.array([[0.5, 2.0, 1.0, -1.0], [3.0, 1.0, 3.0, 0.0]])
    pi = f(Q, 0.2)
    shape(pi, (2, 4), label="epsilon_soft output")
    # The second row ties actions 0 and 2; the first maximiser wins, as in rlenvs.greedy.
    want = np.array([[0.05, 0.85, 0.05, 0.05], [0.85, 0.05, 0.05, 0.05]])
    close(pi, want, atol=1e-12, label="epsilon_soft with epsilon = 0.2")
    close(f(Q, 0.0), rlenvs.greedy(Q), atol=1e-12, label="epsilon_soft with epsilon = 0")


@check("mc_control reaches the best epsilon-soft value on a small grid (median of five seeds)")
def test_mc_control(sub):
    f = has(sub, "mc_control")
    soft = has(sub, "epsilon_soft")
    grid = rlenvs.gridworld(["....", ".#..", "S..G"])
    start = grid.names["start"]
    epsilon = 0.2
    best = rlenvs.value_iteration(_soft_mdp(grid, epsilon))[start]
    got = []
    for seed in range(5):
        Q = np.asarray(f(grid, n_episodes=1000, epsilon=epsilon, alpha=0.1, seed=seed), dtype=float)
        shape(Q, (grid.n_states, grid.n_actions), label="mc_control output")
        v = rlenvs.policy_evaluation(grid, soft(Q, epsilon))
        got.append(v[start])
    median = float(np.median(got))
    # nan fails this comparison too, so a policy that never ends is caught.
    if not (median >= best - 0.5):
        fail(
            f"the best epsilon-soft policy is worth {best:.3f} from the start. Your "
            f"five runs gave {np.round(got, 3)}, median {median:.3f}, more than 0.5 below it."
        )


@check("importance_ratios returns the tail products rho_{t:T-1} of pi / b")
def test_importance_ratios(sub):
    f = has(sub, "importance_ratios")
    pi = np.array([[0.9, 0.1], [0.2, 0.8]])
    b = np.array([[0.5, 0.5], [0.5, 0.5]])
    states = np.array([0, 1, 0])
    actions = np.array([0, 1, 1])
    # Step ratios 1.8, 1.6, 0.2, so the tails are 0.576, 0.32, 0.2.
    got = f(states, actions, pi, b)
    shape(got, (3,), label="importance_ratios output")
    close(got, np.array([0.576, 0.32, 0.2]), atol=1e-12, label="rho_{t:T-1}")


@check("importance_sampling_estimates gives OIS 1.2 and WIS 0.947 on the three-episode example")
def test_is_worked_example(sub):
    f = has(sub, "importance_sampling_estimates")
    pi = np.array([[0.9, 0.1], [0.5, 0.5]])
    b = np.array([[0.5, 0.5], [0.5, 0.5]])
    L = (np.array([0]), np.array([0]), np.array([1.0]))
    R = (np.array([0]), np.array([1]), np.array([0.0]))
    ois, wis = f([L, R, L], pi, b, 1.0)
    almost(ois, 1.2, 1e-12, label="ordinary importance sampling")
    almost(wis, 3.6 / 3.8, 1e-12, label="weighted importance sampling")


@check("Over 2,000 batches OIS is unbiased for V = 0.9, and WIS has lower variance and stays in [0, 1]")
def test_is_statistics(sub):
    f = has(sub, "importance_sampling_estimates")
    mdp = _one_step_problem()
    pi = np.array([[0.9, 0.1], [0.5, 0.5]])
    b = np.array([[0.5, 0.5], [0.5, 0.5]])
    rng = np.random.default_rng(1)
    # Draw every behaviour action at once: 2,000 batches of 10 one-step episodes.
    acts = rng.integers(2, size=(2000, 10))
    ois, wis = [], []
    for row in acts:
        episodes = [(np.array([0]), np.array([a]), np.array([mdp.R[0, a]])) for a in row]
        o, w = f(episodes, pi, b, 1.0)
        ois.append(o)
        wis.append(w)
    ois, wis = np.array(ois), np.array(wis)
    se = ois.std(ddof=1) / np.sqrt(len(ois))
    if abs(ois.mean() - 0.9) > 3 * se:
        fail(f"the mean OIS estimate should be 0.9 within {3 * se:.4f}, got {ois.mean():.4f}")
    if not (wis.var() < ois.var()):
        fail(f"WIS variance {wis.var():.4f} should be below OIS variance {ois.var():.4f}")
    if wis.min() < -1e-12 or wis.max() > 1 + 1e-12:
        fail(f"every WIS estimate should lie in [0, 1], got a range {wis.min():.3f} to {wis.max():.3f}")
    if ois.max() <= 1.0:
        fail("some OIS estimates should exceed 1, the largest possible return")

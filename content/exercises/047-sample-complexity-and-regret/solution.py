"""Sample complexity and regret: the simulation lemma, UCB-VI, and step sizes."""

import numpy as np

import rlenvs


def simulation_lemma_bound(P_hat, P, R, gamma):
    """The simulation-lemma bound on |Q_hat^pi - Q^pi|, the same for every policy pi.

    With eps = max over (s, a) of the L1 distance ||P_hat[s, a] - P[s, a]||_1 and
    Rmax = max |R|, the bound is gamma / (1 - gamma)^2 * eps * Rmax.
    Returns a float.
    """
    eps = np.abs(P_hat - P).sum(axis=2).max()
    r_max = np.abs(R).max()
    return float(gamma / (1.0 - gamma) ** 2 * eps * r_max)


def policy_q(P, R, gamma, pi):
    """The exact action values Q^pi of policy pi (shape (S, A)) in the model (P, R).

    P has shape (S, A, S), R has shape (S, A). Solve for V^pi as a linear
    system, then return R + gamma * P V^pi, shape (S, A).
    """
    S = P.shape[0]
    P_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = (pi * R).sum(axis=1)
    v = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
    return R + gamma * P @ v


def simulation_gap(P_hat, P, R, gamma, pi):
    """Q^pi - Q_hat^pi from the simulation lemma, without solving for either Q.

    The lemma says Q^pi - Q_hat^pi = gamma (I - gamma P_hat^pi)^{-1} (P - P_hat) V^pi,
    where P_hat^pi is the (S*A, S*A) matrix whose row (s, a) and column (s2, a2) is
    P_hat[s, a, s2] * pi[s2, a2]. Returns an array of shape (S, A).
    """
    S, A = R.shape
    v = (pi * policy_q(P, R, gamma, pi)).sum(axis=1)
    P_hat_pi = np.einsum("sat,tb->satb", P_hat, pi).reshape(S * A, S * A)
    rhs = ((P - P_hat) @ v).reshape(S * A)
    gap = gamma * np.linalg.solve(np.eye(S * A) - gamma * P_hat_pi, rhs)
    return gap.reshape(S, A)


def hoeffding_bonus(N, horizon, delta):
    """The Hoeffding exploration bonus for every state-action pair, shape of N.

    bonus = horizon * sqrt(log(2 * N.size * horizon / delta) / (2 * max(N, 1))),
    where N holds the visit counts N(s, a). An untried pair counts as N = 1.
    """
    log_term = np.log(2.0 * N.size * horizon / delta)
    return horizon * np.sqrt(log_term / (2.0 * np.maximum(N, 1)))


def backward_induction(P, R, horizon, clip=False):
    """Optimal finite-horizon action values by dynamic programming from the last step.

    Returns Q of shape (horizon, S, A), where Q[h] is the value of acting at step h
    and then optimally for the remaining steps. Q[horizon - 1] = R. With clip=True,
    every Q[h] is capped at horizon - h, the most that rewards in [0, 1] can pay.
    """
    S, A = R.shape
    Q = np.zeros((horizon, S, A))
    v_next = np.zeros(S)
    for h in range(horizon - 1, -1, -1):
        Q[h] = R + P @ v_next
        if clip:
            Q[h] = np.minimum(Q[h], horizon - h)
        v_next = Q[h].max(axis=1)
    return Q


def finite_horizon_value(P, R, pi):
    """The value at step 0 of a step-dependent policy pi of shape (horizon, S, A).

    Returns V of shape (S,): the expected total reward over `horizon` steps from each
    start state, when step h uses pi[h].
    """
    horizon = pi.shape[0]
    v = np.zeros(R.shape[0])
    for h in range(horizon - 1, -1, -1):
        v = (pi[h] * (R + P @ v)).sum(axis=1)
    return v


def ucbvi(mdp, episodes, horizon, delta, seed, bonus_scale=1.0, epsilon=0.0):
    """Run UCB-VI for `episodes` episodes of `horizon` steps, with known rewards.

    Before each episode, build P_hat from the counts (an untried pair is a self-loop),
    run backward_induction(P_hat, R + bonus_scale * hoeffding_bonus(N, horizon, delta),
    horizon, clip=True), and act greedily, except that with probability `epsilon` a step
    takes a uniformly random action. bonus_scale = 0 and epsilon > 0 is the
    epsilon-greedy baseline on the same empirical model.

    Returns (Qs, N, regret):
      Qs      (episodes, horizon, S, A) the optimistic values used in each episode,
      N       (S, A) the final visit counts,
      regret  (episodes,) the exact gap mu0 @ (V*_0 - V^played_0) of each episode's
              policy, where the played policy mixes the greedy action with epsilon.
    """
    rng = np.random.default_rng(seed)
    S, A = mdp.R.shape
    counts = np.zeros((S, A, S))
    v_star = backward_induction(mdp.P, mdp.R, horizon).max(axis=2)[0]
    Qs = np.zeros((episodes, horizon, S, A))
    regret = np.zeros(episodes)
    for k in range(episodes):
        N = counts.sum(axis=2)
        P_hat = np.where(N[..., None] > 0, counts / np.maximum(N, 1)[..., None], np.eye(S)[:, None, :])
        bonus = bonus_scale * hoeffding_bonus(N, horizon, delta)
        Q = backward_induction(P_hat, mdp.R + bonus, horizon, clip=True)
        Qs[k] = Q
        pi = np.full((horizon, S, A), epsilon / A)
        best = Q.argmax(axis=2)
        for h in range(horizon):
            pi[h, np.arange(S), best[h]] += 1.0 - epsilon
        regret[k] = mdp.mu0 @ (v_star - finite_horizon_value(mdp.P, mdp.R, pi))
        s = int(rng.choice(S, p=mdp.mu0))
        for h in range(horizon):
            a = int(rng.choice(A, p=pi[h, s]))
            s2, _ = mdp.sample(s, a, rng)
            counts[s, a, s2] += 1
            s = s2
    return Qs, counts.sum(axis=2), regret


def step_size_sums(alphas):
    """Return (sum of alpha_n, sum of alpha_n squared) for a step-size sequence.

    Robbins-Monro needs the first to grow without bound and the second to stay finite.
    """
    alphas = np.asarray(alphas, dtype=float)
    return float(alphas.sum()), float((alphas**2).sum())


def estimator_variance(alphas, sigma2):
    """Exact variance of the running estimate after the last step in `alphas`.

    The estimate updates q <- q + alpha_n (x_n - q) on noisy samples x_n with variance
    sigma2. Each sample x_k reaches the end with weight w_k = alpha_k * prod over j > k of
    (1 - alpha_j), so the variance is sigma2 * sum of w_k squared. The starting value
    carries no variance of its own.
    """
    alphas = np.asarray(alphas, dtype=float)
    keep = np.cumprod((1.0 - alphas)[::-1])[::-1]
    after = np.append(keep[1:], 1.0)
    weights = alphas * after
    return float(sigma2 * (weights**2).sum())


if __name__ == "__main__":
    mdp = rlenvs.river_swim()
    horizon = 20
    for name, scale, eps in (("ucbvi", 0.05, 0.0), ("epsilon-greedy", 0.0, 0.1)):
        _, N, regret = ucbvi(mdp, 200, horizon, 0.1, seed=0, bonus_scale=scale, epsilon=eps)
        print(f"{name:15s} total regret {regret.sum():8.1f}   visits to state 5: {int(N[5].sum())}")
    n = np.arange(1, 1001)
    print("alpha = 1/n :", step_size_sums(1.0 / n))
    print("alpha = 0.1 :", step_size_sums(np.full(1000, 0.1)))

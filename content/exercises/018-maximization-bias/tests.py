"""Checks for the maximization bias and double Q-learning exercise."""

import math

import numpy as np

from rlcheck import almost, check, close, fail, has, shape


def _phi(x):
    return np.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def _Phi(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2)))


def _expected_max_std_normals(n):
    # E[max of n independent standard normals] = integral of x n phi(x) Phi(x)^(n-1).
    x = np.linspace(-10, 10, 40001)
    f = x * n * _phi(x) * _Phi(x) ** (n - 1)
    return float(np.sum((f[1:] + f[:-1]) / 2) * (x[1] - x[0]))


@check("single_estimate takes the max, and double_estimate reads the other table at the first argmax")
def test_estimates_by_hand(sub):
    single = has(sub, "single_estimate")
    double = has(sub, "double_estimate")
    a = np.array([[1.0, 3.0, 2.0], [0.0, -1.0, 5.0]])
    b = np.array([[0.5, 1.0, 4.0], [2.0, 2.0, -1.0]])
    got = single(a)
    shape(got, (2,), label="single_estimate output")
    close(got, np.array([3.0, 5.0]), atol=1e-12, label="single_estimate")
    close(double(a, b), np.array([1.0, -1.0]), atol=1e-12, label="double_estimate(a, b)")
    close(double(b, a), np.array([2.0, 0.0]), atol=1e-12, label="double_estimate(b, a), first argmax on ties")


@check("The single estimate is biased up by exactly E[max of normals]; the double estimate is not")
def test_bias(sub):
    single = has(sub, "single_estimate")
    double = has(sub, "double_estimate")
    rng = np.random.default_rng(0)

    # Two actions, true values 0, one unit-variance sample each.
    X = rng.standard_normal((200000, 2))
    Y = rng.standard_normal((200000, 2))
    almost(np.mean(single(X)), 1 / math.sqrt(math.pi), 0.01, label="single, 2 actions")
    almost(np.mean(double(X, Y)), 0.0, 0.01, label="double, 2 actions")

    # Ten actions, true values 0, each estimate a mean of 5 samples.
    X = rng.standard_normal((20000, 5, 10)).mean(axis=1)
    Y = rng.standard_normal((20000, 5, 10)).mean(axis=1)
    want = _expected_max_std_normals(10) / math.sqrt(5)
    almost(np.mean(single(X)), want, 0.01, label="single, 10 actions of 5 samples")
    almost(np.mean(double(X, Y)), 0.0, 0.01, label="double, 10 actions of 5 samples")

    # True values (0, 0.5): the double estimate falls below the true max, 0.5.
    mu = np.array([0.0, 0.5])
    X = mu + rng.standard_normal((200000, 2))
    Y = mu + rng.standard_normal((200000, 2))
    d = 0.5 / math.sqrt(2)
    want_single = 0.5 * float(_Phi(d)) + math.sqrt(2) * float(_phi(d))
    want_double = 0.5 * float(_Phi(d))
    almost(np.mean(single(X)), want_single, 0.01, label="single, true values (0, 0.5)")
    almost(np.mean(double(X, Y)), want_double, 0.01, label="double, true values (0, 0.5)")


@check("double_q_target picks the action with one table and scores it with the other")
def test_double_q_target(sub):
    f = has(sub, "double_q_target")
    q1 = np.array([1.0, 3.0])
    q2 = np.array([4.0, 0.0])
    # q1 picks action 1, and q2 says it is worth 0. Single Q-learning would use 3.
    almost(f(q1, q2, -1.0, False, 0.9), -1.0, 1e-12, label="select with q1, evaluate with q2")
    # q2 picks action 0, and q1 says it is worth 1.
    almost(f(q2, q1, -1.0, False, 0.9), -1.0 + 0.9, 1e-12, label="select with q2, evaluate with q1")
    almost(f(q1, q2, 2.0, True, 0.9), 2.0, 1e-12, label="a terminal step")


@check("Early on, Q-learning goes left from A far more often than double Q-learning (100 runs)")
def test_maximization_bias_early(sub):
    q_learning = has(sub, "q_learning")
    double_q = has(sub, "double_q_learning")
    env = sub.MaxBiasMDP() if hasattr(sub, "MaxBiasMDP") else None
    if env is None:
        fail("your module defines no `MaxBiasMDP`; keep the class from the starter")
    q = np.mean([q_learning(env, 30, 0.1, 0.1, seed) for seed in range(100)], axis=0)
    d = np.mean([double_q(env, 30, 0.1, 0.1, seed) for seed in range(100)], axis=0)
    shape(q, (30,), label="q_learning output, averaged over runs")
    q_mean, d_mean = float(q.mean()), float(d.mean())
    # Seeded replications give about 0.84 for Q-learning and 0.38 for double Q.
    if not (q_mean > 0.7 and d_mean < 0.5 and q_mean - d_mean > 0.3):
        fail(
            "over episodes 1 to 30, expected Q-learning to go left in about 0.84 of runs "
            f"and double Q-learning in about 0.38; got {q_mean:.3f} and {d_mean:.3f}"
        )


@check("Double Q-learning settles near the 0.05 floor of epsilon-greedy by episodes 181 to 200")
def test_double_q_late(sub):
    double_q = has(sub, "double_q_learning")
    env = sub.MaxBiasMDP()
    d = np.mean([double_q(env, 200, 0.1, 0.1, seed) for seed in range(100)], axis=0)
    late = float(d[-20:].mean())
    if not (late < 0.12):
        fail(f"expected double Q-learning's left fraction over episodes 181 to 200 near 0.08, below 0.12; got {late:.3f}")


@check("epsilon_greedy_probs gives epsilon / A to every action and splits 1 - epsilon among ties")
def test_epsilon_greedy_probs(sub):
    f = has(sub, "epsilon_greedy_probs")
    close(f(np.array([2.0, 5.0]), 0.2), np.array([0.1, 0.9]), atol=1e-12, label="Q = (2, 5), epsilon = 0.2")
    close(f(np.array([1.0, 1.0, 0.0]), 0.3), np.array([0.45, 0.45, 0.1]), atol=1e-12,
          label="Q = (1, 1, 0), epsilon = 0.3")


@check("expected_sarsa_target gives 3.7 on the worked example, and Q-learning's target at epsilon = 0")
def test_expected_sarsa_target(sub):
    f = has(sub, "expected_sarsa_target")
    q_next = np.array([2.0, 5.0])
    # -1 + 0.1 * 2 + 0.9 * 5 = 3.7, so a step of 0.5 from 0 lands on 1.85.
    almost(f(q_next, -1.0, False, 1.0, 0.2), 3.7, 1e-12, label="epsilon = 0.2")
    almost(f(q_next, -1.0, False, 1.0, 0.0), 4.0, 1e-12, label="epsilon = 0, the Q-learning target")
    almost(f(q_next, -1.0, False, 1.0, 1.0), 2.5, 1e-12, label="epsilon = 1, the uniform average")
    almost(f(q_next, -1.0, True, 1.0, 0.2), -1.0, 1e-12, label="a terminal step")

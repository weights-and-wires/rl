"""Checks for the exploration exercise."""

import numpy as np
import torch

import rlenvs
from rlcheck import almost, check, close, fail, has, shape

torch.set_num_threads(1)

SEEDS = (0, 1, 2)
BUDGET = 2000


def _river():
    mdp = rlenvs.river_swim()
    return mdp, float(mdp.mu0 @ rlenvs.value_iteration(mdp))


def _run(sub, method):
    explore = has(sub, "explore")
    mdp, j_star = _river()
    rows = []
    for seed in SEEDS:
        pi, counts, total = explore(mdp, method, steps=BUDGET, seed=seed, beta=0.5, epsilon=0.1,
                                    replan_every=20)
        shape(pi, (6, 2), label="the returned policy")
        shape(counts, (6, 2, 6), label="the returned transition counts")
        if not np.isclose(np.sum(counts), BUDGET):
            fail(f"the counts should add up to {BUDGET} steps, got {np.sum(counts):g}")
        j = float(mdp.mu0 @ rlenvs.policy_evaluation(mdp, pi))
        rows.append((seed, j, float(np.sum(counts[5])), float(total)))
    return rows, j_star


@check("count_bonus is beta / sqrt(N), with unvisited pairs treated as N = 1")
def test_count_bonus(sub):
    count_bonus = has(sub, "count_bonus")
    got = count_bonus(np.array([[0, 1], [4, 100]]), 0.5)
    shape(got, (2, 2), label="bonus")
    close(got, [[0.5, 0.5], [0.25, 0.05]], atol=1e-12, label="beta / sqrt(max(N, 1)) with beta = 0.5")


@check("pseudo_count recovers exact counts from an empirical density, and gives 4.4 for rho = 0.1, rho' = 0.12")
def test_pseudo_count(sub):
    pseudo_count = has(sub, "pseudo_count")
    almost(pseudo_count(0.1, 0.12), 4.4, 1e-9, label="rho = 0.1, rho' = 0.12")
    N = np.array([1.0, 3.0, 7.0, 20.0])
    n = 50.0
    close(pseudo_count(N / n, (N + 1) / (n + 1)), N, rtol=1e-9, label="pseudo-counts of an empirical model")


@check("empirical_model normalises counts, averages rewards, and sends unvisited pairs back to themselves")
def test_empirical_model(sub):
    empirical_model = has(sub, "empirical_model")
    counts = np.zeros((3, 2, 3))
    counts[0, 1] = [2, 6, 0]
    counts[2, 0] = [0, 1, 3]
    reward_sums = np.zeros((3, 2))
    reward_sums[0, 1] = 4.0
    reward_sums[2, 0] = 2.0
    P, R, n = empirical_model(counts, reward_sums)
    shape(P, (3, 2, 3), label="P")
    close(P.sum(axis=2), np.ones((3, 2)), atol=1e-12, label="rows of P")
    close(P[0, 1], [0.25, 0.75, 0.0], atol=1e-12, label="P[0, 1] from counts [2, 6, 0]")
    close(P[1, 0], [0.0, 1.0, 0.0], atol=1e-12, label="P[1, 0], never visited, a self-loop")
    close(R, [[0, 0.5], [0, 0], [0.5, 0]], atol=1e-12, label="mean rewards")
    close(n, [[0, 8], [0, 0], [4, 0]], atol=1e-12, label="visit counts N(s, a)")


@check("plan returns Q* on the true RiverSwim model")
def test_plan(sub):
    plan = has(sub, "plan")
    mdp, _ = _river()
    Q = plan(mdp.P, mdp.R, mdp.gamma)
    shape(Q, (6, 2), label="Q")
    want = rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp))
    close(Q, want, atol=1e-5, label="Q from value iteration")


@check("sample_posterior_mdp draws valid models whose average matches the Dirichlet and Gaussian posteriors")
def test_posterior(sub):
    sample = has(sub, "sample_posterior_mdp")
    counts = np.zeros((2, 1, 3))
    counts[0, 0] = [3, 0, 5]
    reward_sums = np.array([[6.0], [0.0]])
    rng = np.random.default_rng(0)
    Ps, Rs = [], []
    for _ in range(4000):
        P, R = sample(counts, reward_sums, rng, prior=1.0)
        Ps.append(P)
        Rs.append(R)
    Ps, Rs = np.array(Ps), np.array(Rs)
    shape(Ps[0], (2, 1, 3), label="a sampled P")
    shape(Rs[0], (2, 1), label="a sampled R")
    if np.any(Ps < 0) or not np.allclose(Ps.sum(axis=3), 1.0):
        fail("every sampled P[s, a] must be a probability vector")
    # Dirichlet(4, 1, 6) has mean [4, 1, 6] / 11; Dirichlet(1, 1, 1) has mean 1/3 each.
    close(Ps.mean(axis=0)[0, 0], np.array([4, 1, 6]) / 11, atol=0.01, label="posterior mean of P[0, 0]")
    close(Ps.mean(axis=0)[1, 0], np.full(3, 1 / 3), atol=0.01, label="posterior mean of the unvisited P[1, 0]")
    # Reward posterior: mean sum / (N + 1), standard deviation 1 / sqrt(N + 1).
    almost(Rs[:, 0, 0].mean(), 6.0 / 9.0, 0.02, label="posterior mean reward after 8 visits")
    almost(Rs[:, 0, 0].std(), 1.0 / 3.0, 0.02, label="posterior reward std after 8 visits")
    almost(Rs[:, 1, 0].std(), 1.0, 0.04, label="prior reward std for an unvisited pair")


@check("Epsilon-greedy never reaches the far end of RiverSwim in 2000 steps, on three seeds")
def test_explore_epsilon(sub):
    rows, j_star = _run(sub, "epsilon")
    for seed, j, far, total in rows:
        if far > 0 or j > 0.5:
            fail(f"seed {seed}: expected no visits to state 5 and a final policy worth about 0.1, "
                 f"got {far:g} visits and value {j:.3f} (optimal {j_star:.3f})")


@check("Count-based optimism finds the far reward and ends with the optimal policy, on three seeds")
def test_explore_optimism(sub):
    rows, j_star = _run(sub, "optimism")
    for seed, j, far, total in rows:
        if not np.isclose(j, j_star, atol=1e-6) or far < 500:
            fail(f"seed {seed}: expected the optimal policy (value {j_star:.3f}) and over 500 visits to "
                 f"state 5, got value {j:.3f} and {far:g} visits")


@check("Posterior sampling finds the far reward and ends with the optimal policy, on three seeds")
def test_explore_posterior(sub):
    rows, j_star = _run(sub, "posterior")
    for seed, j, far, total in rows:
        if not np.isclose(j, j_star, atol=1e-6) or far < 500:
            fail(f"seed {seed}: expected the optimal policy (value {j_star:.3f}) and over 500 visits to "
                 f"state 5, got value {j:.3f} and {far:g} visits")


@check("rnd_bonus is the squared distance between predictor and target outputs, per state")
def test_rnd_bonus(sub):
    rnd_bonus = has(sub, "rnd_bonus")
    target = torch.nn.Linear(2, 2, bias=False)
    predictor = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        target.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        predictor.weight.copy_(torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    states = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    got = rnd_bonus(target, predictor, states)
    # Differences: [[2, -2], [0, 0]], so squared distances [8, 0].
    close(torch.as_tensor(got).detach().numpy(), [8.0, 0.0], atol=1e-6, label="bonus per state")


@check("rnd_train drives the bonus down on visited states and leaves novel states over 10 times higher")
def test_rnd_train(sub):
    make_rnd = has(sub, "make_rnd")
    rnd_bonus = has(sub, "rnd_bonus")
    rnd_train = has(sub, "rnd_train")
    target, predictor = make_rnd(2, 8, 32, seed=0)
    gen = torch.Generator().manual_seed(0)
    seen = torch.rand(256, 2, generator=gen) * 0.5
    novel = torch.rand(256, 2, generator=gen) * 0.5 + 0.5
    before = float(rnd_bonus(target, predictor, seen).mean())
    rnd_train(target, predictor, seen, steps=300, lr=1e-2)
    after = float(rnd_bonus(target, predictor, seen).mean())
    fresh = float(rnd_bonus(target, predictor, novel).mean())
    if after > 0.01 * before:
        fail(f"training should cut the bonus on visited states by over 99%: {before:.4g} to {after:.4g}")
    if fresh < 10 * after:
        fail(f"novel states should keep a bonus over 10 times the visited ones: {fresh:.4g} against {after:.4g}")
    target2, _ = make_rnd(2, 8, 32, seed=0)
    close(target2[0].weight.detach().numpy(), target[0].weight.detach().numpy(), atol=0,
          label="the target network's first layer, which training must not change")

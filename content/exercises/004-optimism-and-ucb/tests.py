"""Checks for the optimism and UCB exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, fail, has, shape

MEANS = [0.1, 0.5, 0.9]
GAPS = np.array([0.8, 0.4, 0.0])


def _mean_curve(run_bandit, method, T, seeds, **kwargs):
    curves = []
    for s in seeds:
        curve, counts = run_bandit(rlenvs.BernoulliBandit(MEANS, seed=s), method, T, seed=s, **kwargs)
        curve = np.asarray(curve, dtype=float)
        shape(curve, (T,), label=f"run_bandit(..., {method!r}, T={T})")
        curves.append(curve)
    return np.mean(curves, axis=0)


@check("ucb_bonus is sqrt(2 ln t / N), infinite for an unpulled arm")
def test_ucb_bonus(sub):
    ucb_bonus = has(sub, "ucb_bonus")
    got = np.asarray(ucb_bonus(np.array([5, 2]), 7), dtype=float)
    shape(got, (2,), label="ucb_bonus output")
    close(got, [np.sqrt(2 * np.log(7) / 5), np.sqrt(2 * np.log(7) / 2)], atol=1e-12, label="bonuses for counts (5, 2) at t = 7")
    close(got, [0.882249, 1.394959], atol=1e-6, label="the chapter's table")
    got = np.asarray(ucb_bonus(np.array([4, 0, 3]), 7), dtype=float)
    if not np.isposinf(got[1]):
        fail(f"an arm with count 0 must have bonus +inf, got {got[1]}")
    if not (got[0] < got[2] < np.inf):
        fail("the bonus must shrink as the count grows: an arm pulled 4 times should have a smaller bonus than one pulled 3 times")


@check("the bonus grows with t, so a neglected arm is eventually tried again")
def test_bonus_grows(sub):
    ucb_bonus = has(sub, "ucb_bonus")
    early = float(ucb_bonus(np.array([10]), 20)[0])
    late = float(ucb_bonus(np.array([10]), 20000)[0])
    close(early, np.sqrt(2 * np.log(20) / 10), atol=1e-12, label="bonus at t = 20")
    if late <= early:
        fail(f"with the count fixed at 10, the bonus at t=20000 ({late:.3f}) must exceed the bonus at t=20 ({early:.3f}): use ln t, the total pulls")


@check("ucb1_scores matches the chapter's table")
def test_ucb1_scores(sub):
    ucb1_scores = has(sub, "ucb1_scores")
    got = ucb1_scores(np.array([0.6, 0.5]), np.array([5, 2]), 7)
    close(got, [0.6 + np.sqrt(2 * np.log(7) / 5), 0.5 + np.sqrt(2 * np.log(7) / 2)], atol=1e-12,
          label="scores for means (0.6, 0.5), counts (5, 2), t = 7")
    close(got, [1.482249, 1.894959], atol=1e-6, label="the chapter's table")
    got = np.asarray(ucb1_scores(np.array([0.3, 0.0, 0.8]), np.array([4, 0, 3]), 7), dtype=float)
    if not np.isposinf(got[1]):
        fail(f"an arm with count 0 must score +inf, got {got[1]}")


@check("ucb1_action picks the less-pulled arm in the table, tries unpulled arms first, and splits ties")
def test_ucb1_action(sub):
    ucb1_action = has(sub, "ucb1_action")
    rng = np.random.default_rng(0)
    pick = ucb1_action(np.array([0.6, 0.5]), np.array([5, 2]), 7, rng)
    if pick != 1:
        fail(f"arm 1 has the higher score (1.895 against 1.482), so UCB1 should pull it; got arm {pick}")
    pick = ucb1_action(np.array([0.6, 1.0 / 3.0]), np.array([5, 3]), 8, rng)
    if pick != 0:
        fail(f"after arm 1 pays 0 on its third pull, arm 0 scores 1.512 and arm 1 scores 1.511 at t = 8, so UCB1 should pull arm 0; got arm {pick}")
    picks = [ucb1_action(np.array([0.9, 0.0, 0.0]), np.array([5, 0, 0]), 5, rng) for _ in range(2000)]
    if set(picks) != {1, 2}:
        fail(f"arms 1 and 2 were never pulled and both score +inf, so UCB1 should choose between exactly them; got {sorted(set(picks))}")
    share = np.mean(np.array(picks) == 1)
    if abs(share - 0.5) > 4 * np.sqrt(0.25 / 2000):
        fail(f"two unpulled arms tie, so each should be picked about half the time; arm 1 was picked {share:.3f} of the time")


@check("ucb1_regret_bound gives 233 for the row 0.1, 0.5, 0.9 at T = 2000")
def test_regret_bound(sub):
    ucb1_regret_bound = has(sub, "ucb1_regret_bound")
    want = 8 * np.log(2000) * (1 / 0.8 + 1 / 0.4) + (1 + np.pi**2 / 3) * 1.2
    close(ucb1_regret_bound(GAPS, 2000), want, atol=1e-9, label="8 sum ln T / gap + (1 + pi^2/3) sum gap")
    close(ucb1_regret_bound(GAPS, 2000), 233.2, atol=0.1, label="the chapter's number")
    one = ucb1_regret_bound(np.array([0.5, 0.0]), 1000)
    close(one, 8 * np.log(1000) / 0.5 + (1 + np.pi**2 / 3) * 0.5, atol=1e-9, label="two arms with one gap of 0.5")


@check("UCB1 pulls every arm once before it repeats any, and mostly pulls the best arm")
def test_pulls_every_arm(sub):
    run_bandit = has(sub, "run_bandit")
    _, counts = run_bandit(rlenvs.BernoulliBandit(MEANS, seed=0), "ucb1", 3, seed=0)
    if not np.array_equal(np.asarray(counts), [1, 1, 1]):
        fail(f"after 3 pulls UCB1 should have tried each of the 3 arms once, but the counts were {list(counts)}")
    _, counts = run_bandit(rlenvs.BernoulliBandit(MEANS, seed=0), "ucb1", 2000, seed=0)
    if counts[2] < 0.8 * 2000:
        fail(f"after 2000 pulls the best arm should have most of them, but the counts were {list(counts)}")


@check("UCB1 has far less regret than epsilon-greedy at T = 2000, and below its own bound")
def test_regret_comparison(sub):
    run_bandit = has(sub, "run_bandit")
    ucb_bound = has(sub, "ucb1_regret_bound")
    seeds = range(4)
    eps = _mean_curve(run_bandit, "epsilon-greedy", 2000, seeds, epsilon=0.1)[-1]
    ucb = _mean_curve(run_bandit, "ucb1", 2000, seeds)[-1]
    bound = ucb_bound(GAPS, 2000)
    print(f"mean regret at T=2000: epsilon-greedy {eps:.1f}, UCB1 {ucb:.1f}, UCB1 bound {bound:.1f}")
    if ucb > 0.75 * eps:
        fail(f"UCB1's mean regret {ucb:.1f} should be well under epsilon-greedy's {eps:.1f} (below 75 percent of it)")
    if ucb > bound:
        fail(f"UCB1's mean regret {ucb:.1f} should sit below the bound {bound:.1f}")


@check("UCB1's regret grows sublinearly from T=2000 to T=8000, epsilon-greedy's almost linearly")
def test_growth(sub):
    run_bandit = has(sub, "run_bandit")
    seeds = range(4)
    report = []
    for method, kwargs in [("ucb1", {}), ("epsilon-greedy", {"epsilon": 0.1})]:
        curve = _mean_curve(run_bandit, method, 8000, seeds, **kwargs)
        ratio = curve[-1] / max(curve[1999], 1e-12)
        report.append(f"{method} {ratio:.2f}")
        if method == "ucb1" and ratio > 2.2:
            fail(f"UCB1's regret grew {ratio:.2f} times from T=2000 to T=8000. Logarithmic growth gives well under 2.2")
        if method == "epsilon-greedy" and ratio < 2.8:
            fail(f"epsilon-greedy's regret grew only {ratio:.2f} times from T=2000 to T=8000. A fixed epsilon should grow close to 4 times")
    print("regret(8000) / regret(2000): " + ", ".join(report))

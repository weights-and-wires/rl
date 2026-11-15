"""Checks for the rubric-rewards exercise.

Every call into the reader's module sits inside a check. A missing
implementation fails that check with the expected value and the actual
exception, and the rest of the suite still runs.
"""

import itertools

import numpy as np

from rlcheck import almost, check, close, equal, fail, finite_difference, has

# The hand example from the lesson: four claims, cost 5 per false claim.
W = np.array([3.0, 2.0, 2.0, 1.0])
P = np.array([0.9, 0.7, 0.5, 0.2])
C = 5.0


def call(fn, *args, expected, **kwargs):
    """Call reader code. NotImplementedError fails the check in place."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        fail(f"not implemented yet: expected {expected}, actual NotImplementedError")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@check("rubric_reward adds the weight of every included claim")
def test_rubric_reward(sub):
    rubric_reward = has(sub, "rubric_reward")
    got = call(rubric_reward, np.array([1, 1, 0, 1]), W, expected=6.0)
    almost(got, 6.0, 1e-12, label="include claims 1, 2, 4: 3 + 2 + 1")
    got = call(rubric_reward, np.zeros(4), W, expected=0.0)
    almost(got, 0.0, 1e-12, label="include nothing")
    got = call(rubric_reward, np.ones(4), W, expected=8.0)
    almost(got, 8.0, 1e-12, label="brain dump: include everything")


@check("claims_reward charges c for each included false claim and nothing else")
def test_claims_reward(sub):
    claims_reward = has(sub, "claims_reward")
    truth = np.array([1, 0, 1, 0])
    got = call(claims_reward, np.ones(4), truth, C, expected=-10.0)
    almost(got, -10.0, 1e-12, label="all four included, two false, c = 5")
    got = call(claims_reward, np.array([1, 0, 1, 0]), truth, C, expected=0.0)
    almost(got, 0.0, 1e-12, label="only the true claims included")
    got = call(claims_reward, np.zeros(4), truth, C, expected=0.0)
    almost(got, 0.0, 1e-12, label="say nothing")


@check("gated_reward is 0 whenever the test fails and the rubric score when it passes")
def test_gated_reward(sub):
    gated_reward = has(sub, "gated_reward")
    got = call(gated_reward, False, 0.75, expected=0.0)
    almost(got, 0.0, 1e-12, label="failed test, rubric 0.75")
    got = call(gated_reward, True, 0.75, expected=0.75)
    almost(got, 0.75, 1e-12, label="passed test, rubric 0.75")
    got = call(gated_reward, False, 1e6, expected=0.0)
    almost(got, 0.0, 1e-12, label="failed test, huge rubric score")


@check("optimal_subset matches the closed-form threshold on the hand example")
def test_optimal_subset_hand(sub):
    optimal_subset = has(sub, "optimal_subset")
    cases = [
        # alpha, beta, expected mask, explanation
        (1.0, 1.0, [1, 1, 0, 0], "alpha*w > beta*c*(1-p): 3>0.5, 2>1.5, 2<2.5, 1<4"),
        (2.0, 1.0, [1, 1, 1, 0], "doubling alpha: 6>0.5, 4>1.5, 4>2.5, 2<4"),
        (0.0, 1.0, [0, 0, 0, 0], "claims grader alone: say nothing"),
        (1.0, 0.0, [1, 1, 1, 1], "rubric grader alone: brain dump"),
    ]
    for alpha, beta, want, why in cases:
        got = call(optimal_subset, W, P, C, alpha, beta, expected=want)
        equal(np.asarray(got).astype(int), np.array(want), label=f"alpha={alpha}, beta={beta} ({why})")


@check("a claim with expected net value exactly 0 stays out")
def test_optimal_subset_tie(sub):
    optimal_subset = has(sub, "optimal_subset")
    # w = 2 and c*(1-p) = 4*0.5 = 2: a tie, which is not worth including.
    got = call(optimal_subset, np.array([2.0]), np.array([0.5]), 4.0, 1.0, 1.0, expected=[False])
    equal(np.asarray(got).astype(int), np.array([0]), label="tie goes to leaving the claim out")
    got = call(optimal_subset, np.array([2.0]), np.array([0.6]), 4.0, 1.0, 1.0, expected=[True])
    equal(np.asarray(got).astype(int), np.array([1]), label="p = 0.6 is above the threshold")


@check("expected_reward equals the average over all 2^K truth worlds")
def test_expected_reward_enumeration(sub):
    expected_reward = has(sub, "expected_reward")
    # Reference by enumeration, using the grader formulas written out here.
    def enumerate_world(include, w, p, c, alpha, beta):
        total = 0.0
        for truth in itertools.product([0, 1], repeat=len(w)):
            truth = np.array(truth)
            prob = np.prod(np.where(truth == 1, p, 1 - p))
            r = np.sum(include * w)
            pen = -c * np.sum(include * (1 - truth))
            total += prob * (alpha * r + beta * pen)
        return total

    for include, alpha, beta in [
        (np.array([1.0, 1.0, 1.0, 1.0]), 1.0, 1.0),
        (np.array([1.0, 1.0, 0.0, 0.0]), 1.0, 1.0),
        (np.array([1.0, 0.0, 1.0, 1.0]), 2.0, 0.5),
        (np.array([0.3, 0.9, 0.5, 0.1]), 1.5, 1.0),
    ]:
        want = enumerate_world(include, W, P, C, alpha, beta)
        got = call(expected_reward, include, W, P, C, alpha, beta, expected=want)
        almost(got, want, 1e-9, label=f"include={include}, alpha={alpha}, beta={beta}")
    # The lesson's numbers: the optimal subset is worth 3.0 and the dump -0.5.
    almost(call(expected_reward, np.array([1.0, 1.0, 0.0, 0.0]), W, P, C, 1.0, 1.0, expected=3.0), 3.0, 1e-9, label="optimal subset")
    almost(call(expected_reward, np.ones(4), W, P, C, 1.0, 1.0, expected=-0.5), -0.5, 1e-9, label="brain dump")


@check("brute force over all 2^K subsets picks the subset optimal_subset returns")
def test_optimal_subset_brute_force(sub):
    optimal_subset = has(sub, "optimal_subset")
    expected_reward = has(sub, "expected_reward")
    rng = np.random.default_rng(0)
    for trial in range(6):
        k = 8
        w = rng.uniform(0.2, 3.0, size=k)
        p = rng.uniform(0.05, 0.99, size=k)
        c = float(rng.uniform(1.0, 6.0))
        alpha = float(rng.uniform(0.3, 2.0))
        beta = float(rng.uniform(0.3, 2.0))
        best_mask, best_value = None, -np.inf
        for bits in itertools.product([0.0, 1.0], repeat=k):
            mask = np.array(bits)
            value = call(expected_reward, mask, w, p, c, alpha, beta, expected="a float")
            if value > best_value:
                best_mask, best_value = mask, value
        got = call(optimal_subset, w, p, c, alpha, beta, expected=best_mask)
        equal(np.asarray(got).astype(int), best_mask.astype(int), label=f"trial {trial}: brute-force best of 256 subsets")


@check("policy_gradient matches finite differences of the expected reward")
def test_policy_gradient(sub):
    policy_gradient = has(sub, "policy_gradient")
    expected_reward = has(sub, "expected_reward")
    rng = np.random.default_rng(1)
    for alpha, beta in [(1.0, 1.0), (2.0, 0.5), (1.0, 0.0)]:
        theta = rng.normal(size=4)

        def objective(t):
            return call(expected_reward, sigmoid(t), W, P, C, alpha, beta, expected="a float")

        want = finite_difference(objective, theta)
        got = call(policy_gradient, theta, W, P, C, alpha, beta, expected=want)
        close(got, want, rtol=1e-5, atol=1e-7, label=f"gradient at alpha={alpha}, beta={beta}")


@check("training converges to the threshold policy on the hand example")
def test_train_converges(sub):
    train_policy = has(sub, "train_policy")
    optimal_subset = has(sub, "optimal_subset")
    theta = call(train_policy, W, P, C, 1.0, 1.0, expected="logits of shape (4,)")
    q = sigmoid(np.asarray(theta))
    keep = np.asarray(call(optimal_subset, W, P, C, 1.0, 1.0, expected=[1, 1, 0, 0]))
    if not np.all(q[keep] > 0.9):
        fail(f"claims worth including should have probability above 0.9, got {np.round(q, 3)}")
    if not np.all(q[~keep] < 0.1):
        fail(f"claims not worth including should have probability below 0.1, got {np.round(q, 3)}")
    # The caller's starting logits must stay untouched.
    start = np.zeros(4)
    call(train_policy, W, P, C, 1.0, 1.0, steps=10, theta0=start, expected="logits")
    equal(start, np.zeros(4), label="theta0 after training")


@check("with beta = 0 training adds claims: the expected claim count rises toward K")
def test_brain_dump(sub):
    train_policy = has(sub, "train_policy")
    start = float(np.sum(sigmoid(np.zeros(4))))
    mid = call(train_policy, W, P, C, 1.0, 0.0, steps=20, expected="logits")
    end = call(train_policy, W, P, C, 1.0, 0.0, steps=600, expected="logits")
    count_mid = float(np.sum(sigmoid(np.asarray(mid))))
    count_end = float(np.sum(sigmoid(np.asarray(end))))
    if not (start < count_mid < count_end):
        fail(f"expected claim count should rise from {start:.2f}, got {count_mid:.2f} then {count_end:.2f}")
    if count_end < 3.9:
        fail(f"a rubric-only policy should include nearly all 4 claims, got {count_end:.2f}")
    # Mixing the claims grader back in pulls the count down.
    mixed = call(train_policy, W, P, C, 1.0, 1.0, expected="logits")
    count_mixed = float(np.sum(sigmoid(np.asarray(mixed))))
    if count_mixed > 2.3:
        fail(f"the mix should keep about 2 claims, got {count_mixed:.2f}")

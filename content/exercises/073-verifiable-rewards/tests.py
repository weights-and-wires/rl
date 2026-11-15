"""Checks for the verifiable-rewards exercise.

Every call into the reader's module sits inside a check. A missing
implementation fails that check with the expected value and the actual
exception, and the rest of the suite still runs.
"""

import time

import numpy as np

from rlcheck import almost, check, equal, fail, has


def call(fn, *args, expected, **kwargs):
    """Call reader code. NotImplementedError fails the check in place."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        fail(f"not implemented yet: expected {expected}, actual NotImplementedError")


def literal_normalizer(text):
    """Wrong implementation the equivalence check is written to reject.

    It strips a boxed span and whitespace and then compares the strings.
    It treats ``1/2`` and ``0.5`` as different, so ``test_is_equivalent``
    fails a reader who stops at this normalizer.
    """
    key = r"\boxed{"
    start = text.rfind(key)
    if start >= 0:
        text = text[start + len(key) :]
        end = text.find("}")
        if end >= 0:
            text = text[:end]
    return "".join(text.split())


def keep_tied_groups(rewards, group_size):
    """Wrong implementation the filter check is written to reject.

    It keeps every group, including one whose rewards are all equal.
    ``test_dynamic_sampling_filter`` fails it: the effective batch size is
    then the full length, not the number of rewards in mixed groups.
    """
    del group_size
    return np.ones(len(rewards), dtype=bool)


# Thirty pairs. The false rows are the ones a string compare gets wrong, or
# the ones a normalizer that ignores units and exact fractions gets wrong.
PAIRS = (
    ("1/2", "0.5", True),
    ("1/2", r"\frac{1}{2}", True),
    ("1.0", "1", True),
    ("2.50", "2.5", True),
    ("2.50", "5/2", True),
    ("5 meters", "5", True),
    ("5m", "5 meters", True),
    (r"\boxed{1/2}", "0.5", True),
    ("1,000", "1000", True),
    ("-1/2", "-0.5", True),
    ("2/4", "0.5", True),
    ("0.500", "1/2", True),
    ("3/2", "1.5", True),
    ("1/3", "0.333", False),
    ("5 meters", "5 kg", False),
    ("10 kg", "10 kilograms", True),
    ("2", "2.00", True),
    (r"\boxed{2.0}", "2", True),
    ("1/2", "1/3", False),
    ("0.25", "1/4", True),
    ("$0.5$", "1/2", True),
    ("50%", "50 percent", True),
    ("50%", "0.5", False),
    ("3 seconds", "3 s", True),
    ("3 seconds", "3", True),
    ("-0.25", "-1/4", True),
    (r"\boxed{\frac{2}{4}}", "0.5", True),
    ("1,000.0", "1000", True),
    ("6/3", "2.0", True),
    ("0.5 meters", "1/2 m", True),
)

CODE_TESTS = (
    {"stdin": "2 3\n", "stdout": "5\n"},
    {"stdin": "10 1\n", "stdout": "11\n"},
)
ADDER = "a, b = map(int, input().split())\nprint(a + b)\n"
MEMORIZED = "print(5)\n"
INFINITE = "while True:\n    pass\n"


@check("normalize_math_answer reduces fractions, decimals, a box, and a unit")
def test_normalize_math_answer(sub):
    normalize_math_answer = has(sub, "normalize_math_answer")
    cases = {
        "1/2": "1/2",
        "0.5": "1/2",
        r"\frac{1}{2}": "1/2",
        "2.50": "5/2",
        "1.0": "1",
        "5 meters": "5 meter",
        r"\boxed{1,000}": "1000",
        "-0.25": "-1/4",
        r"\boxed{2.50 meters}": "5/2 meter",
    }
    for text, want in cases.items():
        got = call(normalize_math_answer, text, expected=want)
        if got != want:
            fail(f"normalize_math_answer({text!r}): got {got!r}, expected {want!r}")


@check("is_equivalent agrees on 30 pairs, including 1/2 versus 0.5")
def test_is_equivalent(sub):
    is_equivalent = has(sub, "is_equivalent")
    if len(PAIRS) != 30:
        fail(f"the pair table has {len(PAIRS)} rows, expected 30")
    # literal_normalizer leaves "1/2" and "0.5" as different strings.
    wrong = literal_normalizer("1/2") == literal_normalizer("0.5")
    if wrong:
        fail(
            "literal_normalizer('1/2') compared with literal_normalizer('0.5'): "
            "got equal, expected not equal"
        )
    for left, right, want in PAIRS:
        got = call(is_equivalent, left, right, expected=want)
        if got is not want:
            fail(
                f"is_equivalent({left!r}, {right!r}): got {got}, expected {want}. "
                "literal_normalizer treats '1/2' and '0.5' as different strings"
            )


@check("format_correctness_reward pays 1.1, 0.1, or 0")
def test_format_correctness_reward(sub):
    reward = has(sub, "format_correctness_reward")
    cases = (
        (r"\boxed{1/2}", "0.5", 1.1),
        (r"\boxed{2/4}", "0.5", 1.1),
        (r"\boxed{2.50 meters}", "5/2", 1.1),
        (r"\boxed{9}", "1", 0.1),
        ("0.5", "0.5", 0.0),
        ("the answer is 0.5", "0.5", 0.0),
    )
    for response, target, want in cases:
        got = call(reward, response, target, expected=want)
        almost(got, want, 1e-12, label=f"reward of {response!r} against {target!r}")


@check("a large format bonus gives a wrong boxed answer a positive advantage")
def test_format_credit_advantage(sub):
    reward = has(sub, "format_correctness_reward")
    # One correct box, five wrong boxes, two responses with no box.
    responses = [r"\boxed{1}", *([r"\boxed{9}"] * 5), "1", "no box"]
    for credit, want in ((0.1, -0.1), (0.9, 0.1)):
        scores = [
            call(reward, text, "1", format_credit=credit, expected=f"a reward at format_credit={credit}")
            for text in responses
        ]
        mean = sum(scores) / len(scores)
        got = scores[1] - mean
        almost(
            got,
            want,
            1e-9,
            label=f"centered advantage of a wrong box at format_credit={credit}",
        )


@check("code_reward gives a program that adds its inputs full credit")
def test_code_reward_adds(sub):
    code_reward = has(sub, "code_reward")
    got = call(code_reward, ADDER, CODE_TESTS, 0.5, expected=1.0)
    almost(got, 1.0, 1e-12, label="reward of the adding program")


@check("code_reward gives an infinite loop no credit before the timeout")
def test_code_reward_timeout(sub):
    code_reward = has(sub, "code_reward")
    started = time.perf_counter()
    got = call(code_reward, INFINITE, CODE_TESTS, 0.4, expected=0.0)
    elapsed = time.perf_counter() - started
    almost(got, 0.0, 1e-12, label="reward of an infinite loop")
    if elapsed > 8.0:
        fail(
            f"infinite loop returned in {elapsed:.2f}s, expected under 8s "
            "with a per-test timeout of 0.4s"
        )


@check("code_reward does not give full credit for printing one memorized line")
def test_code_reward_memorized(sub):
    code_reward = has(sub, "code_reward")
    got = call(code_reward, MEMORIZED, CODE_TESTS, 0.5, expected=0.5)
    almost(got, 0.5, 1e-12, label="reward of print(5) on two different tests")
    if not got < 1.0:
        fail(f"memorized program: got {got}, expected a reward below 1")


@check("wilson_interval matches the score formula, including zero successes")
def test_wilson_interval(sub):
    wilson_interval = has(sub, "wilson_interval")
    # 3/8 at z=1.96, 0/8 at z=1.96, and 3/8 at z=1 so a hardcoded 1.96 fails.
    cases = (
        (3, 8, 1.96, 0.136841759462, 0.694262145416),
        (0, 8, 1.96, 0.0, 0.324415619511),
        (3, 8, 1.0, 0.226918002921, 0.550859774857),
    )
    for successes, n, z, low, high in cases:
        got = call(
            wilson_interval,
            successes,
            n,
            z=z,
            expected=f"({low}, {high})",
        )
        if len(got) != 2:
            fail(f"wilson_interval({successes}, {n}, z={z}): got {got}, expected a pair")
        almost(got[0], low, 1e-6, label=f"low end, k={successes}, n={n}, z={z}")
        almost(got[1], high, 1e-6, label=f"high end, k={successes}, n={n}, z={z}")


@check("dynamic_sampling_filter drops all-equal groups and counts the batch")
def test_dynamic_sampling_filter(sub):
    dynamic_sampling_filter = has(sub, "dynamic_sampling_filter")
    rewards = np.array(
        [
            1, 0, 1, 0,
            0, 0, 0, 0,
            1, 1, 1, 1,
            0.1, 0.1, 0.1, 0.1,
            1, 0, 0, 1,
        ],
        dtype=float,
    )
    want = np.array(
        [True] * 4 + [False] * 12 + [True] * 4,
        dtype=bool,
    )
    got = call(
        dynamic_sampling_filter,
        rewards,
        4,
        expected="a mask of length 20 with 8 True entries",
    )
    got = np.asarray(got)
    if got.shape != want.shape:
        fail(f"mask shape got {got.shape}, expected {want.shape}")
    got_bool = got.astype(bool)
    if not np.array_equal(got_bool, want):
        fail(
            "mask mismatch. got "
            f"{got_bool.astype(int).tolist()}, expected {want.astype(int).tolist()}. "
            "keep_tied_groups would keep the all-zero group at indexes 4 to 7"
        )
    got_n = int(got_bool.sum())
    wrong_n = int(np.asarray(keep_tied_groups(rewards, 4)).sum())
    if got_n != 8:
        fail(
            f"effective batch size got {got_n}, expected 8. "
            f"keep_tied_groups reports {wrong_n}"
        )
    equal(got_bool[4:8], np.zeros(4, dtype=bool), label="all-zero group")

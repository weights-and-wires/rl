"""Verifiable rewards: normalize an answer, score it, and drop empty groups.

The checks call the six functions below. A math answer becomes a reduced
fraction plus an optional unit. A code answer is the fraction of unit tests
that pass inside a subprocess. A prompt group stays in the batch only when
its rewards are not all equal.
"""

import math
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np


# Longer aliases come first so "meters" is not read as a trailing "s".
_UNITS = (
    ("kilograms", "kilogram"),
    ("kilogram", "kilogram"),
    ("centimeters", "centimeter"),
    ("centimeter", "centimeter"),
    ("millimeters", "millimeter"),
    ("millimeter", "millimeter"),
    ("seconds", "second"),
    ("second", "second"),
    ("percent", "percent"),
    ("meters", "meter"),
    ("meter", "meter"),
    ("grams", "gram"),
    ("gram", "gram"),
    ("kg", "kilogram"),
    ("cm", "centimeter"),
    ("mm", "millimeter"),
    ("m", "meter"),
    ("g", "gram"),
    ("s", "second"),
    ("%", "percent"),
)
_CANONICAL_UNITS = {canon for _, canon in _UNITS}

_NUMBER = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?(?:\d+(?:\.\d*)?|\.\d+))?"
)


def _last_boxed(text):
    """Return the inside of the last \\boxed{...}, or the whole string."""
    key = r"\boxed{"
    start = text.rfind(key)
    if start < 0:
        return text
    i = start + len(key)
    depth = 1
    chars = []
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth:
            chars.append(ch)
        i += 1
    if depth != 0:
        return text
    return "".join(chars)


def _expand_frac(text):
    pattern = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
    expanded = text
    while True:
        nxt = pattern.sub(r"(\1)/(\2)", expanded)
        if nxt == expanded:
            return expanded
        expanded = nxt


def _cleanup(text):
    span = _last_boxed(text)
    span = span.replace("$", "")
    span = span.replace(r"\left", "").replace(r"\right", "")
    span = span.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "")
    span = span.replace(r"\ ", " ")
    span = span.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    span = _expand_frac(span)
    for command in ("text", "mathrm", "operatorname"):
        span = re.sub(r"\\" + command + r"\{([^{}]*)\}", r"\1", span)
    span = span.replace("{", "").replace("}", "")
    span = re.sub(r"(?<=\d),(?=\d)", "", span)
    span = span.lower()
    span = re.sub(r"\s+", " ", span).strip()
    span = span.rstrip(".")
    return span


def _parse_number(text):
    compact = text.replace(" ", "").replace("(", "").replace(")", "")
    if _NUMBER.fullmatch(compact) is None:
        return None
    return Fraction(compact)


def _canonical_number(value):
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _split_unit(text):
    """Return (number, unit or None) when `text` is a number plus a known unit."""
    for alias, canon in _UNITS:
        if not text.endswith(alias):
            continue
        head = text[: -len(alias)].rstrip()
        if head == "" or (alias[0].isalpha() and head[-1].isalpha()):
            continue
        number = _parse_number(head)
        if number is not None:
            return number, canon
    number = _parse_number(text)
    if number is None:
        return None
    return number, None


def normalize_math_answer(text):
    """Canonical form of a math answer.

    Takes the last ``\\boxed{...}`` span when one is present. Fractions and
    decimals become a reduced fraction, trailing zeros disappear, commas
    inside a number are removed, and a trailing unit is rewritten to one
    canonical word.

    Args:
        text: the model's answer, or a whole response that contains a box.

    Returns:
        A string such as ``"1/2"``, ``"5"``, or ``"5 meter"``. A string that
        is not a number comes back cleaned and lowercased.
    """
    cleaned = _cleanup(text)
    parsed = _split_unit(cleaned)
    if parsed is None:
        return cleaned
    number, unit = parsed
    canon = _canonical_number(number)
    if unit is None:
        return canon
    return f"{canon} {unit}"


def _split_canonical(norm):
    bits = norm.rsplit(" ", 1)
    if len(bits) == 2 and bits[1] in _CANONICAL_UNITS:
        return bits[0], bits[1]
    return norm, None


def is_equivalent(left, right):
    """Whether two answers name the same quantity.

    Both sides are normalized. When both carry a unit, the units must match.
    When only one carries a unit, the numbers are compared and the missing
    unit is ignored.

    Args:
        left: an answer string.
        right: an answer string.

    Returns:
        True when the canonical numbers agree under the unit rule.
    """
    left_num, left_unit = _split_canonical(normalize_math_answer(left))
    right_num, right_unit = _split_canonical(normalize_math_answer(right))
    if left_unit and right_unit and left_unit != right_unit:
        return False
    return left_num == right_num


def format_correctness_reward(response, target, format_credit=0.1):
    """Format bonus plus correctness.

    A response with no ``\\boxed{...}`` scores 0, even when the right number
    appears elsewhere. A boxed answer scores ``format_credit``. A boxed
    answer equivalent to ``target`` scores ``1 + format_credit``.

    Args:
        response: the full model response.
        target: the gold answer.
        format_credit: the bonus for producing a box. The worked example
            uses 0.1.

    Returns:
        0, ``format_credit``, or ``1 + format_credit``.
    """
    if r"\boxed{" not in response:
        return 0.0
    correct = 1.0 if is_equivalent(response, target) else 0.0
    return correct + float(format_credit)


def code_reward(program, tests, timeout):
    """Fraction of unit tests a program passes, each under ``timeout`` seconds.

    The program runs in a fresh subprocess with ``python -I``, with its
    working directory set to a temporary directory. A timeout, a non-zero
    exit, or stdout that does not match the expected text fails that test.
    Matching ignores leading and trailing whitespace.

    Args:
        program: Python source.
        tests: a sequence of mappings with ``stdin`` and ``stdout``.
        timeout: seconds allowed for each test. Must be positive.

    Returns:
        The number of passed tests divided by the number of tests.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if len(tests) == 0:
        raise ValueError("tests must not be empty")
    passed = 0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "program.py"
        path.write_text(program, encoding="utf-8")
        for test in tests:
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(path)],
                    input=test["stdin"],
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    cwd=tmp,
                    start_new_session=True,
                )
            except subprocess.TimeoutExpired:
                continue
            if proc.returncode != 0:
                continue
            if proc.stdout.strip() == str(test["stdout"]).strip():
                passed += 1
    return passed / len(tests)


def wilson_interval(successes, n, z=1.96):
    """Wilson score interval for a pass rate.

    Args:
        successes: how many of the ``n`` samples were correct.
        n: the number of samples. Must be positive.
        z: the normal quantile. 1.96 is a 95% interval.

    Returns:
        ``(low, high)``, the two ends of the interval.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must lie between 0 and n")
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    low = min(1.0, max(0.0, center - margin))
    high = min(1.0, max(0.0, center + margin))
    return low, high


def dynamic_sampling_filter(rewards, group_size):
    """Mask that keeps groups whose rewards are not all equal.

    Args:
        rewards: sequence rewards, shape ``(B,)``, with ``B`` divisible by
            ``group_size`` and each group stored contiguously.
        group_size: the number of responses per prompt.

    Returns:
        A boolean array of shape ``(B,)``. True means the response stays.
        The effective batch size is the number of True entries.
    """
    values = np.asarray(rewards, dtype=float)
    if values.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    if group_size < 2 or values.size % group_size != 0:
        raise ValueError("group_size must divide the number of rewards and be at least 2")
    mask = np.empty(values.size, dtype=bool)
    for start in range(0, values.size, group_size):
        group = values[start : start + group_size]
        mask[start : start + group_size] = float(np.max(group) - np.min(group)) > 0.0
    return mask


if __name__ == "__main__":
    print(normalize_math_answer(r"\boxed{2.50 meters}"))
    print(format_correctness_reward(r"\boxed{2.50 meters}", "5/2"))
    program = "a, b = map(int, input().split())\nprint(a + b)\n"
    tests = (
        {"stdin": "2 3\n", "stdout": "5\n"},
        {"stdin": "10 1\n", "stdout": "11\n"},
    )
    print(code_reward(program, tests, 0.5))
    print(wilson_interval(3, 8))

"""The assertion vocabulary an exercise's `tests.py` writes against.

A check is a function registered with `@check`, taking the reader's module as its
only argument. It passes by returning and fails by raising `CheckFailure`, which
the helpers here raise on your behalf with a message the reader can act on.

The harness imports this module under the name `rlcheck` and reads `REGISTRY`.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence


class CheckFailure(AssertionError):
    """Raised when a check's expectation is not met."""


@dataclass
class Check:
    description: str
    func: Callable[[Any], None]
    timings: dict[str, float] = field(default_factory=dict)


REGISTRY: list[Check] = []

# Populated by `timed`, drained by the harness after each check.
_TIMINGS: dict[str, float] = {}


def check(description: str) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
    """Register a check. The description is the line the reader sees."""

    def decorate(func: Callable[[Any], None]) -> Callable[[Any], None]:
        REGISTRY.append(Check(description=description, func=func))
        return func

    return decorate


def fail(message: str) -> None:
    """Fail the current check with `message`."""
    raise CheckFailure(message)


def _numpy():
    import numpy as np

    return np


def _describe(value: Any) -> str:
    np = _numpy()
    array = np.asarray(value)
    if array.ndim == 0:
        return f"{array.item():.6g}"
    if array.size <= 8:
        return np.array2string(array, precision=6, suppress_small=True)
    flat = array.ravel()
    head = np.array2string(flat[:4], precision=6, suppress_small=True)
    tail = np.array2string(flat[-2:], precision=6, suppress_small=True)
    return f"shape {array.shape}, {head[:-1]} ... {tail[1:]}"


def _label(label: str | None) -> str:
    return f"{label}: " if label else ""


def close(
    got: Any,
    want: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    label: str | None = None,
) -> None:
    """Fail unless `got` and `want` agree elementwise within tolerance."""
    np = _numpy()
    got_a = np.asarray(got, dtype=float)
    want_a = np.asarray(want, dtype=float)
    if got_a.shape != want_a.shape:
        fail(
            f"{_label(label)}shape {got_a.shape} does not match the expected "
            f"{want_a.shape}"
        )
    if not np.all(np.isfinite(got_a)):
        fail(f"{_label(label)}result holds NaN or infinity: {_describe(got_a)}")
    if np.allclose(got_a, want_a, rtol=rtol, atol=atol):
        return
    diff = np.abs(got_a - want_a)
    worst = float(diff.max()) if diff.size else 0.0
    fail(
        f"{_label(label)}largest elementwise difference is {worst:.3g}, above the "
        f"tolerance rtol={rtol:g} atol={atol:g}.\n"
        f"  got:  {_describe(got_a)}\n"
        f"  want: {_describe(want_a)}"
    )


def almost(got: Any, want: Any, tol: float, label: str | None = None) -> None:
    """Fail unless two scalars agree within `tol`."""
    got_f = float(got)
    want_f = float(want)
    # Written as `not <=` so that a NaN, which compares false with everything,
    # fails the check rather than slipping past it.
    if not abs(got_f - want_f) <= tol:
        fail(
            f"{_label(label)}got {got_f:.6g}, expected {want_f:.6g} within {tol:g}"
        )


def equal(got: Any, want: Any, label: str | None = None) -> None:
    """Fail unless two values are exactly equal."""
    np = _numpy()
    got_a = np.asarray(got)
    want_a = np.asarray(want)
    if got_a.shape != want_a.shape or not np.array_equal(got_a, want_a):
        fail(
            f"{_label(label)}values differ.\n"
            f"  got:  {_describe(got_a)}\n"
            f"  want: {_describe(want_a)}"
        )


def shape(value: Any, want: Sequence[int], label: str | None = None) -> None:
    """Fail unless `value` has the given shape."""
    np = _numpy()
    got = tuple(np.asarray(value).shape)
    if got != tuple(want):
        fail(f"{_label(label)}shape is {got}, expected {tuple(want)}")


def finite(value: Any, label: str | None = None) -> None:
    """Fail when any element is NaN or infinite."""
    np = _numpy()
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        fail(f"{_label(label)}result holds NaN or infinity: {_describe(array)}")


def has(module: Any, name: str) -> Any:
    """Return `module.name`, failing with a readable message when it is missing."""
    if not hasattr(module, name):
        fail(f"your module defines no `{name}`")
    return getattr(module, name)


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Record how long a block takes, and report it beside the check."""
    start = time.perf_counter()
    try:
        yield
    finally:
        _TIMINGS[label] = time.perf_counter() - start


def drain_timings() -> dict[str, float]:
    """Take and clear the timings recorded since the last call."""
    taken = dict(_TIMINGS)
    _TIMINGS.clear()
    return taken


def finite_difference(
    func: Callable[[Any], float], x: Any, eps: float = 1e-6
) -> Any:
    """Central-difference gradient of a scalar function, for checking gradients."""
    np = _numpy()
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        index = it.multi_index
        original = x[index]
        x[index] = original + eps
        high = float(func(x))
        x[index] = original - eps
        low = float(func(x))
        x[index] = original
        grad[index] = (high - low) / (2 * eps)
        it.iternext()
    return grad

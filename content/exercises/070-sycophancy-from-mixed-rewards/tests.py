"""Checks for the draft exercise."""

from rlcheck import check, fail, has

@check("draft_answer returns 0 when it is given 0")
def test_steps_0(sub):
    got = has(sub, "draft_answer")(0)
    if got != 0:
        fail(f"got {got}, expected 0")


@check("draft_answer returns 1 when it is given 1")
def test_steps_1(sub):
    got = has(sub, "draft_answer")(1)
    if got != 1:
        fail(f"got {got}, expected 1")


@check("draft_answer returns 2 when it is given 2")
def test_steps_2(sub):
    got = has(sub, "draft_answer")(2)
    if got != 2:
        fail(f"got {got}, expected 2")


@check("draft_answer returns 3 when it is given 3")
def test_steps_3(sub):
    got = has(sub, "draft_answer")(3)
    if got != 3:
        fail(f"got {got}, expected 3")


@check("draft_answer returns 5 when it is given 5")
def test_steps_5(sub):
    got = has(sub, "draft_answer")(5)
    if got != 5:
        fail(f"got {got}, expected 5")


@check("draft_answer returns 8 when it is given 8")
def test_steps_8(sub):
    got = has(sub, "draft_answer")(8)
    if got != 8:
        fail(f"got {got}, expected 8")

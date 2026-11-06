"""Checks for the Dyna and prioritized sweeping exercise."""

import numpy as np

import rlenvs
from rlcheck import almost, check, close, fail, has, shape

_MAZE = [
    ".......#G",
    "..#....#.",
    "S.#....#.",
    "..#......",
    ".....#...",
    ".........",
]


def _maze():
    return rlenvs.gridworld(_MAZE, gamma=0.95, step_reward=0.0, goal_reward=1.0, max_steps=2000)


def _full_model(mdp):
    # Every transition of a deterministic world, as (r, s2, terminated).
    model = {}
    for s in range(mdp.n_states):
        if mdp.terminal[s]:
            continue
        for a in range(mdp.n_actions):
            s2, r = mdp.sample(s, a, np.random.default_rng(0))
            model[(s, a)] = (r, s2, bool(mdp.terminal[s2]))
    return model


def _q_star(mdp):
    return rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp))


@check("q_update moves Q[s, a] a fraction alpha toward the one-step target, and drops the bootstrap at a terminal")
def test_q_update(sub):
    q_update = has(sub, "q_update")
    Q = np.array([[0.0, 0.0], [1.0, 3.0]])
    # Target 0.5 + 0.9 * max(1, 3) = 3.2, error 3.2, step 0.5 * 3.2 = 1.6.
    got = q_update(Q.copy(), 0, 1, 0.5, 1, False, 0.5, 0.9)
    close(got, np.array([[0.0, 1.6], [1.0, 3.0]]), atol=1e-12, label="Q after a non-terminal update")
    # Terminal: the target is r alone, 0.5, and the step is 0.5 * 0.5.
    got = q_update(Q.copy(), 0, 1, 0.5, 1, True, 0.5, 0.9)
    close(got, np.array([[0.0, 0.25], [1.0, 3.0]]), atol=1e-12, label="Q after a terminal update")


@check("plan: one planning step on the chain sets Q(s0, right) = 0.9 with no real experience")
def test_plan_chain(sub):
    plan = has(sub, "plan")
    # Chain s0 -> s1 -> s2 (terminal), one action, reward 1 on reaching s2.
    # After one real episode with alpha = 1, Q(s1) = 1 and Q(s0) = 0.
    Q = np.array([[0.0], [1.0], [0.0]])
    model = {(0, 0): (0.0, 1, False)}
    got = plan(Q.copy(), model, 1, 1.0, 0.9, np.random.default_rng(0))
    close(got, np.array([[0.9], [1.0], [0.0]]), atol=1e-12, label="Q after one planning step")
    # With a terminal transition in the model, planning must not bootstrap past it.
    model = {(1, 0): (1.0, 2, True)}
    got = plan(np.array([[0.0], [0.3], [5.0]]), model, 1, 1.0, 0.9, np.random.default_rng(0))
    almost(got[1, 0], 1.0, 1e-12, label="Q(s1) after planning on the terminal step")


@check("plan with the maze's full model and alpha = 1 converges to Q* from value iteration")
def test_plan_converges(sub):
    plan = has(sub, "plan")
    mdp = _maze()
    Q = plan(np.zeros((mdp.n_states, mdp.n_actions)), _full_model(mdp), 20000, 1.0, mdp.gamma,
             np.random.default_rng(1))
    close(Q, _q_star(mdp), atol=1e-6, label="Q after 20,000 planning updates")


@check("Dyna-Q with 50 planning steps needs far fewer real steps in episodes 2 to 10 than with none")
def test_dyna_q(sub):
    f = has(sub, "dyna_q")
    mdp = _maze()
    Q0, steps0 = f(mdp, 10, 0, 0.1, 0.1, seed=0)
    Q50, steps50 = f(mdp, 10, 50, 0.1, 0.1, seed=0)
    shape(steps0, (10,), label="steps per episode")
    shape(Q50, (mdp.n_states, mdp.n_actions), label="Q")
    a, b = int(np.sum(steps0[1:])), int(np.sum(steps50[1:]))
    # Twenty seeds gave 1,659 to 4,606 steps without planning and 152 to 248 with it.
    if not (b < 400 and b * 4 < a):
        fail(
            "over episodes 2 to 10, expected about 150 to 250 real steps with 50 planning "
            f"steps and several times more with none; got {b} with planning and {a} without"
        )


@check("prioritized_sweeping reaches Q* in a few hundred updates, far fewer than random planning")
def test_prioritized_sweeping(sub):
    f = has(sub, "prioritized_sweeping")
    mdp = _maze()
    model = _full_model(mdp)
    q_star = _q_star(mdp)
    Q, updates = f(model, mdp.n_states, mdp.n_actions, mdp.gamma, theta=1e-5)
    shape(Q, (mdp.n_states, mdp.n_actions), label="Q")
    close(Q, q_star, atol=1e-3, label="Q after prioritized sweeping")
    # Uniform random planning, alpha = 1, counted in batches of 50.
    rng = np.random.default_rng(0)
    pairs = list(model)
    R = np.zeros((mdp.n_states, mdp.n_actions))
    random_updates = 0
    while np.abs(R - q_star).max() >= 1e-3:
        for _ in range(50):
            s, a = pairs[int(rng.integers(len(pairs)))]
            r, s2, done = model[(s, a)]
            R[s, a] = r + mdp.gamma * R[s2].max() * (not done)
        random_updates += 50
    if not (int(updates) < 1000 and int(updates) * 5 < random_updates):
        fail(
            f"prioritized sweeping took {int(updates)} updates; random planning took "
            f"{random_updates}. Expected a few hundred, at least five times fewer."
        )

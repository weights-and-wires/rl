"""Checks for the goal-conditioned RL exercise."""

import numpy as np
import torch

from rlcheck import almost, check, close, equal, fail, finite_difference, has, shape

torch.set_num_threads(1)

N_BITS = 12


def _episode():
    """Three steps on 3 bits: 000 -> 100 -> 110 -> 010, aiming at 111 and missing."""
    states = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    return {"states": states, "actions": np.array([0, 1, 0]), "goal": np.array([1, 1, 1], dtype=np.float32)}


def _chain_episode(T=6):
    """T steps on 8 bits that each set one more bit, so every achieved state is distinct."""
    states = np.zeros((T + 1, 8), dtype=np.float32)
    for t in range(1, T + 1):
        states[t:, t - 1] = 1.0
    return {"states": states, "actions": np.arange(T), "goal": np.ones(8, dtype=np.float32)}


def _index_of(states, g):
    """The step whose state equals g, or -1."""
    hits = np.flatnonzero(np.all(states == g, axis=1))
    return int(hits[0]) if len(hits) else -1


@check("goal_reward is 0 where the achieved goal equals the goal and -1 elsewhere")
def test_goal_reward(sub):
    goal_reward = has(sub, "goal_reward")
    achieved = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=np.float32)
    goal = np.array([[1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float32)
    got = goal_reward(achieved, goal)
    shape(got, (3,), label="goal_reward on a batch")
    close(got, [0.0, -1.0, -1.0], atol=1e-12, label="rewards for a batch of three")
    close(goal_reward(achieved[0], goal[0]), 0.0, atol=1e-12, label="reward for one matching pair")
    close(goal_reward(achieved[0], goal[1]), -1.0, atol=1e-12, label="reward for one mismatched pair")


@check("relabel with strategy 'final' adds one copy per step aimed at the last state, with the reward recomputed")
def test_relabel_final(sub):
    relabel = has(sub, "relabel")
    ep = _episode()
    out = relabel(ep, "final", 4, np.random.default_rng(0))
    equal(len(out), 6, label="number of transitions (3 original + 3 relabeled)")
    # The first three are the original transitions, with goal 111 and reward -1.
    for t in range(3):
        s, a, r, s2, g, done = out[t]
        close(g, [1, 1, 1], label=f"original goal at step {t}")
        close(r, -1.0, atol=1e-12, label=f"original reward at step {t}")
        if done:
            fail(f"original step {t} never reached 111, so done should be False, got True")
    # The relabeled copies aim at states[3] = 010. Only the last step reaches it.
    # Wrong implementation: relabel the goal but keep the original reward (-1).
    want_r = [-1.0, -1.0, 0.0]
    want_done = [False, False, True]
    for t in range(3):
        s, a, r, s2, g, done = out[3 + t]
        close(s, ep["states"][t], label=f"relabeled s at step {t}")
        equal(int(a), int(ep["actions"][t]), label=f"relabeled action at step {t}")
        close(s2, ep["states"][t + 1], label=f"relabeled s2 at step {t}")
        close(g, [0, 1, 0], label=f"relabeled goal at step {t}")
        close(r, want_r[t], atol=1e-12, label=f"relabeled reward at step {t}, expected {want_r[t]}")
        equal(bool(done), want_done[t], label=f"relabeled done at step {t}")
    none = relabel(ep, "none", 4, np.random.default_rng(0))
    equal(len(none), 3, label="strategy 'none' returns only the three original transitions")


@check("relabel 'future' draws each goal from steps t+1 .. T, never the state before the action")
def test_relabel_future(sub):
    relabel = has(sub, "relabel")
    # Wrong implementation: sample j from t .. T (picks the state before the action),
    # or from t + 2 .. T (skips this step's own achieved goal), or from the whole episode.
    ep = _chain_episode(6)
    states, T, k = ep["states"], 6, 5
    seen = {t: set() for t in range(T)}
    for seed in range(40):
        out = relabel(ep, "future", k, np.random.default_rng(seed))
        equal(len(out), T + T * k, label="number of transitions, T originals plus k per step")
        for t in range(T):
            for c in range(k):
                g = out[T + t * k + c][4]
                j = _index_of(states, g)
                if j < t + 1:
                    fail(f"step {t} got a goal from step {j}; expected one of steps {t + 1}..{T}")
                seen[t].add(j)
    for t in range(T):
        want = set(range(t + 1, T + 1))
        if seen[t] != want:
            fail(f"over 40 draws, step {t} should see goals from steps {sorted(want)}, got {sorted(seen[t])}")


@check("relabel 'episode' draws goals from any step 1 .. T of the episode")
def test_relabel_episode(sub):
    relabel = has(sub, "relabel")
    ep = _chain_episode(6)
    states, T, k = ep["states"], 6, 5
    seen = set()
    for seed in range(40):
        out = relabel(ep, "episode", k, np.random.default_rng(seed))
        equal(len(out), T + T * k, label="number of transitions, T originals plus k per step")
        for t in range(T):
            for c in range(k):
                j = _index_of(states, out[T + t * k + c][4])
                if j < 1:
                    fail(f"step {t} got the start state as a goal (step {j}); expected a step in 1..{T}")
                if t == T - 1:
                    seen.add(j)
    if seen != set(range(1, T + 1)):
        fail(f"the last step should see goals from steps {list(range(1, T + 1))} over 40 draws, got {sorted(seen)}")


@check("every relabeled transition has r = goal_reward(s2, g) and done = (r == 0), and relabeling creates successes")
def test_relabel_consistency(sub):
    relabel = has(sub, "relabel")
    goal_reward = has(sub, "goal_reward")
    ep = _chain_episode(6)
    for strategy in ("final", "future", "episode"):
        out = relabel(ep, strategy, 3, np.random.default_rng(1))
        successes = 0
        for i, (s, a, r, s2, g, done) in enumerate(out):
            want = float(goal_reward(s2, g))
            if not np.isclose(r, want):
                fail(f"strategy {strategy!r}, transition {i}: reward should be goal_reward(s2, g) = {want}, got {r}")
            if bool(done) != (want == 0.0):
                fail(f"strategy {strategy!r}, transition {i}: done should be {want == 0.0}, got {bool(done)}")
            successes += want == 0.0
        if successes < 1:
            fail(f"strategy {strategy!r} should create at least one transition that reaches its goal, got {successes}")


def _linear(w, b):
    net = torch.nn.Linear(4, 2).double()
    with torch.no_grad():
        net.weight.copy_(torch.tensor(w, dtype=torch.float64))
        net.bias.copy_(torch.tensor(b, dtype=torch.float64))
    return net


def _batch():
    return {
        "s": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        "g": torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float64),
        "s2": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64),
        "a": torch.tensor([1, 0]),
        "r": torch.tensor([0.0, -1.0], dtype=torch.float64),
        "done": torch.tensor([1.0, 0.0], dtype=torch.float64),
    }


@check("dqn_loss matches a hand-computed value and ignores the bootstrap on done transitions")
def test_dqn_loss_value(sub):
    dqn_loss = has(sub, "dqn_loss")
    q_net = _linear([[1, 0, 0, 0], [0, 1, 0, 0]], [0.0, 0.0])
    target = _linear([[0, 0, 1, 0], [0, 0, 0, 1]], [0.5, 0.0])
    loss = dqn_loss(q_net, target, _batch(), 0.9)
    # Row 0: input [1,0,1,1], Q = [1, 0], a = 1 -> Q = 0. done, so y = r = 0. Error 0.
    # Row 1: input [0,1,0,0], Q = [0, 1], a = 0 -> Q = 0. Next input [1,1,0,0], target values
    #   [0.5 + 0, 0 + 0] -> max 0.5, so y = -1 + 0.9 * 0.5 = -0.55. Error 0.55, squared 0.3025.
    # Mean over two rows: 0.15125. Wrong implementation: forget (1 - done), which adds
    # row 0 a bootstrap of 0.9 * max([0.5 + 1, 1]) = 1.35 and gives a different loss.
    shape(loss.detach(), (), label="the loss is a scalar")
    almost(float(loss.detach()), 0.15125, 1e-9, label="loss on the two-row batch")


@check("dqn_loss gradient matches finite differences and does not flow into the target network")
def test_dqn_loss_gradient(sub):
    dqn_loss = has(sub, "dqn_loss")
    rng = np.random.default_rng(0)
    w = rng.standard_normal((2, 4))
    b = rng.standard_normal(2)
    target = _linear(rng.standard_normal((2, 4)), rng.standard_normal(2))
    q_net = _linear(w, b)
    dqn_loss(q_net, target, _batch(), 0.9).backward()
    got = np.concatenate([q_net.weight.grad.numpy().ravel(), q_net.bias.grad.numpy().ravel()])

    def f(x):
        net = _linear(x[:8].reshape(2, 4), x[8:])
        return float(dqn_loss(net, target, _batch(), 0.9).detach())

    want = finite_difference(f, np.concatenate([w.ravel(), b]))
    close(got, want, rtol=1e-5, atol=1e-7, label="gradient of the loss with respect to the Q network")
    # Wrong implementation: let the loss differentiate through the target.
    if target.weight.grad is not None and float(target.weight.grad.abs().sum()) > 0:
        fail("the target network received a gradient; compute y under torch.no_grad()")


@check("gcsl_targets pairs each step with a goal drawn from a later state and keeps the action")
def test_gcsl_targets(sub):
    gcsl_targets = has(sub, "gcsl_targets")
    # Wrong implementation: draw the goal from t .. T, so a step can be labeled with the
    # state it started from, or from the whole episode, so a step is labeled with its past.
    ep = _chain_episode(6)
    states, T = ep["states"], 6
    seen = {t: set() for t in range(T)}
    for seed in range(40):
        obs, goals, actions = gcsl_targets(ep, np.random.default_rng(seed))
        shape(obs, (T, 8), label="obs")
        shape(goals, (T, 8), label="goals")
        close(obs, states[:T], label="obs should be states[:T]")
        equal(np.asarray(actions).tolist(), ep["actions"].tolist(), label="actions")
        for t in range(T):
            j = _index_of(states, goals[t])
            if j < t + 1:
                fail(f"step {t} got a goal from step {j}; expected one of steps {t + 1}..{T}")
            seen[t].add(j)
    equal(sorted(seen[T - 1]), [T], label="the last step's only possible goal is the final state")
    equal(sorted(seen[0]), list(range(1, T + 1)), label="the first step's goals over 40 draws")


@check("plain goal-conditioned DQN fails on 12-bit bit-flipping: it almost never sees a reward")
def test_plain_fails(sub):
    train_gcdqn = has(sub, "train_gcdqn")
    rate, _ = train_gcdqn(N_BITS, "none", seed=0)
    if not rate < 0.15:
        fail(f"without relabeling, the greedy success rate on {N_BITS} bits should stay under 0.15, got {rate:.2f}")


@check("with 'future' relabeling the same agent reaches new goals on 12 bits")
def test_her_solves(sub):
    train_gcdqn = has(sub, "train_gcdqn")
    rate, _ = train_gcdqn(N_BITS, "future", seed=0)
    if not rate > 0.7:
        fail(f"with future relabeling, the greedy success rate on {N_BITS} bits should exceed 0.7, got {rate:.2f}")

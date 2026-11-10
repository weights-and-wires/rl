"""Checks for the partial-observability exercise."""

import itertools

import numpy as np
import torch

import rlenvs
from rlcheck import almost, check, close, fail, finite_difference, has, shape

torch.set_num_threads(1)

# The tiger problem. States: 0 tiger left, 1 tiger right. Actions: 0 listen,
# 1 open left, 2 open right. Observations: 0 hear left, 1 hear right.
# Listening keeps the state and is right 85% of the time. Opening a door resets
# the tiger to a random side and gives a useless coin-flip observation.
T_TIGER = np.stack([np.eye(2), np.full((2, 2), 0.5), np.full((2, 2), 0.5)])
O_TIGER = np.stack([
    [[0.85, 0.15], [0.15, 0.85]],
    np.full((2, 2), 0.5),
    np.full((2, 2), 0.5),
])

SEEDS = (0, 1, 2, 3, 4)


@check("belief_update turns two 'hear left' listens into 0.85 and then 0.9698 for tiger-left")
def test_belief_listen(sub):
    belief_update = has(sub, "belief_update")
    b0 = np.array([0.5, 0.5])
    b1 = belief_update(b0, 0, 0, T_TIGER, O_TIGER)
    shape(b1, (2,), label="belief")
    close(b1, [0.85, 0.15], atol=1e-12, label="one listen, hear left, from [0.5, 0.5]")
    b2 = belief_update(b1, 0, 0, T_TIGER, O_TIGER)
    # 0.85 * 0.85 = 0.7225 and 0.15 * 0.15 = 0.0225, so 0.7225 / 0.745.
    close(b2, [0.7225 / 0.745, 0.0225 / 0.745], atol=1e-12, label="two listens, both hear left")
    b3 = belief_update(b2, 0, 1, T_TIGER, O_TIGER)
    # The odds are 0.7225 / 0.0225 = 32.11; one 'hear right' divides by 0.85 / 0.15 = 5.667.
    odds = (0.7225 / 0.0225) * (0.15 / 0.85)
    close(b3, [odds / (1 + odds), 1 / (1 + odds)], atol=1e-12, label="then one listen that hears right")
    if not np.array_equal(b0, [0.5, 0.5]):
        fail(f"belief_update changed its input in place: b is now {b0}, expected [0.5 0.5]")


@check("belief_update pushes the belief through the transition: opening a door forgets everything")
def test_belief_open(sub):
    belief_update = has(sub, "belief_update")
    # Named wrong implementation: apply the observation likelihood to b without
    # first multiplying by T. For listening, T is the identity, so only the
    # open-door case exposes the missing step.
    for a in (1, 2):
        got = belief_update(np.array([0.97, 0.03]), a, 0, T_TIGER, O_TIGER)
        close(got, [0.5, 0.5], atol=1e-12,
              label=f"after action {a} (an open door) the tiger is re-placed, so the belief is uniform")
    # A skewed prior with a listen that disagrees with it.
    got = belief_update(np.array([0.9, 0.1]), 0, 1, T_TIGER, O_TIGER)
    # 0.9 * 0.15 = 0.135 and 0.1 * 0.85 = 0.085, so 0.135 / 0.22.
    close(got, [0.135 / 0.22, 0.085 / 0.22], atol=1e-12, label="belief [0.9, 0.1], listen, hear right")
    # Second wrong implementation: weight b by the observation first, then push
    # it through T. That treats o as evidence about the old state. On a POMDP
    # whose transitions move the state, it differs from Bayes' rule over the
    # landing state, P(s2 | b, a, o) proportional to sum_s b(s) T(s, s2) O(s2, o).
    rng = np.random.default_rng(7)
    T = rng.dirichlet(np.ones(3), size=(2, 3))
    O = rng.dirichlet(np.ones(2), size=(2, 3))
    b = np.array([0.6, 0.3, 0.1])
    joint = b[:, None] * T[1] * O[1][None, :, 0]
    expected = joint.sum(axis=0) / joint.sum()
    got = belief_update(b, 1, 0, T, O)
    close(got, expected, atol=1e-12, label="a random 3-state POMDP, action 1, observation 0")


@check("memoryless_value is the exact value of a policy that sees only the observation")
def test_memoryless_value(sub):
    memoryless_value = has(sub, "memoryless_value")
    mdp, obs_of_state = sub.tmaze_mdp()
    # Advance until the junction (observation 3), then always open the left door.
    pi_obs = np.zeros((5, 3))
    pi_obs[[0, 1, 2], 0] = 1.0
    pi_obs[3, 1] = 1.0
    pi_obs[4, 0] = 1.0
    got = memoryless_value(mdp, obs_of_state, pi_obs)
    # Left is right half the time, so the door is worth +1 and -1 equally: 0.
    almost(got, 0.0, 1e-9, label="advance, then always open the left door")
    # A policy that walks into the wall at the start earns -1 on every episode.
    pi_wall = np.zeros((5, 3))
    pi_wall[:, 1] = 1.0
    almost(memoryless_value(mdp, obs_of_state, pi_wall), -1.0, 1e-9, label="open a door at the start")
    # Walk into the wall when the signal says right; otherwise advance and open the left door.
    pi_mix = np.zeros((5, 3))
    pi_mix[[0, 2], 0] = 1.0
    pi_mix[1, 1] = 1.0
    pi_mix[3, 1] = 1.0
    pi_mix[4, 0] = 1.0
    # Signal left (probability 0.5): +1 at step 4, worth 0.95 ** 4. Signal right: -1 at step 0.
    almost(memoryless_value(mdp, obs_of_state, pi_mix), 0.5 * 0.95 ** 4 - 0.5, 1e-9,
           label="advance and open left on a left signal, walk into the wall on a right signal")


@check("best_memoryless_value finds 0, a coin flip, and no stochastic memoryless policy beats it")
def test_best_memoryless(sub):
    best_memoryless_value = has(sub, "best_memoryless_value")
    mdp, obs_of_state = sub.tmaze_mdp()
    value, pi_obs = best_memoryless_value(mdp, obs_of_state)
    almost(value, 0.0, 1e-9, label="best deterministic memoryless value")
    full = float(mdp.mu0 @ rlenvs.value_iteration(mdp))
    almost(full, 0.95 ** 4, 1e-9, label="the value of the best policy that sees the full state")
    rng = np.random.default_rng(0)
    worst_excess = -np.inf
    for _ in range(300):
        pi = rng.dirichlet(np.full(3, 0.3), size=5)
        v = float(mdp.mu0 @ rlenvs.policy_evaluation(mdp, pi[obs_of_state]))
        worst_excess = max(worst_excess, v - value)
    if worst_excess > 1e-9:
        fail(f"a random memoryless policy scored {worst_excess:.3g} above the reported best {value:.6g}")
    # The reported policy must really have that value.
    got = float(mdp.mu0 @ rlenvs.policy_evaluation(mdp, np.asarray(pi_obs)[obs_of_state]))
    almost(got, value, 1e-9, label="value of the policy that best_memoryless_value returned")


@check("reset_hidden zeroes the rows that start an episode and leaves the others alone")
def test_reset_hidden(sub):
    reset_hidden = has(sub, "reset_hidden")
    h = torch.arange(1.0, 7.0).reshape(3, 2)
    first = torch.tensor([0.0, 1.0, 0.0])
    got = reset_hidden(h, first)
    close(got.numpy(), [[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]], atol=1e-12, label="reset rows")
    close(h.numpy(), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], atol=1e-12, label="the input h")


@check("unroll resets the hidden state at a boundary: episode B inside a stream equals episode B alone")
def test_unroll_reset(sub):
    unroll = has(sub, "unroll")
    # Named wrong implementations: never reset (episode A leaks into B), or
    # reset after the step instead of before it (the first step of B still sees A).
    torch.manual_seed(0)
    cell = torch.nn.GRUCell(3, 4).double()
    obs = torch.randn(7, 2, 3, dtype=torch.float64)
    first = torch.zeros(7, 2, dtype=torch.float64)
    first[0] = 1.0
    first[3, 0] = 1.0  # column 0 starts a new episode at t = 3
    first[2, 1] = 1.0  # column 1 starts a new episode at t = 2
    out = unroll(cell, obs, first)
    if tuple(out.shape) != (7, 2, 4):
        fail(f"unroll should return shape (7, 2, 4), got {tuple(out.shape)}")
    for col, start in ((0, 3), (1, 2)):
        h = torch.zeros(1, 4, dtype=torch.float64)
        alone = []
        for t in range(start, 7):  # the reference: the cell, stepped by hand from zeros
            h = cell(obs[t, col:col + 1], h)
            alone.append(h[0])
        close(out[start:, col].detach().numpy(), torch.stack(alone).detach().numpy(), atol=1e-12,
              label=f"column {col}, steps from the boundary at t = {start}, against the same steps run alone")
    # A nonzero starting state is wiped by first = 1 at step 0.
    h0 = torch.ones(2, 4, dtype=torch.float64)
    close(unroll(cell, obs, first, h0=h0).detach().numpy(), out.detach().numpy(), atol=1e-12,
          label="outputs when h0 is nonzero but first[0] = 1")
    # Without a reset at step 0, h0 matters.
    first_no0 = first.clone()
    first_no0[0] = 0.0
    shifted = unroll(cell, obs, first_no0, h0=h0)
    if torch.allclose(shifted[0], out[0]):
        fail("with first[0] = 0 and a nonzero h0, the first output should change but it did not")


@check("no gradient crosses an episode boundary, and every step inside an episode still gets one")
def test_no_gradient_across_boundary(sub):
    unroll = has(sub, "unroll")
    torch.manual_seed(1)
    cell = torch.nn.GRUCell(3, 4).double()
    obs = torch.randn(6, 1, 3, dtype=torch.float64, requires_grad=True)
    first = torch.zeros(6, 1, dtype=torch.float64)
    first[0] = 1.0
    first[3] = 1.0
    out = unroll(cell, obs, first)
    out[3:].sum().backward()  # a loss that reads only the second episode
    g = obs.grad[:, 0]
    if not torch.all(g[:3] == 0):
        fail(f"gradient reached the first episode through the boundary: {g[:3].abs().max():.3g}, expected exactly 0")
    for t in range(3, 6):
        if float(g[t].abs().max()) == 0.0:
            fail(f"step {t} is inside the second episode but received no gradient")
    # And the other direction: a loss on the first episode alone sees no later step.
    obs.grad = None
    unroll(cell, obs, first)[:3].sum().backward()
    if not torch.all(obs.grad[3:] == 0):
        fail("gradient from the first episode reached steps of the second")


@check("masked_pg_loss averages over real steps only, ignores padding, and matches finite differences")
def test_masked_pg_loss(sub):
    masked_pg_loss = has(sub, "masked_pg_loss")
    # Named wrong implementations: forget the mask, or divide by T * B instead of mask.sum().
    logits = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]], [[5.0, -5.0]]], dtype=torch.float64)  # T=3, B=1
    actions = torch.tensor([[0], [1], [1]])
    adv = torch.tensor([[2.0], [-1.0], [100.0]], dtype=torch.float64)
    mask = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float64)
    got = float(masked_pg_loss(logits, actions, adv, mask))
    # Two real steps, each with log pi = log 0.5. Loss = -(2 * log 0.5 + -1 * log 0.5) / 2.
    want = -(2.0 * np.log(0.5) - 1.0 * np.log(0.5)) / 2.0
    almost(got, want, 1e-12, label="loss with one padded step (garbage advantage 100)")
    rng = np.random.default_rng(0)
    L0 = rng.standard_normal((4, 3, 3))
    A0 = torch.as_tensor(rng.integers(0, 3, size=(4, 3)))
    ADV = torch.as_tensor(rng.standard_normal((4, 3)))
    MASK = torch.as_tensor(np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float))

    def f(x):
        return float(masked_pg_loss(torch.as_tensor(x.reshape(4, 3, 3)), A0, ADV, MASK))

    x = torch.as_tensor(L0, dtype=torch.float64).clone().requires_grad_(True)
    masked_pg_loss(x, A0, ADV, MASK).backward()
    close(x.grad.numpy().ravel(), finite_difference(f, L0.ravel()), atol=1e-6,
          label="gradient of the loss with respect to the logits, against finite differences")
    if not np.all(x.grad.numpy()[3] == 0):
        fail("the row of padding steps must receive zero gradient")


@check("a padded, masked batch gives the same gradient as running each episode step by step")
def test_sequence_loss_matches_stepwise(sub):
    sequence_loss = has(sub, "sequence_loss")
    torch.manual_seed(2)
    lengths = [5, 3, 4]
    T, B, D, H, A = 5, 3, 3, 4, 3
    cell = torch.nn.GRUCell(D, H).double()
    head = torch.nn.Linear(H, A).double()
    obs = torch.randn(T, B, D, dtype=torch.float64)  # padded steps hold random garbage
    actions = torch.randint(0, A, (T, B))
    adv = torch.randn(T, B, dtype=torch.float64)
    mask = torch.zeros(T, B, dtype=torch.float64)
    for i, n in enumerate(lengths):
        mask[:n, i] = 1.0
    first = torch.zeros(T, B, dtype=torch.float64)
    first[0] = 1.0
    params = list(cell.parameters()) + list(head.parameters())

    batched = sequence_loss(cell, head, obs, actions, adv, mask, first)
    g_batch = torch.autograd.grad(batched, params)

    total = 0.0
    for i, n in enumerate(lengths):
        h = torch.zeros(1, H, dtype=torch.float64)
        for t in range(n):
            h = cell(obs[t, i:i + 1], h)
            logp = torch.log_softmax(head(h), dim=-1)[0, actions[t, i]]
            total = total - adv[t, i] * logp
    stepwise = total / sum(lengths)
    g_step = torch.autograd.grad(stepwise, params)

    almost(float(batched.detach()), float(stepwise.detach()), 1e-10, label="padded batch loss against the step-by-step loss")
    for k, (gb, gs) in enumerate(zip(g_batch, g_step)):
        close(gb.numpy(), gs.numpy(), atol=1e-10, label=f"gradient of parameter tensor {k}")

    # Finite differences on the inputs, through the whole unrolled loss.
    obs_np = obs.numpy().copy()

    def f(x):
        return float(sequence_loss(cell, head, torch.as_tensor(x.reshape(T, B, D)), actions, adv, mask, first))

    xo = obs.clone().requires_grad_(True)
    sequence_loss(cell, head, xo, actions, adv, mask, first).backward()
    close(xo.grad.numpy().ravel(), finite_difference(f, obs_np.ravel()), atol=1e-6,
          label="gradient of the sequence loss with respect to the observations, against finite differences")


@check("a recurrent policy trained by policy gradient beats the best memoryless policy on every seed")
def test_train_policy(sub):
    train_policy = has(sub, "train_policy")
    greedy_value = has(sub, "greedy_value")
    mdp, obs_of_state = sub.tmaze_mdp()
    best_memoryless = _best_memoryless(mdp, obs_of_state)
    rows = []
    for seed in SEEDS:
        net = train_policy(seed, memory=True)
        rows.append((seed, greedy_value(net, memory=True)))
    bad = [(s, v) for s, v in rows if v < best_memoryless + 0.5]
    if bad:
        fail("the greedy recurrent policy must beat the best memoryless value "
             f"{best_memoryless:.3f} by at least 0.5 (the full-information optimum is {0.95 ** 4:.3f}). "
             f"Got (seed, value): {[(s, round(v, 3)) for s, v in bad]}")
    # The same learner with its memory switched off cannot rise above the exact memoryless bound.
    net = train_policy(0, memory=False)
    v = greedy_value(net, memory=False)
    if v > best_memoryless + 1e-6:
        fail(f"a memoryless policy scored {v:.6f}, above the exact best memoryless value {best_memoryless:.6f}. "
             "Check that memory=False resets the hidden state at every step in training and sampling.")


def _best_memoryless(mdp, obs_of_state):
    """Exact best deterministic memoryless value, enumerated here so the check never trusts the reader."""
    best = -np.inf
    for choice in itertools.product(range(3), repeat=4):
        pi_obs = np.zeros((5, 3))
        pi_obs[:, 0] = 1.0
        pi_obs[:4] = np.eye(3)[list(choice)]
        v = float(mdp.mu0 @ rlenvs.policy_evaluation(mdp, pi_obs[obs_of_state]))
        best = max(best, v)
    return best

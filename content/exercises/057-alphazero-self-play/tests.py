"""Checks for the AlphaZero-style self-play exercise."""

import numpy as np
import torch

from rlcheck import check, close, equal, fail, finite_difference, has

torch.set_num_threads(1)


def uniform_evaluate(sub):
    """An evaluator with a flat prior over legal moves and a value of 0."""

    def evaluate(board):
        legal = np.asarray(board) == 0
        return legal / legal.sum(), 0.0

    return evaluate


@check("visit_count_policy at tau = 1 normalizes the counts, and at tau = 0 picks the most visited move, splitting ties")
def test_visit_count_policy_basic(sub):
    vcp = has(sub, "visit_count_policy")
    close(vcp(np.array([0, 6, 2, 0]), 1.0), [0, 0.75, 0.25, 0], atol=1e-12, label="tau = 1")
    close(vcp(np.array([0, 6, 2, 0]), 0.0), [0, 1, 0, 0], atol=1e-12, label="tau = 0")
    close(vcp(np.array([3, 5, 5, 0]), 0.0), [0, 0.5, 0.5, 0], atol=1e-12, label="tau = 0 with a tie")


# Named wrong implementation: normalizing the raw visit counts and ignoring tau.
# It passes the check above at tau = 1 but not this one.
@check("visit_count_policy sharpens with a small tau and flattens with a large one, without overflowing")
def test_visit_count_policy_temperature(sub):
    vcp = has(sub, "visit_count_policy")
    N = np.array([1.0, 2.0, 3.0, 0.0])
    close(vcp(N, 0.5), np.array([1, 4, 9, 0]) / 14, atol=1e-12, label="tau = 0.5 squares the counts")
    close(vcp(N, 2.0), np.array([1, np.sqrt(2), np.sqrt(3), 0]) / (1 + np.sqrt(2) + np.sqrt(3)), atol=1e-12,
          label="tau = 2 takes square roots")
    big = vcp(np.array([10000.0, 20000.0]), 0.01)
    if not np.all(np.isfinite(big)):
        fail(f"tau = 0.01 with counts of 10000 and 20000 should stay finite, got {big}")
    close(big, [0.0, 1.0], atol=1e-12, label="tau = 0.01 is almost greedy")


@check("add_root_noise mixes (1 - eps) * prior + eps * Dirichlet noise over the legal moves only")
def test_root_noise(sub):
    add_root_noise = has(sub, "add_root_noise")
    legal = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=bool)
    prior = legal / legal.sum()
    noise = np.zeros(9)
    noise[legal] = np.random.default_rng(7).dirichlet(0.3 * np.ones(5))
    got = add_root_noise(prior, legal, 0.3, 0.25, np.random.default_rng(7))
    close(got, 0.75 * prior + 0.25 * noise, atol=1e-12, label="mixed prior")
    close(got.sum(), 1.0, atol=1e-12, label="total probability")
    equal(got[~legal], np.zeros(4), label="illegal moves")
    close(add_root_noise(prior, legal, 0.3, 0.0, np.random.default_rng(7)), prior, atol=1e-12, label="eps = 0")


@check("az_loss equals value MSE plus policy cross-entropy plus L2, worked by hand")
def test_az_loss_value(sub):
    az_loss = has(sub, "az_loss")
    f = torch.float64
    # One position: uniform logits make the cross-entropy log 9 for any target.
    loss = az_loss(torch.zeros(1, 9, dtype=f), torch.tensor([0.5], dtype=f),
                   torch.eye(9, dtype=f)[:1], torch.tensor([1.0], dtype=f),
                   [torch.tensor([1.0, 2.0], dtype=f)], 0.1)
    want = (1.0 - 0.5) ** 2 + np.log(9.0) + 0.1 * (1.0 + 4.0)
    close(float(loss), want, atol=1e-9, label="loss with one position")
    # Two positions: the batch mean averages the two terms separately.
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=f)
    pi = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], dtype=f)
    loss = az_loss(logits, torch.tensor([0.0, 1.0], dtype=f), pi, torch.tensor([1.0, 1.0], dtype=f), [], 0.0)
    ce1 = np.log(np.exp(2) + 2) - 2
    lse2 = np.log(1 + np.e + 1)
    ce2 = lse2 - 0.5 * 1 - 0.5 * 0
    close(float(loss), (1.0 + 0.0) / 2 + (ce1 + ce2) / 2, atol=1e-9, label="loss with two positions")


@check("az_loss gradients match finite differences, and the logit gradient is (softmax - pi) / batch")
def test_az_loss_gradient(sub):
    az_loss = has(sub, "az_loss")
    f = torch.float64
    rng = np.random.default_rng(0)
    B = 3
    logits0 = rng.normal(size=(B, 9))
    v0 = rng.uniform(-0.8, 0.8, size=B)
    pi = rng.dirichlet(np.ones(9), size=B)
    z = np.array([1.0, -1.0, 0.0])
    theta0 = rng.normal(size=4)
    c = 0.05
    tpi, tz = torch.tensor(pi, dtype=f), torch.tensor(z, dtype=f)

    def loss_of(x):
        lg = torch.tensor(x[: 9 * B].reshape(B, 9), dtype=f)
        v = torch.tensor(x[9 * B: 10 * B], dtype=f)
        th = torch.tensor(x[10 * B:], dtype=f)
        return float(az_loss(lg, v, tpi, tz, [th], c))

    x0 = np.concatenate([logits0.ravel(), v0, theta0])
    lg = torch.tensor(logits0, dtype=f, requires_grad=True)
    v = torch.tensor(v0, dtype=f, requires_grad=True)
    th = torch.tensor(theta0, dtype=f, requires_grad=True)
    az_loss(lg, v, tpi, tz, [th], c).backward()
    analytic = np.concatenate([lg.grad.numpy().ravel(), v.grad.numpy(), th.grad.numpy()])
    close(analytic, finite_difference(loss_of, x0), atol=1e-6, label="autograd against finite differences")
    softmax = np.exp(logits0) / np.exp(logits0).sum(axis=1, keepdims=True)
    close(lg.grad.numpy(), (softmax - pi) / B, atol=1e-9, label="logit gradient")
    close(th.grad.numpy(), 2 * c * theta0, atol=1e-9, label="L2 gradient")


@check("dihedral_augment returns the 8 symmetries with the board and policy moved the same way")
def test_dihedral(sub):
    dihedral_augment = has(sub, "dihedral_augment")
    board = np.array([1, 1, 0, -1, -1, 0, 0, 0, 0], dtype=np.int8)
    pi = np.zeros(9)
    pi[2] = 1.0  # the winning cell for the player to move
    pairs = dihedral_augment(board, pi)
    equal(len(pairs), 8, label="number of symmetries")
    equal(np.asarray(pairs[0][0]), board, label="first board is the identity")
    equal(np.asarray(pairs[0][1]), pi, label="first policy is the identity")
    distinct = {tuple(np.asarray(b).tolist()) for b, _ in pairs}
    equal(len(distinct), 8, label="distinct boards from an asymmetric position")
    for i, (b, p) in enumerate(pairs):
        b, p = np.asarray(b, dtype=np.int8), np.asarray(p)
        close(p.sum(), 1.0, atol=1e-12, label=f"symmetry {i} policy total")
        # Moving the board and the policy together keeps the winning move winning.
        a = int(np.argmax(p))
        if b[a] != 0:
            fail(f"symmetry {i}: the policy points at cell {a}, which is occupied in the transformed board {b}")
        after = b.copy()
        after[a] = 1
        lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        if not any(after[line].sum() == 3 for line in lines):
            fail(f"symmetry {i}: playing the transformed policy's move {a} on board {b} does not win, "
                 "so the board and the policy were transformed differently")


@check("self_play_game labels every position with the result for the player to move there")
def test_self_play_labels(sub):
    self_play_game = has(sub, "self_play_game")
    rng = np.random.default_rng(0)
    evaluate = uniform_evaluate(sub)
    lines = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]])
    outcomes = set()
    for g in range(12):
        ex = self_play_game(evaluate, 20, 3.0, 9, 0.5, 0.25, rng)
        L = len(ex)
        if L < 5:
            fail(f"a tic-tac-toe game lasts at least 5 moves, got {L} examples")
        for i in range(L - 1):
            step = -np.asarray(ex[i + 1][0], dtype=int) - np.asarray(ex[i][0], dtype=int)
            if sorted(step.tolist())[-1] != 1 or np.count_nonzero(step) != 1 or ex[i][0][int(np.argmax(step))] != 0:
                fail(f"example {i + 1} is not the board after one move from example {i}, in the next player's view")
        z = np.array([e[2] for e in ex])
        last = np.asarray(ex[-1][0], dtype=int)
        wins = [a for a in np.flatnonzero(last == 0)
                if (np.where(np.arange(9) == a, 1, last)[lines].sum(axis=1) == 3).any()]
        # The last player to move either wins (z = +1) or fills the board (z = 0).
        want_last = 1.0 if wins else 0.0
        if z[-1] != want_last:
            fail(f"game {g}: the last position should be labeled {want_last}, got {z[-1]}")
        want = want_last * (-1.0) ** np.arange(L - 1, -1, -1)
        close(z, want, atol=1e-12, label=f"game {g} labels")
        outcomes.add(want_last)
    if outcomes != {0.0, 1.0}:
        fail(f"12 games of uniform self-play should include wins and draws, saw results {sorted(outcomes)}")


@check("self_play_game's policy target is the tau = 1 visit distribution, not the network prior")
def test_self_play_targets(sub):
    self_play_game = has(sub, "self_play_game")
    visit_count_policy = has(sub, "visit_count_policy")
    evaluate = uniform_evaluate(sub)
    board = np.array([1, 1, 0, -1, -1, 0, 0, 0, 0], dtype=np.int8)
    ex = self_play_game(evaluate, 40, 3.0, 1, 0.5, 0.0, np.random.default_rng(0), board=board)
    b0, pi0, _ = ex[0]
    equal(np.asarray(b0), board, label="first board")
    N = sub.search(board, evaluate, 40, 3.0)
    close(pi0, visit_count_policy(N, 1.0), atol=1e-12, label="target against the visit distribution")
    prior = evaluate(board)[0]
    if abs(pi0[2] - prior[2]) < 0.2:
        fail(f"cell 2 wins on the spot, so its target should be well above the prior {prior[2]:.3f}, got {pi0[2]:.3f}")


_TRAINED = {}


def trained(sub):
    if "p" not in _TRAINED:
        _TRAINED["p"] = sub.train(seed=0)
    return _TRAINED["p"]


def play_match(sub, params, opponent, games, rng, n_sim=100):
    """Play `games` games, alternating who moves first. Returns (agent wins, agent losses, draws)."""
    wins = losses = draws = 0
    for g in range(games):
        board, me, turn = np.zeros(9, dtype=np.int8), g % 2, 0
        while sub.terminal_value(board) is None:
            a = sub.az_move(params, board, n_sim) if turn % 2 == me else opponent(board, rng)
            board, turn = sub.play(board, a), turn + 1
        result = sub.terminal_value(board)
        mover_is_me = turn % 2 == me
        if result == 0:
            draws += 1
        elif (result == -1) == mover_is_me:
            losses += 1
        else:
            wins += 1
    return wins, losses, draws


def random_opponent(board, rng):
    legal = np.flatnonzero(board == 0)
    return int(rng.choice(legal))


def make_minimax(sub):
    cache = {}

    def value(board):
        key = board.tobytes()
        if key not in cache:
            tv = sub.terminal_value(board)
            cache[key] = tv if tv is not None else max(-value(sub.play(board, a)) for a in sub.legal_moves(board))
        return cache[key]

    def opponent(board, rng):
        legal = sub.legal_moves(board)
        vals = np.array([-value(sub.play(board, a)) for a in legal])
        return int(rng.choice(legal[vals == vals.max()]))

    return opponent


@check("the trained agent loses at most 3 of 200 games against a random player and wins at least 165")
def test_beats_random(sub):
    train = has(sub, "train")
    params = trained(sub)
    wins, losses, draws = play_match(sub, params, random_opponent, 200, np.random.default_rng(1))
    if losses > 3 or wins < 165:
        fail(f"over 200 games against random play, expected at most 3 losses and at least 165 wins, "
             f"got {wins} wins, {losses} losses, {draws} draws")


@check("the trained agent holds a perfect minimax player to a draw in at least 13 of 20 games and never wins")
def test_against_minimax(sub):
    params = trained(sub)
    wins, losses, draws = play_match(sub, params, make_minimax(sub), 20, np.random.default_rng(2))
    if wins > 0:
        fail(f"a perfect player cannot be beaten, so {wins} wins means a bug in the game; got {wins} wins, "
             f"{losses} losses, {draws} draws")
    if draws < 13:
        fail(f"over 20 games against minimax, expected at least 13 draws, got {draws} draws, {losses} losses")

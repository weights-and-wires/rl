"""Starter for the AlphaZero-style self-play exercise.

Fill in the five functions and `train`. The game, the network, and the search are written for you."""

import numpy as np
import torch

torch.set_num_threads(1)

# --------------------------------------------------------------------------
# Tic-tac-toe, written for you. A board is an int8 vector of 9 cells, always
# seen by the player to move: +1 is that player's mark, -1 the opponent's.
# `play` returns the next board already flipped to the next player's view.
# --------------------------------------------------------------------------

LINES = np.array(
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
)


def legal_moves(board):
    """Indices of empty cells."""
    return np.flatnonzero(np.asarray(board) == 0)


def play(board, action):
    """Place the mover's mark at `action` and flip the view to the next player."""
    b = np.array(board, dtype=np.int8)
    b[action] = 1
    return -b


def terminal_value(board):
    """Value for the player to move: -1 if the opponent has three in a row, 0 for a full board, else None."""
    sums = np.asarray(board)[LINES].sum(axis=1)
    if (sums == -3).any():
        return -1.0
    if (sums == 3).any():
        return 1.0
    if (np.asarray(board) != 0).all():
        return 0.0
    return None


# --------------------------------------------------------------------------
# A tiny policy-value network: 18 inputs, 32 ReLU units, 9 logits, 1 tanh value.
# --------------------------------------------------------------------------


def features(boards):
    """One-hot planes: (B, 9) boards to (B, 18) features, mine then theirs."""
    boards = np.atleast_2d(boards)
    return np.concatenate([boards == 1, boards == -1], axis=1).astype(np.float64)


def init_params(seed=0, hidden=64):
    """Parameters [W1, b1, Wp, bp, Wv, bv] as float64 numpy arrays."""
    rng = np.random.default_rng(seed)
    return [
        rng.normal(0, 0.3, (18, hidden)),
        np.zeros(hidden),
        rng.normal(0, 0.1, (hidden, 9)),
        np.zeros(9),
        rng.normal(0, 0.1, (hidden, 1)),
        np.zeros(1),
    ]


def forward_np(params, x):
    """Logits (B, 9) and values (B,) from features (B, 18), in numpy."""
    W1, b1, Wp, bp, Wv, bv = params
    h = np.maximum(x @ W1 + b1, 0.0)
    return h @ Wp + bp, np.tanh(h @ Wv + bv)[:, 0]


def forward_torch(params, x):
    """The same network on torch tensors, so gradients flow to `params`."""
    W1, b1, Wp, bp, Wv, bv = params
    h = torch.relu(x @ W1 + b1)
    return h @ Wp + bp, torch.tanh(h @ Wv + bv)[:, 0]


def net_evaluate(params, board):
    """Priors over the 9 cells (zero on illegal moves, sum 1) and the value for the player to move."""
    logits, v = forward_np(params, features(board))
    legal = np.asarray(board) == 0
    z = np.where(legal, logits[0], -np.inf)
    p = np.exp(z - z.max())
    return p / p.sum(), float(v[0])


# --------------------------------------------------------------------------
# Search, written for you: pUCT as in chapter 042, with the root noise hook.
# --------------------------------------------------------------------------


def search(board, evaluate, n_sim, c_puct=6.0, rng=None, noise=None):
    """Run `n_sim` simulations from `board`. Returns the root visit counts N, shape (9,).

    `evaluate(board)` returns (priors, value). `noise` is None or (alpha, eps).
    The search is deterministic except for the root noise.
    """
    tree = {}
    board = np.asarray(board, dtype=np.int8)

    def expand(b, add_noise=False):
        p, v = evaluate(b)
        if add_noise:
            p = add_root_noise(p, b == 0, noise[0], noise[1], rng)
        tree[b.tobytes()] = [p, np.zeros(9), np.zeros(9)]
        return v

    expand(board, add_noise=noise is not None)
    if terminal_value(board) is not None:
        return np.zeros(9)
    for _ in range(n_sim):
        b, path = board, []
        while True:
            tv = terminal_value(b)
            if tv is not None:
                value = tv
                break
            key = b.tobytes()
            if key not in tree:
                value = expand(b)
                break
            P, N, W = tree[key]
            q = np.divide(W, N, out=np.zeros(9), where=N > 0)
            u = q + c_puct * P * np.sqrt(max(N.sum(), 1.0)) / (1.0 + N)
            u = np.where(b == 0, u, -np.inf)
            a = int(np.argmax(u))
            path.append((key, a))
            b = play(b, a)
        for key, a in reversed(path):
            value = -value
            tree[key][1][a] += 1
            tree[key][2][a] += value
    return tree[board.tobytes()][1]


def az_move(params, board, n_sim=100, c_puct=6.0):
    """The trained agent's move: the most visited root action, no noise."""
    N = search(board, lambda b: net_evaluate(params, b), n_sim, c_puct)
    return int(np.argmax(N))


def sgd_step(params, X, pi, z, lr, c):
    """One gradient step on `az_loss`, in place on the numpy `params`. Returns the loss."""
    tp = [torch.tensor(p, requires_grad=True) for p in params]
    logits, v = forward_torch(tp, torch.tensor(X))
    loss = az_loss(logits, v, torch.tensor(pi), torch.tensor(z), tp, c)
    loss.backward()
    with torch.no_grad():
        for p, t in zip(params, tp):
            p -= lr * t.grad.numpy()
    return float(loss.item())


# --------------------------------------------------------------------------
# The functions you write.
# --------------------------------------------------------------------------


def visit_count_policy(N, tau):
    """Turn root visit counts into a probability vector.

    Args:
        N: visit counts of the root's actions, shape (A,).
        tau: temperature. With tau > 0, pi is proportional to N ** (1 / tau).
            With tau == 0, pi puts all its mass on the most visited actions and
            splits it equally between ties.

    Returns:
        pi, shape (A,), non-negative and summing to 1. Counts of 0 stay at 0.
        The result stays finite for counts of 20000 and a tau of 0.01.
    """
    raise NotImplementedError("visit_count_policy")


def add_root_noise(prior, legal, alpha, eps, rng):
    """Mix Dirichlet noise into the root prior, over the legal moves only.

    Args:
        prior: network prior at the root, shape (9,), zero on illegal moves.
        legal: boolean mask of legal moves, shape (9,).
        alpha: Dirichlet concentration, the same for every legal move.
        eps: weight of the noise in [0, 1].
        rng: a numpy Generator. Draw exactly one sample,
            rng.dirichlet(alpha * np.ones(k)) with k the number of legal moves.

    Returns:
        (1 - eps) * prior + eps * noise, shape (9,), with noise 0 on illegal moves.
    """
    raise NotImplementedError("add_root_noise")


def az_loss(logits, v, pi, z, params, c):
    """The AlphaZero loss for a batch, as a torch scalar that supports backward().

    Args:
        logits: policy logits, shape (B, 9).
        v: predicted values in [-1, 1], shape (B,).
        pi: policy targets, shape (B, 9), each row summing to 1.
        z: game results for the player to move, shape (B,).
        params: list of parameter tensors for the L2 term.
        c: L2 weight.

    Returns:
        mean((z - v)^2) + mean over the batch of the cross-entropy between pi
        and softmax(logits) + c * the sum of every parameter's squared entries.
    """
    raise NotImplementedError("az_loss")


def dihedral_augment(board, pi):
    """The 8 symmetries of the square, applied to a board and its policy target together.

    Args:
        board: shape (9,), a 3 by 3 board in row-major order.
        pi: policy over the 9 cells, shape (9,).

    Returns:
        A list of 8 (board, pi) pairs, each of shape (9,). The first pair is the
        identity. Four rotations, each with and without a mirror image.
    """
    raise NotImplementedError("dihedral_augment")


def self_play_game(evaluate, n_sim, c_puct, tau_moves, alpha, eps, rng, board=None):
    """Play one game against itself and return its training examples.

    Args:
        evaluate: callable board -> (priors, value), as in `search`.
        n_sim: simulations per move.
        c_puct: exploration constant.
        tau_moves: sample the move from the tau = 1 visit distribution for the
            first `tau_moves` moves, then take the most visited action.
        alpha, eps: root noise, passed to `search` as noise=(alpha, eps).
        rng: a numpy Generator.
        board: optional starting board. Default is the empty board.

    Returns:
        A list with one (board, pi_target, z) per move played. pi_target is the
        tau = 1 visit distribution of that move's search, shape (9,). z is the
        result of the game for the player to move in that board: +1, 0, or -1.
    """
    raise NotImplementedError("self_play_game")


def train(n_iters=20, games_per_iter=12, n_sim=50, seed=0, steps=900, batch=64, lr=0.2, l2=1e-4,
          c_puct=6.0, tau_moves=3, alpha=0.5, eps=0.25):
    """Self-play, augment, and train. Returns the trained parameters.

    Each iteration plays `games_per_iter` self-play games with the current
    network, adds all 8 symmetries of every example to a replay buffer, keeps the
    newest 20000 entries, and takes `steps` minibatch steps with `sgd_step`.
    Use rng = np.random.default_rng(seed) and params = init_params(seed).
    """
    raise NotImplementedError("train")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = train(seed=0)
    wins = losses = 0
    for g in range(40):
        board, me = np.zeros(9, dtype=np.int8), g % 2
        turn = 0
        while terminal_value(board) is None:
            a = az_move(p, board) if turn % 2 == me else int(rng.choice(legal_moves(board)))
            board, turn = play(board, a), turn + 1
        r = terminal_value(board)
        mover_is_me = turn % 2 == me
        wins += int((r == -1) == (not mover_is_me))
        losses += int((r == -1) == mover_is_me)
    print(f"against a random player over 40 games: {wins} wins, {losses} losses")

"""Options and hierarchical RL: option models, SMDP planning, and option updates."""

from collections import deque, namedtuple

import numpy as np

# Sutton, Precup, and Singh (1999) four rooms. '#' is a wall, '.' is open.
LAYOUT = (
    "#############",
    "#.....#.....#",
    "#.....#.....#",
    "#...........#",
    "#.....#.....#",
    "#.....#.....#",
    "##.####.....#",
    "#.....###.###",
    "#.....#.....#",
    "#...........#",
    "#.....#.....#",
    "#.....#.....#",
    "#############",
)
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left
HALLWAYS = ((3, 6), (6, 2), (7, 9), (9, 6))
GOAL = (11, 11)
START = (1, 1)

FourRooms = namedtuple("FourRooms", "P R gamma cells index goal start room hallways")
Option = namedtuple("Option", "name init pi beta")


def four_rooms(slip=0.2, gamma=0.9):
    """The four-rooms task as arrays.

    States are the 104 open cells in row-major order (`cells[s]` is (row, col),
    `index[(row, col)]` is the state). Actions are 0 up, 1 right, 2 down, 3 left.
    A move into a wall, or a slip (probability `slip`), leaves the agent in place.
    Every step costs -1 except in the goal, which is absorbing and costs 0.
    `room[s]` is the room number 0 to 3 of a room cell and -1 for a hallway cell.
    """
    cells = [(r, c) for r, row in enumerate(LAYOUT) for c, ch in enumerate(row) if ch == "."]
    index = {cell: i for i, cell in enumerate(cells)}
    S, A = len(cells), 4
    P = np.zeros((S, A, S))
    R = -np.ones((S, A))
    goal = index[GOAL]
    for s, (r, c) in enumerate(cells):
        for a, (dr, dc) in enumerate(MOVES):
            nxt = index.get((r + dr, c + dc), s)
            P[s, a, nxt] += 1.0 - slip
            P[s, a, s] += slip
    P[goal] = 0.0
    P[goal, :, goal] = 1.0
    R[goal] = 0.0
    room = np.full(S, -1)
    for s, (r, c) in enumerate(cells):
        if (r, c) not in HALLWAYS:
            room[s] = 2 * (r > 6) + (c > 6)
    return FourRooms(P, R, gamma, cells, index, goal, index[START], room,
                     [index[h] for h in HALLWAYS])


def _distance_to(env, target):
    """Steps to `target` in the slip-free grid (breadth-first search)."""
    dist = np.full(len(env.cells), -1)
    dist[target] = 0
    queue = deque([target])
    while queue:
        s = queue.popleft()
        r, c = env.cells[s]
        for dr, dc in MOVES:
            t = env.index.get((r + dr, c + dc))
            if t is not None and dist[t] < 0:
                dist[t] = dist[s] + 1
                queue.append(t)
    return dist


def primitive_options(env):
    """One option per action: available everywhere, one step, then terminate."""
    S = len(env.cells)
    return [Option(f"a{a}", np.ones(S, bool), np.full(S, a), np.ones(S)) for a in range(4)]


def hallway_options(env):
    """Eight options, two per room: walk the shortest path to one of the room's hallways.

    Each option can start in any cell of its room and terminates at its hallway,
    at the goal, and everywhere outside its room.
    """
    S = len(env.cells)
    options = []
    for h in env.hallways:
        dist = _distance_to(env, h)
        pi = np.zeros(S, dtype=int)
        for s, (r, c) in enumerate(env.cells):
            if s == h:
                continue
            nbr = [dist[env.index[(r + dr, c + dc)]] if (r + dr, c + dc) in env.index else 10**6
                   for dr, dc in MOVES]
            pi[s] = int(np.argmin(nbr))
        hr, hc = env.cells[h]
        for room in sorted({env.room[env.index[(hr + dr, hc + dc)]] for dr, dc in MOVES
                            if (hr + dr, hc + dc) in env.index} - {-1}):
            init = (env.room == room) & (np.arange(S) != env.goal)
            beta = np.where(init, 0.0, 1.0)
            beta[h] = 1.0
            options.append(Option(f"room{room}->{env.cells[h]}", init, pi, beta))
    return options


def run_option(env, option, s, rng):
    """Run `option` from state s. Returns (discounted_return, k, s_end, terminated)."""
    ret, k = 0.0, 0
    while True:
        a = option.pi[s]
        ret += env.gamma**k * env.R[s, a]
        s = rng.choice(len(env.cells), p=env.P[s, a])
        k += 1
        if s == env.goal:
            return ret, k, s, True
        if rng.random() < option.beta[s]:
            return ret, k, s, False


def discounted_return(rewards, gamma):
    """The discounted sum of rewards[0], rewards[1], ... with weights 1, gamma, gamma^2, ..."""
    rewards = np.asarray(rewards, dtype=float)
    return float(np.sum(gamma ** np.arange(len(rewards)) * rewards))


def option_model(env, option):
    """The option's reward model R_o (shape (S,)) and discounted transition model P_o ((S, S)).

    R_o[s] is the expected discounted reward collected while the option runs from s.
    P_o[s, x] is the expected value of gamma^k on ending in state x after k steps.
    """
    S = len(env.cells)
    rows = np.arange(S)
    P_pi = env.P[rows, option.pi]
    r_pi = env.R[rows, option.pi]
    M = np.eye(S) - env.gamma * P_pi * (1.0 - option.beta)[None, :]
    R_o = np.linalg.solve(M, r_pi)
    P_o = np.linalg.solve(M, env.gamma * P_pi * option.beta[None, :])
    return R_o, P_o


def smdp_value_iteration(env, options, tol=1e-6, max_sweeps=10000):
    """Optimal values when every option in `options` is available wherever it can start.

    Each sweep sets V[s] = max over options o with init[s] of R_o[s] + P_o[s] @ V, and
    keeps V[env.goal] = 0. Starts from V = min(R) / (1 - gamma) in every other state, a
    lower bound on the true values, and stops after the first sweep whose largest
    change is below `tol`. Returns (V, sweeps), where sweeps counts every sweep
    including the last.
    """
    S = len(env.cells)
    models = [option_model(env, o) for o in options]
    q_rows = np.full((len(options), S), -np.inf)
    V = np.full(S, env.R.min() / (1.0 - env.gamma))
    V[env.goal] = 0.0
    for sweep in range(1, max_sweeps + 1):
        for i, (o, (R_o, P_o)) in enumerate(zip(options, models)):
            q_rows[i] = np.where(o.init, R_o + P_o @ V, -np.inf)
        V_new = q_rows.max(axis=0)
        V_new[env.goal] = 0.0
        done = np.abs(V_new - V).max() < tol
        V = V_new
        if done:
            return V, sweep
    return V, max_sweeps


def smdp_q_update(Q, s, o, ret, k, s2, terminated, avail, alpha, gamma):
    """One SMDP Q-learning update after option o ran k steps from s and ended in s2.

    `ret` is the option's discounted return, `avail[s2, o2]` says whether option o2 can
    start in s2, and Q has shape (S, O). Changes Q[s, o] in place and returns Q.
    """
    best = 0.0 if terminated else np.max(np.where(avail[s2], Q[s2], -np.inf))
    Q[s, o] += alpha * (ret + gamma**k * best - Q[s, o])
    return Q


def intra_option_q_update(Q, s, a, r, s2, terminated, options, alpha, gamma):
    """One intra-option Q-learning update after the primitive step (s, a, r, s2).

    Updates Q[s, o] for every option o that can start in s and would pick action a
    there. The target is r + gamma * U, where U = (1 - beta_o(s2)) Q[s2, o] +
    beta_o(s2) * max over options o2 that can start in s2 of Q[s2, o2], and U = 0 if
    the episode terminated. Q has shape (S, O). Changes Q in place and returns it.
    """
    avail2 = np.array([o.init[s2] for o in options])
    best = np.max(np.where(avail2, Q[s2], -np.inf))
    for i, o in enumerate(options):
        if o.init[s] and o.pi[s] == a:
            U = 0.0 if terminated else (1.0 - o.beta[s2]) * Q[s2, i] + o.beta[s2] * best
            Q[s, i] += alpha * (r + gamma * U - Q[s, i])
    return Q


def interrupted_beta(options, Q, o):
    """Option o's termination vector after interruption, shape (S,).

    The option also ends in every state where some other available option has a
    strictly higher value than continuing: max over options o2 that can start in s of
    Q[s, o2] > Q[s, o]. Elsewhere it keeps its own beta.
    """
    avail = np.array([op.init for op in options]).T
    best = np.max(np.where(avail, Q, -np.inf), axis=1)
    return np.where(best > Q[:, o], 1.0, options[o].beta)


def learn(env, options, mode, steps, alpha, epsilon, seed):
    """Epsilon-greedy option learning for `steps` primitive steps. Returns Q, shape (S, O).

    mode "smdp" runs each chosen option to termination and calls smdp_q_update once.
    mode "intra" calls intra_option_q_update after every primitive step. Episodes
    start in a uniformly random non-goal cell.
    """
    rng = np.random.default_rng(seed)
    S, O = len(env.cells), len(options)
    avail = np.array([o.init for o in options]).T
    Q = np.zeros((S, O))
    used = 0
    while used < steps:
        s = int(rng.choice([x for x in range(S) if x != env.goal]))
        done = False
        while not done and used < steps:
            masked = np.where(avail[s], Q[s], -np.inf)
            o = int(rng.choice(np.flatnonzero(avail[s]))) if rng.random() < epsilon else int(np.argmax(masked))
            opt = options[o]
            if mode == "smdp":
                ret, k, s2, done = run_option(env, opt, s, rng)
                smdp_q_update(Q, s, o, ret, k, s2, done, avail, alpha, env.gamma)
                used += k
                s = s2
            else:
                while True:
                    a = opt.pi[s]
                    s2 = int(rng.choice(S, p=env.P[s, a]))
                    done = s2 == env.goal
                    intra_option_q_update(Q, s, a, env.R[s, a], s2, done, options, alpha, env.gamma)
                    used += 1
                    s = s2
                    if done or rng.random() < opt.beta[s] or used >= steps:
                        break
    return Q


if __name__ == "__main__":
    env = four_rooms()
    prims, halls = primitive_options(env), hallway_options(env)
    print(f"{len(env.cells)} states, {len(halls)} hallway options, goal at {env.cells[env.goal]}")
    print("rewards [-1, -1, -1] at gamma 0.9 sum to", round(discounted_return([-1, -1, -1], 0.9), 4))
    R_o, P_o = option_model(env, halls[0])
    s0 = env.start
    print(f"{halls[0].name} from the start: R_o = {R_o[s0]:.4f}, sum of P_o = {P_o[s0].sum():.4f}")
    V_p, n_p = smdp_value_iteration(env, prims)
    V_o, n_o = smdp_value_iteration(env, prims + halls)
    print(f"V*(start) = {V_p[s0]:.4f} after {n_p} sweeps with primitives only")
    print(f"V*(start) = {V_o[s0]:.4f} after {n_o} sweeps with hallway options added")

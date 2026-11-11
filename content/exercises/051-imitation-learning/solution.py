"""Reference solution for the imitation learning exercise."""

import numpy as np

import rlenvs


def fit_bc(states, actions, n_states, n_actions):
    counts = np.zeros((n_states, n_actions))
    np.add.at(counts, (np.asarray(states, dtype=int), np.asarray(actions, dtype=int)), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    return np.where(totals > 0, counts / np.maximum(totals, 1.0), 1.0 / n_actions)


def add_mistakes(policy, eps):
    policy = np.asarray(policy, dtype=float)
    return (1.0 - eps) * policy + eps / policy.shape[1]


def collect(mdp, policy, n_episodes, max_steps, rng):
    visited = []
    for _ in range(n_episodes):
        s = int(rng.choice(mdp.n_states, p=mdp.mu0))
        for _ in range(max_steps):
            if mdp.terminal[s]:
                break
            visited.append(s)
            a = int(rng.choice(mdp.n_actions, p=policy[s]))
            s, _ = mdp.sample(s, a, rng)
    return np.array(visited, dtype=int)


def finite_horizon_return(mdp, pi, T):
    pi = np.asarray(pi, dtype=float)
    live = ~mdp.terminal
    P_pi = np.einsum("sa,sat->st", pi, mdp.P) * live[:, None]
    r_pi = (pi * mdp.R).sum(axis=1) * live
    d = mdp.mu0.copy()
    total = 0.0
    for _ in range(T):
        total += d @ r_pi
        d = d @ P_pi
    return float(total)


def disagreement(mdp, learner, expert, pi_dist):
    d = rlenvs.state_distribution(mdp, pi_dist)
    agree = (np.asarray(learner) * np.asarray(expert)).sum(axis=1)
    miss = np.where(mdp.terminal, 0.0, 1.0 - agree)
    return float(d @ miss)


def dagger(mdp, expert, eps, n_iters, n_episodes, max_steps, seed):
    rng = np.random.default_rng(seed)
    labels = expert.argmax(axis=1)
    data = np.zeros(0, dtype=int)
    policies = []
    runner = expert
    for _ in range(n_iters):
        visited = collect(mdp, runner, n_episodes, max_steps, rng)
        data = np.concatenate([data, visited])
        learner = fit_bc(data, labels[data], mdp.n_states, mdp.n_actions)
        policies.append(learner)
        runner = add_mistakes(learner, eps)
    return policies


# Shared worlds ----------------------------------------------------------------

LEDGE = [
    "S........G",
    "..........",
    "HHHHHHHHHH",
]


def ledge_world():
    """A deterministic ledge: the expert walks the top row, a safe row lies
    below it, and below that a row of holes that cost 1."""
    return rlenvs.gridworld(LEDGE, gamma=0.99, slip=0.0, step_reward=0.0, goal_reward=1.0,
                            hole_reward=-1.0, max_steps=200)


def expert_policy(mdp):
    """The optimal deterministic policy, from value iteration."""
    return rlenvs.greedy(rlenvs.q_from_v(mdp, rlenvs.value_iteration(mdp)))


def tightrope(eps, recover):
    """Two states: 0 is on the rope, 1 is off it, and every step off it pays -1.

    Action 0 keeps you on the rope, or climbs back on from off it. Action 1
    steps off, or stays off. Returns the MDP and a learner that picks action 1
    with probability eps on the rope and, off it, climbs back
    (recover=True) or never does (recover=False).
    """
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = P[0, 1, 1] = P[1, 0, 0] = P[1, 1, 1] = 1.0
    R = np.array([[0.0, 0.0], [-1.0, -1.0]])
    mdp = rlenvs.TabularMDP(P=P, R=R, gamma=0.99, mu0=np.array([1.0, 0.0]))
    pi = np.array([[1.0 - eps, eps], [1.0, 0.0] if recover else [0.0, 1.0]])
    return mdp, pi


if __name__ == "__main__":
    print("Tightrope with eps = 0.01: expected cost over T steps")
    for T in (10, 100, 1000):
        mdp, stuck = tightrope(0.01, recover=False)
        _, back = tightrope(0.01, recover=True)
        print(f"  T = {T:4d}: never recovers {-finite_horizon_return(mdp, stuck, T):7.2f}, "
              f"recovers {-finite_horizon_return(mdp, back, T):5.2f}")

    eps = 0.1
    mdp = ledge_world()
    expert = expert_policy(mdp)
    start = mdp.names["start"]

    def value(pi):
        return rlenvs.policy_evaluation(mdp, pi)[start]

    print(f"\nLedge world, learner mistake rate {eps}")
    print(f"  expert {value(expert):.4f}, expert making the same mistakes "
          f"{value(add_mistakes(expert, eps)):.4f}")
    policies = dagger(mdp, expert, eps, n_iters=15, n_episodes=20, max_steps=200, seed=0)
    for i in (0, 1, 2, 4, 14):
        pi = policies[i]
        acting = add_mistakes(pi, eps)
        name = "BC       " if i == 0 else f"DAgger {i + 1:2d}"
        print(f"  {name}: value {value(acting):.4f}, error on expert states "
              f"{disagreement(mdp, pi, expert, expert):.4f}, on its own states "
              f"{disagreement(mdp, pi, expert, acting):.4f}")

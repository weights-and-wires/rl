"""Imitation learning: behaviour cloning, compounding error, and DAgger.

Fill in the functions below, in order. You fit a policy to an expert's
actions (behaviour cloning), measure exactly how small mistakes compound over
a horizon, and then fix the compounding with DAgger, which asks the expert to
label the states the learner itself reaches.

Every learned policy in this exercise makes mistakes: when it acts, it takes
a uniformly random action with probability eps. The expert never does. This
stands in for the small error any real learner has, even on states it
trained on.

A policy for a tabular MDP is an array of shape (S, A) whose rows sum to one.
"""

import numpy as np

import rlenvs


def fit_bc(states, actions, n_states, n_actions):
    """Behaviour cloning for a tabular policy: count, then normalise.

    Args:
        states: visited states, int array of shape (n,).
        actions: the expert's action in each, int array of shape (n,).
        n_states, n_actions: S and A.

    Returns:
        A policy of shape (S, A). Row s is the empirical distribution of the
        labelled actions in state s, or uniform if s never appears.
    """
    raise NotImplementedError("fit_bc")


def add_mistakes(policy, eps):
    """A policy that follows `policy` but acts uniformly at random with probability eps.

    Returns:
        (1 - eps) * policy + eps / A, shape (S, A).
    """
    raise NotImplementedError("add_mistakes")


def collect(mdp, policy, n_episodes, max_steps, rng):
    """Roll out a policy and record the states it visits.

    For each episode, draw the start state with rng.choice(S, p=mdp.mu0).
    Then, for at most max_steps steps: stop if the state is terminal; else
    record it, draw an action with rng.choice(A, p=policy[s]), and move with
    s, _ = mdp.sample(s, a, rng).

    Returns:
        An int array of every recorded state, in order. Terminal states are
        never recorded.
    """
    raise NotImplementedError("collect")


def finite_horizon_return(mdp, pi, T):
    """The exact expected sum of the first T rewards, undiscounted, from mdp.mu0.

    Push the state distribution forward T times: d_0 = mu0, and
    d_{t+1} = d_t P_pi. Add d_t @ r_pi at each step. Terminal states pay
    nothing and lead nowhere: zero their rows of P_pi and their entries of r_pi.

    Returns:
        A float, sum over t = 0..T-1 of E[r_t].
    """
    raise NotImplementedError("finite_horizon_return")


def disagreement(mdp, learner, expert, pi_dist):
    """How often the learner would choose differently from the expert, on pi_dist's states.

    Weight states by rlenvs.state_distribution(mdp, pi_dist), the normalised
    discounted visitation of pi_dist. In state s, the chance the learner's
    action differs from the deterministic expert's is
    1 - sum_a learner[s, a] * expert[s, a]. Count terminal states as 0.

    Returns:
        A float in [0, 1].
    """
    raise NotImplementedError("disagreement")


def dagger(mdp, expert, eps, n_iters, n_episodes, max_steps, seed):
    """Dataset aggregation.

    Make rng = np.random.default_rng(seed) and pass it to every collect call.
    Iteration 1 collects n_episodes rollouts of the expert itself. Every later
    iteration collects n_episodes rollouts of add_mistakes(latest learner, eps).
    Each iteration labels the new states with the expert's action
    (expert.argmax(axis=1)), adds them to one growing dataset, and fits a new
    learner with fit_bc on the whole dataset.

    Returns:
        A list of n_iters learned policies, each of shape (S, A), without the
        added mistakes. The first is plain behaviour cloning on expert data.
    """
    raise NotImplementedError("dagger")


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

"""SARSA and Q-learning on the cliff walk.

Fill in the functions below, in order. Both learners play their own episodes
with one np.random.default_rng(seed) and return the action-value table `Q`
with shape `(S, A)`, together with the total reward of each training episode.
"""

import numpy as np

import rlenvs


def epsilon_greedy_action(q_row, epsilon, rng):
    """Return an epsilon-greedy action from one row of `Q`, as an `int`.

    With probability `epsilon`, draw an action uniformly. Otherwise take a
    greedy action, and break ties between equal maxima at random. Draw
    `rng.random()` first to decide whether to explore.
    """
    raise NotImplementedError("explore with probability epsilon, otherwise pick a random best action")


def sarsa_update(Q, s, a, r, s2, a2, terminated, alpha, gamma):
    """Update `Q[s, a]` in place toward the SARSA target, and return `Q`.

    The target is `r + gamma * Q[s2, a2]`, using `a2`, the action the policy
    will take next. When `terminated` is true the target is `r` alone.
    """
    raise NotImplementedError("move Q[s, a] a fraction alpha toward r + gamma * Q[s2, a2]")


def q_learning_update(Q, s, a, r, s2, terminated, alpha, gamma):
    """Update `Q[s, a]` in place toward the Q-learning target, and return `Q`.

    The target is `r + gamma * max(Q[s2])`, whatever action is taken next.
    When `terminated` is true the target is `r` alone.
    """
    raise NotImplementedError("move Q[s, a] a fraction alpha toward r + gamma * max(Q[s2])")


def sarsa(mdp, n_episodes, alpha, epsilon, seed):
    """Learn `Q` by SARSA. Returns `(Q, returns)`, with `returns` of shape `(n_episodes,)`.

    Start every `Q` entry at zero. Choose the first action before the step
    loop. After each step, choose the next action `a2` with
    `epsilon_greedy_action` before you update, update with `sarsa_update`, and
    then take `a2`. An episode ends when `mdp.terminal[s2]` is true or after
    `mdp.max_steps` steps.
    """
    raise NotImplementedError("choose, step, choose again, update, and carry a2 into the next step")


def q_learning(mdp, n_episodes, alpha, epsilon, seed):
    """Learn `Q` by Q-learning. Returns `(Q, returns)`, with `returns` of shape `(n_episodes,)`.

    Start every `Q` entry at zero. In each step, choose an action with
    `epsilon_greedy_action`, step with `mdp.sample(s, a, rng)`, and update
    with `q_learning_update`. An episode ends when `mdp.terminal[s2]` is true
    or after `mdp.max_steps` steps.
    """
    raise NotImplementedError("choose, step, update, repeat")


if __name__ == "__main__":
    cliff = rlenvs.cliff_walking()
    start = cliff.names["start"]
    for name, learn in (("SARSA", sarsa), ("Q-learning", q_learning)):
        Q, returns = learn(cliff, n_episodes=500, alpha=0.1, epsilon=0.1, seed=0)
        v = rlenvs.policy_evaluation(cliff, rlenvs.greedy(Q))
        print(
            f"{name:>10}: greedy value from the start {v[start]:7.2f}, "
            f"mean training return over the last 100 episodes {returns[-100:].mean():7.2f}"
        )

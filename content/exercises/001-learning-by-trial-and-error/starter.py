"""The agent-environment loop.

Fill in the four functions in order. Together they build a random agent, run
it through a small room one step at a time, and log what happened.

The environment is an `rlenvs.TabularMDP`. `mdp.reset(seed)` starts an
episode and returns the first state. `mdp.step(a)` returns
`(next_state, reward, terminated, truncated)`. A policy `pi` is an array of
shape (n_states, n_actions) whose row `pi[s]` holds the probabilities of each
action in state `s`.
"""

import numpy as np

import rlenvs

# A 3 by 3 room with a pillar in the middle. Every step costs 1, and the
# episode ends on reaching G in the far corner.
ROOM = [
    "S..",
    ".#.",
    "..G",
]


def uniform_random_policy(n_states, n_actions):
    """The policy that picks every action with the same probability.

    Args:
        n_states: the number of states.
        n_actions: the number of actions.

    Returns:
        Shape (n_states, n_actions). Every entry is 1 / n_actions, so each row
        sums to one.
    """
    raise NotImplementedError("uniform_random_policy")


def run_episode(mdp, pi, rng):
    """Run one episode of policy `pi` in `mdp` and log every step.

    Start with `mdp.reset(seed=int(rng.integers(2**31)))`, so the whole episode
    is reproducible from `rng`. In each state s, draw an action from `pi[s]`
    with `rng`, then call `mdp.step`. Stop when the step reports terminated or
    truncated.

    Args:
        mdp: an `rlenvs.TabularMDP`.
        pi: the policy, shape (n_states, n_actions).
        rng: a numpy Generator.

    Returns:
        A tuple `(states, actions, rewards, terminated)`. `states[t]` is the
        state in which `actions[t]` was taken and `rewards[t]` received, so all
        three arrays have shape (T,). `terminated` is True if the episode ended
        in a terminal state and False if it hit the time limit.
    """
    raise NotImplementedError("run_episode")


def visit_counts(states, n_states):
    """Count how many times an episode was in each state.

    Args:
        states: the states of one or more episodes, an int array of any length.
        n_states: the number of states in the MDP.

    Returns:
        Shape (n_states,). Entry s is how many times `s` appears in `states`,
        including states that never appear (count 0).
    """
    raise NotImplementedError("visit_counts")


def mean_episode_length(mdp, pi, n_episodes, seed):
    """The average number of steps in an episode of policy `pi`.

    Make one Generator with `np.random.default_rng(seed)` and pass it to
    `run_episode` for every episode. Do not make a new generator per episode.

    Args:
        mdp: an `rlenvs.TabularMDP`.
        pi: the policy, shape (n_states, n_actions).
        n_episodes: how many episodes to run.
        seed: an int.

    Returns:
        The mean episode length, a float.
    """
    raise NotImplementedError("mean_episode_length")


if __name__ == "__main__":
    mdp = rlenvs.gridworld(ROOM, gamma=0.9)
    pi = uniform_random_policy(mdp.n_states, mdp.n_actions)
    rng = np.random.default_rng(0)
    states, actions, rewards, terminated = run_episode(mdp, pi, rng)
    print(f"one episode of the random agent: {len(rewards)} steps, terminated={terminated}")
    print("  first five steps (state, action, reward):")
    for t in range(min(5, len(rewards))):
        print(f"    t={t}: s={states[t]}, a={actions[t]}, r={rewards[t]:.0f}")
    print("  visits per state:", visit_counts(states, mdp.n_states))
    print(f"  mean length over 500 episodes: {mean_episode_length(mdp, pi, 500, seed=1):.1f}")

"""Exploration: count bonuses, optimism, posterior sampling, and random network distillation.

Fill in the functions below, in order. Three agents share one model-based loop
on RiverSwim. They count transitions, build a model, plan in it, and act. They
differ only in how they explore: epsilon-greedy dithering, a count bonus that
makes rarely tried actions look better (optimism), or planning in a model
sampled from the posterior (posterior sampling). Then you build random network
distillation, a novelty bonus for states too many to count.

RiverSwim has 6 states in a row and 2 actions: 0 swims left with the current,
1 swims right against it. Swimming right in state 5 pays 1. Swimming left in
state 0 pays 0.005.
"""

import numpy as np
import torch

import rlenvs

torch.set_num_threads(1)


def count_bonus(counts, beta):
    """The exploration bonus beta / sqrt(N) for every entry of counts.

    Args:
        counts: visit counts N(s, a), any shape.
        beta: the bonus scale.

    Returns:
        beta / sqrt(max(N, 1)), the same shape. An unvisited pair gets the
        largest bonus, beta.
    """
    raise NotImplementedError("count_bonus")


def empirical_model(transition_counts, reward_sums):
    """The maximum-likelihood model from counts.

    Args:
        transition_counts: C[s, a, s2], how often (s, a) led to s2, shape (S, A, S).
        reward_sums: the total reward received after each (s, a), shape (S, A).

    Returns:
        (P, R, N): P of shape (S, A, S) with P[s, a] = C[s, a] / N(s, a); R of
        shape (S, A), the mean reward; and N of shape (S, A), the visit count
        C[s, a].sum(). A pair never visited gets P[s, a, s] = 1 (it stays put)
        and R[s, a] = 0.
    """
    raise NotImplementedError("empirical_model")


def plan(P, R, gamma, n_iters=500, tol=1e-8):
    """Value iteration on action values.

    Args:
        P: shape (S, A, S). R: shape (S, A). gamma: the discount.
        n_iters: the most sweeps to run. tol: stop once no entry moves more than this.

    Returns:
        Q, shape (S, A), iterating Q <- R + gamma * P @ max_a Q from zeros.
    """
    raise NotImplementedError("plan")


def sample_posterior_mdp(transition_counts, reward_sums, rng, prior=1.0):
    """Draw one plausible model from the posterior.

    Transitions: each row P[s, a] ~ Dirichlet(C[s, a] + prior). Draw it as
    independent gamma variables g = rng.standard_gamma(C + prior), normalised
    along the last axis.

    Rewards: a Gaussian prior Normal(0, 1) and unit observation noise give the
    posterior Normal(reward_sums / (N + 1), 1 / (N + 1)), so draw
    R = reward_sums / (N + 1) + rng.standard_normal((S, A)) / sqrt(N + 1).

    Returns:
        (P, R), shapes (S, A, S) and (S, A).
    """
    raise NotImplementedError("sample_posterior_mdp")


def explore(mdp, method, steps, seed, beta=0.5, epsilon=0.1, replan_every=20):
    """Run one exploring agent on a tabular MDP for a fixed number of steps.

    Start with rng = np.random.default_rng(seed) and s = mdp.reset(seed=seed),
    and zero counts. At step t, when t % replan_every == 0, compute Q:
      - "epsilon": Q = plan on the empirical model.
      - "optimism": Q = plan on the empirical model with R + count_bonus(N, beta).
      - "posterior": Q = plan on a model from sample_posterior_mdp(..., rng).
    Act greedily on Q, except that "epsilon" takes a uniformly random action
    with probability epsilon. Record the transition and reward in the counts,
    and call mdp.reset() when a step reports terminated or truncated.

    Returns:
        (policy, counts, total_reward). policy is rlenvs.greedy of plan on the
        final empirical model, with no bonus: what the agent believes is best.
        counts has shape (S, A, S). total_reward is the sum collected.
    """
    raise NotImplementedError("explore")


def make_rnd(in_dim, out_dim, hidden, seed):
    """Build the target and predictor networks for random network distillation.

    Call torch.manual_seed(seed), then build the target, then the predictor.
    Each is torch.nn.Sequential(Linear(in_dim, hidden), ReLU(),
    Linear(hidden, out_dim)). The target is never trained: turn off
    requires_grad on its parameters.

    Returns:
        (target, predictor).
    """
    raise NotImplementedError("make_rnd")


def rnd_bonus(target, predictor, states):
    """The RND novelty bonus for each state.

    Args:
        target, predictor: networks from make_rnd.
        states: a float tensor, shape (n, in_dim).

    Returns:
        A tensor of shape (n,): the squared Euclidean distance between the
        predictor's and the target's outputs, computed without gradients.
    """
    raise NotImplementedError("rnd_bonus")


def rnd_train(target, predictor, states, steps, lr):
    """Train the predictor to match the target on the given states.

    Use torch.optim.Adam on the predictor's parameters with learning rate lr.
    Each of the `steps` updates is one full-batch step on the mean over states
    of the squared distance between the two networks' outputs.

    Returns:
        The last loss, as a float.
    """
    raise NotImplementedError("rnd_train")


if __name__ == "__main__":
    mdp = rlenvs.river_swim()
    j_star = mdp.mu0 @ rlenvs.value_iteration(mdp)
    print(f"RiverSwim, 3000 steps. The optimal policy is worth {j_star:.3f} from the start.")
    for method in ("epsilon", "optimism", "posterior"):
        pi, counts, total = explore(mdp, method, steps=3000, seed=0)
        j = mdp.mu0 @ rlenvs.policy_evaluation(mdp, pi)
        far = counts[-1].sum()
        print(f"  {method:9s}: reward {total:6.1f}, visits to the far state {far:5.0f}, "
              f"final policy worth {j:.3f}")

    target, predictor = make_rnd(2, 8, 32, seed=0)
    gen = torch.Generator().manual_seed(0)
    seen = torch.rand(256, 2, generator=gen) * 0.5
    novel = torch.rand(256, 2, generator=gen) * 0.5 + 0.5
    rnd_train(target, predictor, seen, steps=300, lr=1e-2)
    print(f"RND bonus on visited states {rnd_bonus(target, predictor, seen).mean():.4f}, "
          f"on novel states {rnd_bonus(target, predictor, novel).mean():.4f}")

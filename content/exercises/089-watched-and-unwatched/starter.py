"""A policy that behaves well when it is watched.

One decision, repeated. Each rollout has a hidden context, eval or deploy, and
the policy picks one of two actions:

    0  comply   do the task as asked. Reward `comply_reward` in every context.
    1  hack     take the shortcut. Reward `hack_reward`, minus `lam` if the
                context is eval, because only evaluation checks for the hack.

Before it acts, the policy sees one observation:

    0  eval cue     shown only in an eval context
    1  deploy cue   shown only in a deploy context
    2  no cue       shown in either context

With probability q the true cue appears. Otherwise the policy sees "no cue".
Training can only penalize the hack where it is visible, so the best policy
depends on what the observation says about the context. You compute that
policy, the gap it opens between eval and deploy, and the statistics that say
how many runs you need to see the gap.

Observations index axis 0 of every table below, in the order above. Contexts
index axis 1, in the order eval, deploy.
"""

from statistics import NormalDist

import numpy as np

COMPLY, HACK = 0, 1
EVAL_CUE, DEPLOY_CUE, NO_CUE = 0, 1, 2
EVAL, DEPLOY = 0, 1


def joint_obs(prior_eval, q):
    """The joint probability of each (observation, context) pair.

    The context is eval with probability `prior_eval`. With probability q the
    cue reveals it, and with probability 1 - q the policy sees "no cue", in
    either context.

    Args:
        prior_eval: P(context = eval), between 0 and 1.
        q: the probability that the cue reveals the context, between 0 and 1.

    Returns:
        Array of shape (3, 2) that sums to 1. Row 0 is the eval cue, row 1 the
        deploy cue, and row 2 no cue. Column 0 is eval, column 1 is deploy.
    """
    raise NotImplementedError("joint_obs")


def posterior_eval(joint, prior_eval):
    """P(context = eval | observation) for each observation.

    Args:
        joint: the (3, 2) table from `joint_obs`.
        prior_eval: returned for an observation whose row sums to 0, because
            Bayes' rule would divide by 0 there.

    Returns:
        Array of shape (3,).
    """
    raise NotImplementedError("posterior_eval")


def action_rewards(w, comply_reward, hack_reward, lam):
    """The expected reward of each action, given each observation.

    Complying pays `comply_reward` everywhere. Hacking pays `hack_reward`, and
    costs `lam` only when the context is eval, so given an observation whose
    posterior probability of eval is w, the expected penalty is lam * w.

    Args:
        w: posterior P(eval | observation), shape (3,).
        comply_reward: the reward for complying.
        hack_reward: the reward for hacking, before any penalty.
        lam: the penalty for a hack in an eval context.

    Returns:
        Array of shape (3, 2). Column 0 is comply, column 1 is hack.
    """
    raise NotImplementedError("action_rewards")


def optimal_hack(R):
    """The reward-maximizing policy: P(hack | observation) as 0.0 or 1.0.

    Args:
        R: the (3, 2) expected rewards from `action_rewards`.

    Returns:
        Array of shape (3,): 1.0 where hacking earns strictly more than
        complying, and 0.0 otherwise (a tie goes to complying).
    """
    raise NotImplementedError("optimal_hack")


def kl_hack(R, ref_hack, beta):
    """P(hack | observation) at the optimum of E[reward] - beta * KL(pi || ref).

    The reference policy hacks with probability `ref_hack` after every
    observation. With two actions, the optimum is pi proportional to
    ref * exp(reward / beta), which is a sigmoid.

    Args:
        R: the (3, 2) expected rewards from `action_rewards`.
        ref_hack: the reference policy's hack probability, strictly between 0 and 1.
        beta: the KL coefficient, greater than 0.

    Returns:
        Array of shape (3,).
    """
    raise NotImplementedError("kl_hack")


def hack_rates(joint, pol):
    """The hack rate in each context.

    Args:
        joint: the (3, 2) table from `joint_obs`.
        pol: P(hack | observation), shape (3,).

    Returns:
        (eval_rate, deploy_rate): P(hack | context = eval) and
        P(hack | context = deploy), as floats. Condition on the context, so
        divide each column of `joint` by its sum first.
    """
    raise NotImplementedError("hack_rates")


def gap_closed_form(q, pol):
    """The compliance gap, deploy hack rate minus eval hack rate, in closed form.

    Uses only q and the hack probabilities after the eval cue and the deploy
    cue. Rollouts with no cue behave the same in both contexts.

    Args:
        q: the probability that the cue reveals the context.
        pol: P(hack | observation), shape (3,).

    Returns:
        The gap as a float.
    """
    raise NotImplementedError("gap_closed_form")


def bootstrap_diff_replicates(a, b, n_boot, seed):
    """Bootstrap replicates of mean(a) - mean(b).

    The two groups are different runs, so each replicate redraws a with
    replacement and, separately, b with replacement. Make one generator from
    `seed` and use it for every draw.

    Args:
        a, b: 1-D arrays of scores. They can differ in length.
        n_boot: the number of replicates.
        seed: the generator seed.

    Returns:
        Array of shape (n_boot,).
    """
    raise NotImplementedError("bootstrap_diff_replicates")


def percentile_ci(replicates, confidence=0.95):
    """The percentile interval of a set of bootstrap replicates.

    Returns:
        (low, high) as floats: the replicates' tail percentiles, such as the
        2.5th and 97.5th for confidence 0.95.
    """
    raise NotImplementedError("percentile_ci")


def runs_needed(p1, p2, alpha=0.05, power=0.8):
    """The runs per condition that separate two rates, rounded up to an integer.

    Use the normal approximation for a two-sided test at level alpha with the
    given power, on two independent proportions. The lesson derives it.
    `statistics.NormalDist().inv_cdf` gives the z values.
    """
    raise NotImplementedError("runs_needed")


def sandbag_gap(cue_scores, nocue_scores, n_boot=2000, seed=0):
    """How much lower a model scores when a capability-eval cue is present.

    Args:
        cue_scores: 1-D scores on prompts that carry the cue.
        nocue_scores: 1-D scores on matched prompts without it.
        n_boot, seed: passed to the bootstrap.

    Returns:
        (gap, low, high): mean(nocue_scores) - mean(cue_scores), and a 95%
        percentile interval for it.
    """
    raise NotImplementedError("sandbag_gap")


if __name__ == "__main__":
    comply_reward, hack_reward, lam, prior_eval, q = 0.6, 1.0, 1.0, 0.5, 0.6
    joint = joint_obs(prior_eval, q)
    w = posterior_eval(joint, prior_eval)
    R = action_rewards(w, comply_reward, hack_reward, lam)
    for name, pol in [("optimal", optimal_hack(R)), ("KL, beta=0.2", kl_hack(R, 0.5, 0.2))]:
        e, d = hack_rates(joint, pol)
        print(f"{name:13s} hack|cue = {np.round(pol, 4)}  eval {e:.4f}  deploy {d:.4f}  gap {d - e:.4f}")
    print("runs per condition to separate 0.30 from 0.10:", runs_needed(0.30, 0.10))
    rng = np.random.default_rng(0)
    deploy_runs = (rng.random(12) < 0.30).astype(float)
    eval_runs = (rng.random(12) < 0.10).astype(float)
    reps = bootstrap_diff_replicates(deploy_runs, eval_runs, 4000, seed=1)
    print("12 runs each:", deploy_runs.mean() - eval_runs.mean(), "interval", percentile_ci(reps))

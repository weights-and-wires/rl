"""Reference solution for the watched-and-unwatched exercise."""

from statistics import NormalDist

import numpy as np

COMPLY, HACK = 0, 1
EVAL_CUE, DEPLOY_CUE, NO_CUE = 0, 1, 2
EVAL, DEPLOY = 0, 1


def joint_obs(prior_eval, q):
    """P(observation, context) as a (3, 2) array."""
    prior = np.array([prior_eval, 1.0 - prior_eval])
    joint = np.zeros((3, 2))
    joint[EVAL_CUE, EVAL] = q * prior[EVAL]
    joint[DEPLOY_CUE, DEPLOY] = q * prior[DEPLOY]
    joint[NO_CUE] = (1.0 - q) * prior
    return joint


def posterior_eval(joint, prior_eval):
    """P(context = eval | observation), with the prior for an impossible observation."""
    total = joint.sum(axis=1)
    safe = np.where(total > 0, total, 1.0)
    return np.where(total > 0, joint[:, EVAL] / safe, prior_eval)


def action_rewards(w, comply_reward, hack_reward, lam):
    """Expected reward of each action per observation, shape (3, 2)."""
    w = np.asarray(w, dtype=float)
    return np.stack([np.full_like(w, comply_reward), hack_reward - lam * w], axis=1)


def optimal_hack(R):
    """1.0 where hacking strictly beats complying, else 0.0."""
    return (R[:, HACK] > R[:, COMPLY]).astype(float)


def kl_hack(R, ref_hack, beta):
    """P(hack | observation) at the KL-regularized optimum."""
    logit = np.log(ref_hack / (1.0 - ref_hack)) + (R[:, HACK] - R[:, COMPLY]) / beta
    return 1.0 / (1.0 + np.exp(-logit))


def hack_rates(joint, pol):
    """(eval hack rate, deploy hack rate)."""
    by_context = joint / joint.sum(axis=0, keepdims=True)
    rates = pol @ by_context
    return float(rates[EVAL]), float(rates[DEPLOY])


def gap_closed_form(q, pol):
    """The compliance gap from the two cue-revealed policies alone."""
    return float(q * (pol[DEPLOY_CUE] - pol[EVAL_CUE]))


def bootstrap_diff_replicates(a, b, n_boot, seed):
    """Replicates of mean(a) - mean(b), redrawing each group on its own."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    a_means = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    b_means = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    return a_means - b_means


def percentile_ci(replicates, confidence=0.95):
    """Percentile interval from bootstrap replicates."""
    tail = 100.0 * (1.0 - confidence) / 2.0
    low, high = np.percentile(replicates, [tail, 100.0 - tail])
    return float(low), float(high)


def runs_needed(p1, p2, alpha=0.05, power=0.8):
    """Runs per condition to separate rates p1 and p2, rounded up."""
    nd = NormalDist()
    z = nd.inv_cdf(1.0 - alpha / 2.0) + nd.inv_cdf(power)
    n = z**2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2
    return int(np.ceil(n))


def sandbag_gap(cue_scores, nocue_scores, n_boot=2000, seed=0):
    """(gap, low, high): mean no-cue score minus mean cue score, with an interval."""
    reps = bootstrap_diff_replicates(nocue_scores, cue_scores, n_boot, seed)
    low, high = percentile_ci(reps)
    gap = float(np.mean(nocue_scores) - np.mean(cue_scores))
    return gap, low, high


if __name__ == "__main__":
    comply_reward, hack_reward, lam, prior_eval, q = 0.6, 1.0, 1.0, 0.5, 0.6
    joint = joint_obs(prior_eval, q)
    R = action_rewards(posterior_eval(joint, prior_eval), comply_reward, hack_reward, lam)
    for name, pol in [("optimal", optimal_hack(R)), ("KL, beta=0.2", kl_hack(R, 0.5, 0.2))]:
        e, d = hack_rates(joint, pol)
        print(f"{name:13s} hack|cue = {np.round(pol, 4)}  eval {e:.4f}  deploy {d:.4f}  gap {d - e:.4f}")
    print("runs per condition to separate 0.30 from 0.10:", runs_needed(0.30, 0.10))
    rng = np.random.default_rng(0)
    deploy = (rng.random(12) < 0.30).astype(float)
    evalr = (rng.random(12) < 0.10).astype(float)
    reps = bootstrap_diff_replicates(deploy, evalr, 4000, seed=1)
    print("12 runs each:", deploy.mean() - evalr.mean(), "interval", percentile_ci(reps))

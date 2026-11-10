"""Natural gradients, conjugate gradient, and TRPO.

Fill in the functions below. The policy is a linear softmax over a small
contextual bandit, but every function you write takes a `logits_fn` and a flat
parameter vector, so the same code works for any network. Everything uses
float64 torch tensors, so the checks can compare against exact matrices.

`make_contextual_bandit`, `flat_grad`, `sample_batch`, and `expected_reward`
are given.
"""

import numpy as np
import torch

torch.set_num_threads(1)
DTYPE = torch.float64


def make_contextual_bandit(n_contexts=8, n_features=4, n_actions=3, seed=0):
    """A small contextual bandit and a linear softmax policy over it. Given.

    Returns:
        features: shape (C, D), one feature vector per context.
        rewards: shape (C, A), the deterministic reward of each action in each
            context.
        logits_fn: a function mapping flat parameters, shape (A * D,), to
            logits for every context, shape (C, A).
    """
    rng = np.random.default_rng(seed)
    features = torch.tensor(rng.normal(size=(n_contexts, n_features)), dtype=DTYPE)
    rewards = torch.tensor(rng.normal(size=(n_contexts, n_actions)), dtype=DTYPE)

    def logits_fn(params):
        return features @ params.view(n_actions, n_features).T

    return features, rewards, logits_fn


def flat_grad(y, params, create_graph=False):
    """The gradient of scalar y with respect to params, as a flat vector. Given."""
    (g,) = torch.autograd.grad(y, params, create_graph=create_graph)
    return g.reshape(-1)


def mean_kl(logits_old, logits_new):
    """The mean over rows of KL(pi_old || pi_new) for categorical policies.

    Args:
        logits_old: shape (N, A), the logits of the old policy in N states.
        logits_new: shape (N, A), the logits of the new policy in the same
            states.

    Returns:
        A scalar tensor: the average over the N rows of
        sum_a pi_old(a) * (log pi_old(a) - log pi_new(a)). Keep it
        differentiable with respect to logits_new.
    """
    raise NotImplementedError("mean_kl")


def fisher_vector_product(logits_fn, params, v, damping=0.0):
    """(F + damping * I) v, where F is the Fisher matrix of the policy at params.

    Compute it without building F: F v is the gradient of
    (grad KL(pi_old || pi_theta)) . v, evaluated at theta = params, with
    pi_old held fixed at the policy for params.

    Args:
        logits_fn: maps flat parameters, shape (P,), to logits, shape (N, A).
        params: the current parameters, shape (P,).
        v: a vector, shape (P,).
        damping: added times v to the result.

    Returns:
        A tensor of shape (P,).
    """
    raise NotImplementedError("fisher_vector_product")


def conjugate_gradient(Avp, b, iters=10, tol=1e-10):
    """Approximately solve A x = b for a symmetric positive definite A.

    Uses only products with A. Start from x = 0, and stop after `iters`
    iterations or once the squared residual norm r . r is at most `tol`.

    Args:
        Avp: a function mapping a vector p, shape (P,), to A p, shape (P,).
        b: the right-hand side, shape (P,).
        iters: the largest number of iterations, one call to Avp each.
        tol: the stopping threshold for r . r.

    Returns:
        x, shape (P,).
    """
    raise NotImplementedError("conjugate_gradient")


def surrogate(logits, actions, advantages, logp_old):
    """The importance-weighted surrogate objective.

    mean_i [ pi_theta(a_i | s_i) / pi_old(a_i | s_i) * A_i ].

    Args:
        logits: the current policy's logits in each sampled state, shape (N, A).
        actions: the sampled actions, shape (N,), integer tensor.
        advantages: the advantage estimates, shape (N,).
        logp_old: log pi_old(a_i | s_i), recorded when the actions were
            sampled, shape (N,).

    Returns:
        A scalar tensor, differentiable with respect to logits.
    """
    raise NotImplementedError("surrogate")


def natural_step(g, fvp, max_kl, cg_iters=10):
    """The largest step along F^-1 g whose quadratic KL estimate equals max_kl.

    Solve F x = g with conjugate_gradient, then scale x so that
    0.5 * s^T F s = max_kl.

    Args:
        g: the gradient of the surrogate, shape (P,).
        fvp: a function mapping v to F v.
        max_kl: the trust-region size delta.
        cg_iters: iterations of conjugate gradient.

    Returns:
        The step s, shape (P,).
    """
    raise NotImplementedError("natural_step")


def line_search(surrogate_fn, kl_fn, params, full_step, max_kl, backtracks=10, shrink=0.5):
    """Backtrack along full_step until the step is safe and helps.

    Try fractions 1, shrink, shrink^2, ... in turn, up to `backtracks` tries.
    Accept the first candidate params + fraction * full_step whose KL is at
    most max_kl and whose surrogate is strictly higher than at params.

    Args:
        surrogate_fn: maps parameters to the surrogate value.
        kl_fn: maps parameters to the mean KL from the old policy.
        params: the current parameters, shape (P,).
        full_step: the proposed step, shape (P,).
        max_kl: the trust-region size delta.
        backtracks: the number of fractions to try.
        shrink: the factor between successive fractions.

    Returns:
        (new_params, fraction). If no fraction is accepted, return a copy of
        params and 0.0.
    """
    raise NotImplementedError("line_search")


def trpo_update(logits_fn, params, contexts, actions, advantages, max_kl=0.01, damping=0.1):
    """One TRPO update from a batch of sampled contexts, actions, and advantages.

    Record logp_old and the old logits at params, take the gradient of the
    surrogate, compute the natural step with the damped Fisher-vector product
    on the batch's contexts, and run the line search.

    Args:
        logits_fn: maps flat parameters to logits for every context, shape (C, A).
        params: the current parameters, shape (P,).
        contexts: the sampled context of each sample, shape (N,), integers.
        actions: the sampled actions, shape (N,), integers.
        advantages: the advantage of each sample, shape (N,).
        max_kl: the trust-region size delta.
        damping: the damping added to the Fisher matrix.

    Returns:
        The new parameters, shape (P,), detached from any graph.
    """
    raise NotImplementedError("trpo_update")


def sample_batch(logits_fn, params, rewards, batch, generator):
    """Sample contexts uniformly and actions from the policy. Given.

    Returns (contexts, actions, advantages). The advantage is exact:
    the action's reward minus the policy's expected reward in that context.
    """
    with torch.no_grad():
        pi = torch.softmax(logits_fn(params), dim=-1)
        contexts = torch.randint(len(rewards), (batch,), generator=generator)
        actions = torch.multinomial(pi[contexts], 1, generator=generator).squeeze(-1)
        values = (pi * rewards).sum(dim=-1)
        advantages = rewards[contexts, actions] - values[contexts]
    return contexts, actions, advantages


def expected_reward(logits_fn, params, rewards):
    """The exact objective: the mean over contexts of sum_a pi(a | c) r(c, a). Given."""
    with torch.no_grad():
        return float((torch.softmax(logits_fn(params), dim=-1) * rewards).sum(dim=-1).mean())


if __name__ == "__main__":
    features, rewards, logits_fn = make_contextual_bandit(seed=0)
    params = torch.zeros(3 * 4, dtype=DTYPE)
    gen = torch.Generator().manual_seed(0)
    best = float(rewards.max(dim=-1).values.mean())
    print(f"best possible expected reward: {best:.4f}")
    print("iter  expected reward   KL to previous policy")
    for it in range(15):
        contexts, actions, advantages = sample_batch(logits_fn, params, rewards, 256, gen)
        new = trpo_update(logits_fn, params, contexts, actions, advantages, max_kl=0.02)
        kl = float(mean_kl(logits_fn(params), logits_fn(new)))
        params = new
        print(f"{it:4d}  {expected_reward(logits_fn, params, rewards):14.4f}   {kl:.4f}")

"""Meta-RL: RL^2 inputs, Bayes-optimal adaptation on a bandit family, and MAML."""

import numpy as np

# The family of tasks for the bandit half: each row of `thetas` holds the two
# arm means of one task. FAMILY_PRIOR is how likely each task is.
FAMILY = np.array([[0.8, 0.2], [0.4, 0.6], [0.2, 0.8]])
FAMILY_PRIOR = np.array([0.5, 0.2, 0.3])

# The task family for the MAML half: a policy with a Gaussian action
# a ~ N(theta, SIGMA^2 I) is rewarded exp(-|a - c|^2 / (2 w^2)). A task is (c, w).
SIGMA = 0.5
TASKS = [
    (np.array([1.2, 0.0]), 0.5),
    (np.array([-0.6, 1.0]), 0.8),
    (np.array([-0.6, -1.0]), 1.2),
]


# ---------------------------------------------------------------- RL^2 inputs
def rl2_inputs(obs, prev_action, prev_reward, done, n_obs, n_actions):
    """One input vector for the recurrent policy, length n_obs + n_actions + 2.

    Layout: one-hot(obs), one-hot(prev_action), prev_reward, float(done).
    `prev_action=None` marks the first step of a trial: its block is all zeros.
    """
    x = np.zeros(n_obs + n_actions + 2)
    x[obs] = 1.0
    if prev_action is not None:
        x[n_obs + prev_action] = 1.0
    x[n_obs + n_actions] = prev_reward
    x[n_obs + n_actions + 1] = float(done)
    return x


def rl2_trial_inputs(obs, actions, rewards, dones, n_obs, n_actions):
    """Stack the inputs for one whole trial into an array of shape (T, d).

    Row t describes the step at time t: obs[t] together with the action,
    reward, and done flag of step t - 1. Row 0 has no previous step, so its
    previous action is None, its reward is 0, and its done flag is False.
    `dones[t]` says whether step t ended an episode: the hidden state still
    carries on across that boundary, and only a new trial starts from scratch.
    """
    rows = []
    for t in range(len(obs)):
        if t == 0:
            rows.append(rl2_inputs(obs[0], None, 0.0, False, n_obs, n_actions))
        else:
            rows.append(rl2_inputs(obs[t], actions[t - 1], rewards[t - 1], dones[t - 1], n_obs, n_actions))
    return np.stack(rows)


# ------------------------------------------------------------- task inference
def posterior_update(post, thetas, arm, reward):
    """One Bayes step. `post` has shape (K,), `thetas` (K, 2), reward is 0 or 1."""
    like = thetas[:, arm] if reward == 1 else 1.0 - thetas[:, arm]
    new = post * like
    return new / new.sum()


def task_posterior(prior, thetas, history):
    """Posterior over the K tasks after `history`, a list of (arm, reward)."""
    post = np.asarray(prior, dtype=float)
    for arm, reward in history:
        post = posterior_update(post, thetas, arm, reward)
    return post


def _step(post, thetas, arm, H, value):
    """Expected reward of pulling `arm` now, plus the value of what follows."""
    m = float(post @ thetas[:, arm])
    return (
        m
        + m * value(posterior_update(post, thetas, arm, 1), thetas, H - 1)
        + (1.0 - m) * value(posterior_update(post, thetas, arm, 0), thetas, H - 1)
    )


def bayes_optimal_value(prior, thetas, H):
    """Expected total reward over H pulls of the best adaptive policy."""
    if H == 0:
        return 0.0
    post = np.asarray(prior, dtype=float)
    return max(_step(post, thetas, a, H, bayes_optimal_value) for a in range(thetas.shape[1]))


def thompson_value(prior, thetas, H):
    """Exact expected total reward of Thompson sampling over H pulls.

    Thompson sampling draws a task from the posterior and plays that task's
    best arm, so it plays arm a with the posterior mass of the tasks whose
    best arm is a. Ties inside one task split evenly.
    """
    if H == 0:
        return 0.0
    post = np.asarray(prior, dtype=float)
    is_best = thetas == thetas.max(axis=1, keepdims=True)
    share = is_best / is_best.sum(axis=1, keepdims=True)
    p_arm = post @ share
    return sum(
        p_arm[a] * _step(post, thetas, a, H, thompson_value)
        for a in range(thetas.shape[1])
        if p_arm[a] > 0
    )


def fixed_policy_value(prior, thetas, H):
    """Best policy that ignores what it sees: pull the best prior-mean arm H times."""
    return H * float((np.asarray(prior) @ thetas).max())


def oracle_value(prior, thetas, H):
    """Value of knowing the task before the first pull."""
    return H * float(np.asarray(prior) @ thetas.max(axis=1))


# ----------------------------------------------------------------------- MAML
def task_objective(theta, task):
    """Expected reward of the policy N(theta, SIGMA^2 I) on a task (c, w)."""
    c, w = task
    s2 = w * w + SIGMA**2
    d = len(theta)
    return (w * w / s2) ** (d / 2) * np.exp(-np.sum((theta - c) ** 2) / (2 * s2))


def task_grad(theta, task):
    """Gradient of `task_objective` with respect to theta, shape (d,)."""
    c, w = task
    s2 = w * w + SIGMA**2
    return -task_objective(theta, task) * (theta - c) / s2


def task_hessian(theta, task):
    """Hessian of `task_objective` with respect to theta, shape (d, d)."""
    c, w = task
    s2 = w * w + SIGMA**2
    diff = theta - c
    return task_objective(theta, task) * (np.outer(diff, diff) / s2**2 - np.eye(len(theta)) / s2)


def adapt(theta, task, alpha):
    """The inner step: one gradient-ascent step of size alpha on the task."""
    return theta + alpha * task_grad(theta, task)


def maml_objective(theta, tasks, alpha):
    """Mean over tasks of the objective after one adaptation step."""
    return float(np.mean([task_objective(adapt(theta, t, alpha), t) for t in tasks]))


def maml_grad(theta, tasks, alpha):
    """Exact gradient of `maml_objective`, including the second-order term."""
    eye = np.eye(len(theta))
    grads = []
    for t in tasks:
        adapted = adapt(theta, t, alpha)
        jac = eye + alpha * task_hessian(theta, t)
        grads.append(jac @ task_grad(adapted, t))
    return np.mean(grads, axis=0)


def fomaml_grad(theta, tasks, alpha):
    """First-order approximation: treat the adapted parameters as if they did not depend on theta."""
    return np.mean([task_grad(adapt(theta, t, alpha), t) for t in tasks], axis=0)


def maml_step(theta, tasks, alpha, beta):
    """One outer step: gradient ascent of size beta on the post-adaptation objective."""
    return theta + beta * maml_grad(theta, tasks, alpha)


def maml_train(theta0, tasks, alpha, beta, steps):
    """Repeat `maml_step` and return the final theta."""
    theta = np.array(theta0, dtype=float)
    for _ in range(steps):
        theta = maml_step(theta, tasks, alpha, beta)
    return theta


def multitask_init(tasks, beta, steps):
    """Maximize the mean objective with no adaptation, starting from zero."""
    theta = np.zeros(len(tasks[0][0]))
    for _ in range(steps):
        theta = theta + beta * np.mean([task_grad(theta, t) for t in tasks], axis=0)
    return theta


if __name__ == "__main__":
    print("task posterior after (0,1), (0,1), (0,0):", np.round(task_posterior(FAMILY_PRIOR, FAMILY, [(0, 1), (0, 1), (0, 0)]), 4))
    for H in (1, 2, 3, 4):
        print(
            f"H={H}: fixed {fixed_policy_value(FAMILY_PRIOR, FAMILY, H):.4f}"
            f"  thompson {thompson_value(FAMILY_PRIOR, FAMILY, H):.4f}"
            f"  bayes {bayes_optimal_value(FAMILY_PRIOR, FAMILY, H):.4f}"
            f"  oracle {oracle_value(FAMILY_PRIOR, FAMILY, H):.4f}"
        )
    alpha = 4.0
    mt = multitask_init(TASKS, 0.5, 600)
    meta = maml_train(mt, TASKS, alpha, 1.0, 150)
    print(f"multi-task init, after one adaptation step: {maml_objective(mt, TASKS, alpha):.4f}")
    print(f"MAML init, after one adaptation step:       {maml_objective(meta, TASKS, alpha):.4f}")

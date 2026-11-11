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


# ---------------------------------------------------------------- the exercise
def rl2_inputs(obs, prev_action, prev_reward, done, n_obs, n_actions):
    """One input vector for the recurrent policy, length n_obs + n_actions + 2.

    Layout: one-hot(obs), one-hot(prev_action), prev_reward, float(done).
    `prev_action=None` marks the first step of a trial: its block is all zeros.
    """
    raise NotImplementedError


def rl2_trial_inputs(obs, actions, rewards, dones, n_obs, n_actions):
    """Stack the inputs for one whole trial into an array of shape (T, d).

    Row t describes the step at time t: obs[t] together with the action,
    reward, and done flag of step t - 1. Row 0 has no previous step, so its
    previous action is None, its reward is 0, and its done flag is False.
    `dones[t]` says whether step t ended an episode: the hidden state still
    carries on across that boundary, and only a new trial starts from scratch.
    """
    raise NotImplementedError


def posterior_update(post, thetas, arm, reward):
    """One Bayes step. `post` has shape (K,), `thetas` (K, 2), reward is 0 or 1."""
    raise NotImplementedError


def task_posterior(prior, thetas, history):
    """Posterior over the K tasks after `history`, a list of (arm, reward)."""
    raise NotImplementedError


def bayes_optimal_value(prior, thetas, H):
    """Expected total reward over H pulls of the best adaptive policy."""
    raise NotImplementedError


def thompson_value(prior, thetas, H):
    """Exact expected total reward of Thompson sampling over H pulls.

    Thompson sampling draws a task from the posterior and plays that task's
    best arm, so it plays arm a with the posterior mass of the tasks whose
    best arm is a. Ties inside one task split evenly.
    """
    raise NotImplementedError


def fixed_policy_value(prior, thetas, H):
    """Best policy that ignores what it sees: pull the best prior-mean arm H times."""
    raise NotImplementedError


def oracle_value(prior, thetas, H):
    """Value of knowing the task before the first pull."""
    raise NotImplementedError


def task_objective(theta, task):
    """Expected reward of the policy N(theta, SIGMA^2 I) on a task (c, w)."""
    raise NotImplementedError


def task_grad(theta, task):
    """Gradient of `task_objective` with respect to theta, shape (d,)."""
    raise NotImplementedError


def task_hessian(theta, task):
    """Hessian of `task_objective` with respect to theta, shape (d, d)."""
    raise NotImplementedError


def adapt(theta, task, alpha):
    """The inner step: one gradient-ascent step of size alpha on the task."""
    raise NotImplementedError


def maml_objective(theta, tasks, alpha):
    """Mean over tasks of the objective after one adaptation step."""
    raise NotImplementedError


def maml_grad(theta, tasks, alpha):
    """Exact gradient of `maml_objective`, including the second-order term."""
    raise NotImplementedError


def fomaml_grad(theta, tasks, alpha):
    """First-order approximation: treat the adapted parameters as if they did not depend on theta."""
    raise NotImplementedError


def maml_step(theta, tasks, alpha, beta):
    """One outer step: gradient ascent of size beta on the post-adaptation objective."""
    raise NotImplementedError


def maml_train(theta0, tasks, alpha, beta, steps):
    """Repeat `maml_step` and return the final theta."""
    raise NotImplementedError


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

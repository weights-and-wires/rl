"""Distributional reinforcement learning: the C51 projection and quantile regression.

Fill in the functions below, in order. The first five build C51, which keeps
a probability for each of a fixed set of returns. The last three build
QR-DQN's loss, which instead learns the returns at fixed probability levels.
"""

import numpy as np
import torch

import rlenvs

torch.set_num_threads(1)


def categorical_support(v_min: float, v_max: float, n_atoms: int) -> np.ndarray:
    """The fixed atoms z_i = v_min + i * dz, with dz = (v_max - v_min) / (n_atoms - 1).

    Returns:
        Shape (n_atoms,), from v_min to v_max inclusive.
    """
    raise NotImplementedError("categorical_support")


def c51_project(next_probs, rewards, terminated, gamma: float, support) -> np.ndarray:
    """Project the shifted distribution r + gamma * Z(s2) back onto the support.

    For each atom z_j: move it to r + gamma * (1 - terminated) * z_j, clip that
    into [v_min, v_max], and split its probability between the two
    neighbouring atoms in proportion to closeness. When it lands exactly on an
    atom, that atom gets all of it.

    Args:
        next_probs: p_j(s2, a*) for each transition, shape (B, N).
        rewards: r_t, shape (B,).
        terminated: 1.0 where s2 is terminal, else 0.0, shape (B,).
        gamma: discount.
        support: the atoms, shape (N,), evenly spaced.

    Returns:
        m, shape (B, N), each row summing to 1.
    """
    raise NotImplementedError("c51_project")


def c51_greedy(next_probs, support):
    """Pick the action whose return distribution has the largest mean.

    Args:
        next_probs: p_i(s2, a) for every action, shape (B, A, N).
        support: the atoms, shape (N,).

    Returns:
        A tuple (actions, probs): actions has shape (B,), and probs, shape
        (B, N), holds each row's distribution for its chosen action.
    """
    raise NotImplementedError("c51_greedy")


def c51_loss(logits: torch.Tensor, target_probs) -> torch.Tensor:
    """Mean cross-entropy between projected targets and predicted distributions.

    Args:
        logits: the network's scores for the taken action, shape (B, N).
            The predicted distribution is softmax(logits) along axis 1.
        target_probs: the projected targets m, shape (B, N), treated as
            constants.

    Returns:
        The scalar mean over the batch of -sum_i m_i log p_i.
    """
    raise NotImplementedError("c51_loss")


def categorical_evaluation(P, R, pi, gamma: float, support, iters: int = 200) -> np.ndarray:
    """Evaluate a policy's return distributions with the projected distributional Bellman operator.

    Start every state's distribution with all its mass on the atom nearest 0.
    Then repeat `iters` times: the updated distribution of state s is
    sum_a pi[s, a] sum_s2 P[s, a, s2] * c51_project(dist[s2], R[s, a], ...),
    with no terminal states.

    Args:
        P: transition probabilities, shape (S, A, S).
        R: rewards R[s, a], shape (S, A).
        pi: the policy, shape (S, A).
        gamma: discount, below 1.
        support: the atoms, shape (N,).
        iters: number of sweeps.

    Returns:
        dist, shape (S, N), each row summing to 1.
    """
    raise NotImplementedError("categorical_evaluation")


def quantile_midpoints(n: int) -> np.ndarray:
    """The probability levels tau_hat_i = (2i - 1) / (2n) for i = 1, ..., n.

    Returns:
        Shape (n,).
    """
    raise NotImplementedError("quantile_midpoints")


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, taus, kappa: float = 1.0) -> torch.Tensor:
    """QR-DQN's quantile Huber loss.

    For every batch row b, prediction i, and target j, let
    u = target[b, j] - pred[b, i]. The cost is |taus[i] - (u < 0)| * L(u),
    where L is the Huber loss with threshold kappa (0.5 u**2 inside kappa,
    kappa * (|u| - kappa / 2) outside). When kappa == 0, L(u) = |u|, the plain
    quantile loss.

    Args:
        pred: predicted quantile locations theta_i, shape (B, N).
        target: target samples or quantiles, shape (B, M), treated as constants.
        taus: the probability level of each prediction, shape (N,).
        kappa: the Huber threshold, 0 or positive.

    Returns:
        A scalar: average over j, sum over i, then average over the batch.
    """
    raise NotImplementedError("quantile_huber_loss")


if __name__ == "__main__":
    # 1. The C51 projection on the worked example.
    support = categorical_support(-2.0, 2.0, 5)
    p = np.array([[0.1, 0.2, 0.4, 0.2, 0.1]])
    m = c51_project(p, np.array([1.0]), np.array([0.0]), 0.9, support)
    print(f"projected m = {np.round(m[0], 4)}, mass {m.sum():.4f}, mean {m[0] @ support:.4f}")

    # 2. Distributional policy evaluation: the means match V^pi.
    mdp = rlenvs.random_mdp(n_states=4, n_actions=2, gamma=0.9, seed=0)
    pi = rlenvs.uniform_policy(mdp)
    support = categorical_support(-40.0, 40.0, 81)
    dist = categorical_evaluation(mdp.P, mdp.R, pi, mdp.gamma, support)
    print("means of the return distributions:", np.round(dist @ support, 4))
    print("V^pi from the Bellman equation:   ", np.round(rlenvs.policy_evaluation(mdp, pi), 4))

    # 3. Quantile regression on samples from an exponential distribution.
    torch.manual_seed(0)
    samples = torch.distributions.Exponential(1.0).sample((1, 4000)).double()
    taus = quantile_midpoints(4)
    theta = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.Adam([theta], lr=0.05)
    for _ in range(600):
        loss = quantile_huber_loss(theta, samples, taus, kappa=0.0)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print(f"learned quantiles {np.round(theta.detach().numpy()[0], 3)}, "
          f"true {np.round(-np.log(1 - taus), 3)}")

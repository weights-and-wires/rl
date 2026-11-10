"""Partial observability: beliefs, memoryless limits, and recurrent policies.

Fill in the functions below, in order. You start with the tiger problem and
update a belief with Bayes' rule. You then compute exactly how much a policy
without memory can earn on a T-maze, and you build the pieces that train a
recurrent policy on it: a hidden state that resets at episode boundaries, a
padded and masked batch loss, and a policy-gradient training loop.

The T-maze is a corridor of 4 steps. At the start the agent sees a signal that
says which of two doors at the far end pays, left or right with equal
probability. Every corridor cell looks the same, so the agent has to remember
the signal. Observations are 0 (start, signal left), 1 (start,
signal right), 2 (corridor), 3 (junction), and 4 (episode over, never acted
on). Actions are 0 (advance), 1 (open the left door), and 2 (open the right
door). Only advancing makes sense in the corridor, and only a door makes sense
at the junction. Anything else walks into a wall and ends the episode with
reward -1. A correct door pays +1 and a wrong door pays -1.
"""

import itertools

import numpy as np
import torch

import rlenvs

torch.set_num_threads(1)

LENGTH = 4  # corridor steps between the start and the junction
GAMMA = 0.95
P_LEFT = 0.5  # probability that the signal says "left"
HORIZON = LENGTH + 1  # every episode ends within this many steps
N_OBS = 5
N_ACTIONS = 3


# --- The environment (given, identical in the solution) ---------------------


def tmaze_mdp(length=LENGTH, gamma=GAMMA, p_left=P_LEFT):
    """The T-maze as a TabularMDP over full states, plus what the agent sees.

    A state is (signal, position), numbered signal * (length + 1) + position,
    with one extra terminal state last. Signal 0 means the left door pays.
    Returns (mdp, obs_of_state), where obs_of_state[s] is the observation id
    the agent sees in state s.
    """
    n_pos = length + 1
    n = 2 * n_pos + 1
    term = n - 1
    P = np.zeros((n, N_ACTIONS, n))
    R = np.zeros((n, N_ACTIONS))
    obs_of_state = np.full(n, 4)
    for sig in (0, 1):
        for pos in range(n_pos):
            s = sig * n_pos + pos
            obs_of_state[s] = sig if pos == 0 else (2 if pos < length else 3)
            for a in range(N_ACTIONS):
                if pos < length and a == 0:
                    P[s, a, s + 1] = 1.0
                    continue
                P[s, a, term] = 1.0
                if pos == length and a > 0:
                    R[s, a] = 1.0 if (a - 1) == sig else -1.0
                else:
                    R[s, a] = -1.0  # a wall
    P[term, :, term] = 1.0
    mu0 = np.zeros(n)
    mu0[0], mu0[n_pos] = p_left, 1.0 - p_left
    terminal = np.zeros(n, dtype=bool)
    terminal[term] = True
    return rlenvs.TabularMDP(P=P, R=R, gamma=gamma, mu0=mu0, terminal=terminal), obs_of_state


class TMaze:
    """A batch of T-mazes that step together. signals[i] is 0 or 1 for maze i."""

    def __init__(self, signals, length=LENGTH):
        self.sig = np.asarray(signals)
        self.length = length
        self.pos = np.zeros(len(self.sig), dtype=int)
        self.alive = np.ones(len(self.sig), dtype=bool)

    def obs(self):
        """Observation ids, shape (n,)."""
        mid = np.where(self.pos < self.length, 2, 3)
        return np.where(self.pos == 0, self.sig, mid)

    def step(self, actions):
        """Apply one action per maze. Returns (rewards, done), each shape (n,).

        Mazes that already finished pay 0 and stay finished.
        """
        a = np.asarray(actions)
        advance = (self.pos < self.length) & (a == 0)
        door = (self.pos == self.length) & (a > 0)
        reward = np.where(door, np.where((a - 1) == self.sig, 1.0, -1.0), -1.0)
        reward = np.where(advance, 0.0, reward) * self.alive
        done = ~advance & self.alive
        self.pos = np.where(advance, self.pos + 1, self.pos)
        self.alive = self.alive & advance
        return reward, done


class PolicyNet(torch.nn.Module):
    """A GRU cell that reads a one-hot observation, and a linear head for action logits."""

    def __init__(self, hidden=16):
        super().__init__()
        self.hidden = hidden
        self.cell = torch.nn.GRUCell(N_OBS, hidden)
        self.head = torch.nn.Linear(hidden, N_ACTIONS)


def collect(net, n_episodes, rng, memory=True):
    """Play n_episodes mazes at once for HORIZON steps, sampling from the policy.

    With memory=False the hidden state is zero at every step, so the policy
    sees only the current observation. Returns a dict of tensors, time first:
      obs (T, N, N_OBS) one-hot, actions (T, N), rewards (T, N),
      mask (T, N), 1 while the episode is still running, and
      first (T, N), 1 on the first step of an episode (only row 0 here).
    """
    env = TMaze((rng.random(n_episodes) >= P_LEFT).astype(int))
    h = torch.zeros(n_episodes, net.hidden)
    obs, acts, rews, masks = [], [], [], []
    for _ in range(HORIZON):
        x = torch.nn.functional.one_hot(torch.as_tensor(env.obs()), N_OBS).float()
        with torch.no_grad():
            h = net.cell(x, h if memory else torch.zeros_like(h))
            a = torch.distributions.Categorical(logits=net.head(h)).sample()
        masks.append(env.alive.astype(np.float32))
        r, _ = env.step(a.numpy())
        obs.append(x)
        acts.append(a)
        rews.append(torch.as_tensor(r, dtype=torch.float32))
    first = torch.zeros(HORIZON, n_episodes)
    first[0] = 1.0
    return {
        "obs": torch.stack(obs),
        "actions": torch.stack(acts),
        "rewards": torch.stack(rews),
        "mask": torch.as_tensor(np.stack(masks)),
        "first": first,
    }


def discounted_returns(rewards, gamma):
    """Return-to-go for rewards of shape (T, N), computed backward in time."""
    out = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[1])
    for t in reversed(range(rewards.shape[0])):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def greedy_value(net, memory=True):
    """The exact discounted return of the policy that takes the most likely action.

    The maze is deterministic once the signal is fixed, so weighting the two
    signals by their probabilities gives the exact value, with no sampling
    error.
    """
    env = TMaze([0, 1])
    h = torch.zeros(2, net.hidden)
    total, discount = np.zeros(2), 1.0
    for _ in range(HORIZON):
        x = torch.nn.functional.one_hot(torch.as_tensor(env.obs()), N_OBS).float()
        with torch.no_grad():
            h = net.cell(x, h if memory else torch.zeros_like(h))
            a = net.head(h).argmax(dim=-1).numpy()
        r, _ = env.step(a)
        total += discount * r
        discount *= GAMMA
    return float(total @ np.array([P_LEFT, 1.0 - P_LEFT]))


# --- 1. Beliefs --------------------------------------------------------------


def belief_update(b, a, o, T, O):
    """The belief after taking action a and seeing observation o.

    Args:
        b: the current belief over states, shape (S,), sums to 1.
        a: the action taken (an index).
        o: the observation received (an index).
        T: transition probabilities, shape (A, S, S). T[a, s, s2] is
            P(s2 | s, a).
        O: observation probabilities, shape (A, S, n_obs). O[a, s2, o] is
            P(o | s2, a), the chance of seeing o after landing in s2.

    Returns:
        The new belief, shape (S,), sums to 1. Do not change b in place.
    """
    raise NotImplementedError("belief_update")


# --- 2. The best policy without memory ---------------------------------------


def memoryless_value(mdp, obs_of_state, pi_obs):
    """The exact value, from the start distribution, of a policy that sees only o.

    Args:
        mdp: a TabularMDP over full states.
        obs_of_state: observation id of each state, shape (S,), integers.
        pi_obs: pi_obs[o, a] is the probability of action a after seeing
            observation o, shape (n_obs, A), rows sum to 1.

    Returns:
        The float mu0 . V, where V is the exact value of the state policy
        pi_obs[obs_of_state].
    """
    raise NotImplementedError("memoryless_value")


def best_memoryless_value(mdp, obs_of_state):
    """The best value over deterministic policies that see only the observation.

    Try every assignment of one action to each observation that occurs in a
    non-terminal state.

    Returns:
        (value, pi_obs): the best value, and its policy as a one-hot array of
        shape (n_obs, A).
    """
    raise NotImplementedError("best_memoryless_value")


# --- 3. A hidden state that resets at episode boundaries -----------------------


def reset_hidden(h, first):
    """Zero the hidden state of every row that starts a new episode.

    Args:
        h: hidden states, shape (B, H).
        first: shape (B,), 1 where this step begins an episode, 0 elsewhere.

    Returns:
        A new tensor of shape (B, H). Rows with first = 1 are zero. The rest
        are unchanged.
    """
    raise NotImplementedError("reset_hidden")


def unroll(cell, obs, first, h0=None):
    """Run a recurrent cell over a batch of sequences.

    Args:
        cell: a torch.nn.GRUCell, called as cell(x, h) -> h.
        obs: inputs, shape (T, B, D).
        first: shape (T, B), 1 where step t starts an episode in column b.
        h0: the hidden state before step 0, shape (B, H). Defaults to zeros.

    Returns:
        The hidden state after every step, shape (T, B, H). The hidden state
        is reset before the step whenever first is 1, so nothing flows from one
        episode to the next in the same column.
    """
    raise NotImplementedError("unroll")


# --- 4. The loss over padded episodes ------------------------------------------


def masked_pg_loss(logits, actions, advantages, mask):
    """The policy-gradient loss over a padded batch.

    Args:
        logits: shape (T, B, A).
        actions: integer actions, shape (T, B).
        advantages: shape (T, B).
        mask: shape (T, B), 1 on real steps and 0 on padding.

    Returns:
        A scalar: minus the sum of mask * advantages * log pi(a | logits),
        divided by the number of real steps, mask.sum().
    """
    raise NotImplementedError("masked_pg_loss")


def sequence_loss(cell, head, obs, actions, advantages, mask, first):
    """Unroll the cell, turn hidden states into logits with head, and apply masked_pg_loss."""
    raise NotImplementedError("sequence_loss")


# --- 5. Training --------------------------------------------------------------


def train_policy(seed, memory=True, iters=150, batch=64, lr=0.02, hidden=16):
    """Train a PolicyNet by policy gradient on the T-maze and return it.

    Each iteration: collect a batch, compute discounted returns, subtract a
    baseline (the mean return at each time step over the steps still running),
    and take one Adam step on sequence_loss. With memory=False, pass first of
    all ones to the loss so the hidden state resets at every step, which makes
    the policy memoryless in training too.
    """
    raise NotImplementedError("train_policy")


if __name__ == "__main__":
    # The tiger: two listens that both hear the tiger on the left.
    T = np.stack([np.eye(2), np.full((2, 2), 0.5), np.full((2, 2), 0.5)])
    O = np.stack([[[0.85, 0.15], [0.15, 0.85]], np.full((2, 2), 0.5), np.full((2, 2), 0.5)])
    b = np.array([0.5, 0.5])
    for step in (1, 2):
        b = belief_update(b, 0, 0, T, O)
        print(f"belief after {step} listen(s) that hear left: {np.round(b, 4)}")

    mdp, obs_of_state = tmaze_mdp()
    full = float(mdp.mu0 @ rlenvs.value_iteration(mdp))
    best, _ = best_memoryless_value(mdp, obs_of_state)
    print(f"T-maze: best policy with full state {full:.4f}, best memoryless {best:.4f}")
    for memory in (True, False):
        net = train_policy(seed=0, memory=memory)
        print(f"  trained {'with' if memory else 'without'} memory: "
              f"greedy value {greedy_value(net, memory):.4f}")

"""Reference solution for the generalised advantage estimation exercise."""

import numpy as np

import rlenvs


def td_errors(rewards, values, next_values, terminated, gamma):
    rewards = np.asarray(rewards, dtype=float)
    alive = 1.0 - np.asarray(terminated, dtype=float)
    return rewards + gamma * alive * np.asarray(next_values, dtype=float) - np.asarray(values, dtype=float)


def next_state_values(values, last_value, done, final_values):
    values = np.asarray(values, dtype=float)
    done = np.asarray(done, dtype=bool)
    nxt = np.empty_like(values)
    nxt[:-1] = values[1:]
    nxt[-1] = last_value
    # After an episode ends the environment resets, so values[t + 1] belongs
    # to the next episode. Use the value of the observation the step reached.
    return np.where(done, np.asarray(final_values, dtype=float), nxt)


def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    delta = td_errors(rewards, values, next_values, terminated, gamma)
    ended = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
    adv = np.zeros_like(delta)
    running = np.zeros(delta.shape[1:])
    for t in range(delta.shape[0] - 1, -1, -1):
        running = delta[t] + gamma * lam * (1.0 - ended[t]) * running
        adv[t] = running
    return adv


def lambda_returns(rewards, values, next_values, terminated, truncated, gamma, lam):
    return gae(rewards, values, next_values, terminated, truncated, gamma, lam) + np.asarray(values, dtype=float)


if __name__ == "__main__":
    rewards = np.array([1.0, 0.0, 2.0])
    values = np.array([0.5, 1.0, 1.5])
    gamma, lam = 0.9, 0.8

    # The episode ends at the third step. Either the world ended it, or a time
    # limit cut it off at a state whose value is 1.0.
    next_values = np.array([1.0, 1.5, 1.0])
    never = np.array([False, False, False])
    at_end = np.array([False, False, True])
    print("delta (terminated):   ", td_errors(rewards, values, next_values, at_end, gamma))
    print("GAE   (terminated):   ", gae(rewards, values, next_values, at_end, never, gamma, lam))
    print("GAE   (truncated):    ", gae(rewards, values, next_values, never, at_end, gamma, lam))
    print("lambda=1 (terminated):", gae(rewards, values, next_values, at_end, never, gamma, 1.0))
    print("lambda-returns:       ", lambda_returns(rewards, values, next_values, at_end, never, gamma, lam))

    # With the true V^pi as the critic, GAE averages to A^pi(s, a) for any lambda.
    mdp = rlenvs.random_mdp(n_states=4, n_actions=2, gamma=0.9, seed=0)
    pi = rlenvs.uniform_policy(mdp)
    v = rlenvs.policy_evaluation(mdp, pi)
    adv = rlenvs.q_from_v(mdp, v) - v[:, None]
    rng = np.random.default_rng(0)
    H, N, s0, a0 = 6, 4000, 0, 1
    firsts = {0.0: [], 0.5: [], 1.0: []}
    for _ in range(N):
        s, a = s0, a0
        rs, vs, nvs = np.zeros(H), np.zeros(H), np.zeros(H)
        for t in range(H):
            s2, r = mdp.sample(s, a, rng)
            rs[t], vs[t], nvs[t] = r, v[s], v[s2]
            s, a = s2, int(rng.choice(2, p=pi[s2]))
        term = np.zeros(H, dtype=bool)
        trunc = np.zeros(H, dtype=bool)
        trunc[-1] = True  # a time limit cuts the rollout after H steps
        for lam_ in firsts:
            firsts[lam_].append(gae(rs, vs, nvs, term, trunc, 0.9, lam_)[0])
    print(f"\nA^pi({s0}, {a0}) = {adv[s0, a0]:+.4f}")
    for lam_, xs in firsts.items():
        xs = np.array(xs)
        print(f"  lambda = {lam_:.1f}: mean GAE {xs.mean():+.4f}, std err {xs.std(ddof=1) / np.sqrt(N):.4f}")

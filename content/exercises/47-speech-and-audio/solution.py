"""Reference solution for the speech and audio exercise."""

import numpy as np
import torch

torch.set_num_threads(1)

# The toy text-to-speech world. The same code sits in starter.py. --------

# The policy picks one speaking rate per utterance, 1.0 being normal speed.
SPEAKING_RATES = np.linspace(0.6, 1.6, 11)


def asr_wer(rate):
    """A stand-in for an ASR model's word error rate on speech at `rate`.

    Faster speech is harder to transcribe, so the error rises with the rate.
    """
    rate = np.asarray(rate, dtype=float)
    return 0.02 + 0.3 / (1.0 + np.exp(-4.0 * (rate - 1.2)))


def utmos(rate):
    """A stand-in for a UTMOS-style naturalness score, on a 1-to-5 scale.

    Listeners prefer normal speed. Speech much slower or faster sounds odd.
    """
    rate = np.asarray(rate, dtype=float)
    return 1.0 + 3.5 * np.exp(-((rate - 1.0) ** 2) / (2 * 0.2**2))


def reference_logprobs():
    """log pi_ref over SPEAKING_RATES: the pretrained model talks a little fast."""
    logits = -((SPEAKING_RATES - 1.1) ** 2) / (2 * 0.25**2)
    return logits - np.log(np.sum(np.exp(logits)))


# The exercise. ----------------------------------------------------------


def edit_distance(ref, hyp):
    ref, hyp = list(ref), list(hyp)
    # prev[j] is the distance between the first i-1 reference words and hyp[:j].
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            substitute = prev[j - 1] + (ref[i - 1] != hyp[j - 1])
            delete = prev[j] + 1
            insert = cur[j - 1] + 1
            cur[j] = min(substitute, delete, insert)
        prev = cur
    return prev[-1]


def wer(ref, hyp):
    ref_words, hyp_words = ref.split(), hyp.split()
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def _renormalise(logscores):
    z = np.asarray(logscores, dtype=float)
    p = np.exp(z - z.max())
    return p / p.sum()


def mwer_loss(logscores, errors):
    p_hat = _renormalise(logscores)
    errors = np.asarray(errors, dtype=float)
    # The paper's W-bar: the average error count over the list, a constant.
    w_bar = errors.mean()
    return float(np.sum(p_hat * (errors - w_bar)))


def mwer_grad(logscores, errors):
    p_hat = _renormalise(logscores)
    errors = np.asarray(errors, dtype=float)
    expected = np.sum(p_hat * errors)
    return p_hat * (errors - expected)


def reinforce_mwer_samples(logscores, errors, n_samples, baseline, seed):
    p_hat = _renormalise(logscores)
    errors = np.asarray(errors, dtype=float)
    rng = np.random.default_rng(seed)
    k = rng.choice(len(p_hat), size=n_samples, p=p_hat)
    # d log p_hat[k] / d logscores = onehot(k) - p_hat.
    score = np.eye(len(p_hat))[k] - p_hat
    return (errors[k] - baseline)[:, None] * score


def tts_reward(rate, w_intel=1.0, w_nat=1.0):
    return w_intel * (1.0 - asr_wer(rate)) + w_nat * utmos(rate) / 5.0


def kl_optimal_policy(ref_logprobs, rewards, beta):
    z = np.asarray(ref_logprobs, dtype=float) + np.asarray(rewards, dtype=float) / beta
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def dpo_population_loss(logits, ref_logprobs, rewards, beta):
    ref_logprobs = torch.as_tensor(ref_logprobs, dtype=logits.dtype)
    rewards = torch.as_tensor(rewards, dtype=logits.dtype)
    logp = torch.log_softmax(logits, dim=-1)
    u = beta * (logp - ref_logprobs)  # implicit rewards
    margin = u[:, None] - u[None, :]  # u_i - u_j
    label = torch.sigmoid(rewards[:, None] - rewards[None, :])  # P(i preferred to j)
    ref_p = torch.exp(ref_logprobs)
    weight = ref_p[:, None] * ref_p[None, :]
    weight = weight * (1.0 - torch.eye(len(u), dtype=logits.dtype))
    per_pair = -label * torch.nn.functional.logsigmoid(margin)
    return (weight * per_pair).sum() / weight.sum()


if __name__ == "__main__":
    print("WER of 'the cat sat' against 'the cat sit':", round(wer("the cat sat", "the cat sit"), 4))

    ref = "turn the kitchen lights off"
    nbest = ["turn the kitchen lights off", "turn the kitchen light off",
             "turn a kitchen light of", "burn the chicken lights off"]
    errors = np.array([edit_distance(ref.split(), h.split()) for h in nbest], dtype=float)
    logscores = np.array([0.5, 1.5, 0.2, 1.0])
    print("word errors per hypothesis:", errors)
    for step in range(0, 201):
        if step % 50 == 0:
            p = np.exp(logscores - logscores.max())
            p /= p.sum()
            print(f"step {step:3d}  expected errors {np.sum(p * errors):.3f}  "
                  f"P(correct) {p[0]:.3f}")
        logscores = logscores - 1.0 * mwer_grad(logscores, errors)

    ref_lp = reference_logprobs()
    print("\nrate policy     mean rate  expected WER  expected UTMOS")
    for name, w_nat in [("WER only", 0.0), ("WER + UTMOS", 1.0)]:
        pi = kl_optimal_policy(ref_lp, tts_reward(SPEAKING_RATES, 1.0, w_nat), beta=0.01)
        print(f"{name:14s}  {np.sum(pi * SPEAKING_RATES):9.3f}  "
              f"{np.sum(pi * asr_wer(SPEAKING_RATES)):12.4f}  {np.sum(pi * utmos(SPEAKING_RATES)):14.3f}")
    pi = np.exp(ref_lp)
    print(f"{'reference':14s}  {np.sum(pi * SPEAKING_RATES):9.3f}  "
          f"{np.sum(pi * asr_wer(SPEAKING_RATES)):12.4f}  {np.sum(pi * utmos(SPEAKING_RATES)):14.3f}")

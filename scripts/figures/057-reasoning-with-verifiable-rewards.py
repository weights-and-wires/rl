#!/usr/bin/env python3
"""Numbers quoted in the zero-signal paragraph of the RLVR lesson.

A prompt with pass rate p, sampled in a group of G, gives GRPO no gradient
signal when all G rewards are equal: with probability p**G + (1 - p)**G.
This lesson has no figure for it. The script prints the quoted values.

Run from the repo root:

    python3 scripts/figures/057-reasoning-with-verifiable-rewards.py
"""

import numpy as np


def zero_signal_probability(p, group_size):
    p = np.asarray(p, dtype=float)
    return p**group_size + (1.0 - p) ** group_size


if __name__ == "__main__":
    print("p=0.5,  G=8:", zero_signal_probability(0.5, 8), "(= 2/256)")
    print("p=0.9,  G=8:", zero_signal_probability(0.9, 8))
    print("p=0.2015, G=8 (toy base model):", zero_signal_probability(0.2015, 8))
    print("p=0.005, G=8 (toy minority question after training):",
          zero_signal_probability(0.005, 8))
    print("p=0.005, G=64:", zero_signal_probability(0.005, 64))
    rng = np.random.default_rng(0)
    hits = rng.random((200000, 8)) < 0.9
    print("Monte Carlo p=0.9, G=8:", np.mean(hits.all(1) | ~hits.any(1)))

# Content contract

Every lesson in this course is two things on disk: one Markdown file that the
reader reads, and one exercise directory that the reader's code is checked
against. This document is the exact contract both must satisfy. The loader in
`backend/app/curriculum.py` and the harness in `backend/app/runner/harness.py`
depend on it.

## Audience

Readers are undergraduates who have finished a first machine learning course,
such as the Weights and Wires ML course at ml.ifkash.dev. They can fit a
logistic regression, train a small network with gradient descent, and write
numpy and a little PyTorch. They are new to reinforcement learning, and dense
notation stops them cold. They read a lesson to understand an idea well enough
to build it, and they learn by building. Write for them:

- Start with the intuition in plain words. Before any formula, say what the
  method does, in one or two sentences a first-year student could repeat to a
  friend. Use an everyday analogy when one fits.
- Make every formula earn its place. Include a formula only when the exercise
  implements it or the reader needs it to follow the next step. Everything else
  is stated in words.
- Read each formula aloud. After every display equation, say in plain English
  what each piece is and what the whole thing computes. Then show the same
  formula as one or two lines of numpy or PyTorch.
- Work one small example by hand. Use tiny numbers, a two-state MDP, three
  arms, or a four-token sequence, and show every intermediate value, so the
  reader can check their code against it.
- Define every symbol and every term the first time it appears, including ones
  that feel basic, such as "policy", "return", "on-policy", or "advantage".
  Prefer an ordinary word to jargon when both mean the same thing.
- Keep one notation across the whole course: states $s$, actions $a$, rewards
  $r$, discount $\gamma$, policy $\pi(a \mid s)$, state value $V^\pi(s)$,
  action value $Q^\pi(s, a)$, advantage $A^\pi(s, a)$, return $G_t$, time step
  $t$, parameters $\theta$ for the policy and $\phi$ for a critic or reward
  model. The reward for taking $a_t$ in $s_t$ is $r_t$ (not $r_{t+1}$). For
  language models, a prompt is $x$, a response is $y$, and the reference policy
  is $\pi_{\text{ref}}$.
- Avoid heavy notation. No measure-theoretic language and no full proofs. A
  proof idea in three or four plain sentences is welcome when it explains why
  something works.
- Keep the conditions honest. When a result needs an assumption, such as a
  policy that visits every state or a step size that shrinks, say so in plain
  words and say what goes wrong without it.
- Connect ideas to code. The reader should finish every section knowing which
  numpy or PyTorch operations they will write.
- Be current and exact about the literature. Later parts cover methods from
  2023 to 2026. Name each method's paper with its year, state its objective as
  the paper does, and say plainly where a claim is contested.

Every result on the page must still be correct. Simpler wording must never
change what a result says.

## Prose style

Follow the Google developer documentation style guide:

- Address the reader as "you". Use active voice and present tense.
- One idea per sentence. Split any sentence that needs a second comma-spliced
  clause to land. Long subordinate chains are the main thing that made these
  lessons hard to read.
- Sentence case for every heading. Start task headings with a bare infinitive.
- Serial commas. Em dashes with no surrounding spaces.
- Code, filenames, and identifiers in code font. No `simply`, `just`, `easily`,
  `obviously`, `note that`, `e.g.`, `i.e.`, `etc.`
- Don't write timelessly-rotting words: `currently`, `now`, `new`, `latest`,
  `recently`, `state of the art`.
- Introduce every list, table, and code block with a complete sentence.

## Lesson file

Path: `content/lessons/NN-slug.md`, where `NN` is the two-digit order prefix and
`slug` matches the `slug` field.

The file starts with YAML front matter:

```yaml
---
title: Temporal-difference learning
slug: 07-temporal-difference-learning
part: "Part 2 — Learning from experience"
summary: One sentence, under 140 characters, that says what the lesson delivers.
minutes: 55
gpu: false
objectives:
  - Update a value estimate toward a bootstrapped target after every step.
  - Implement SARSA and Q-learning, and explain why they walk different paths on the cliff.
  - "Say what bootstrapping trades: lower variance for bias from the current estimate."
prereqs:
  - 05-dynamic-programming
  - 06-monte-carlo-methods
exercise: 07-temporal-difference-learning
---
```

Every value that holds a colon must be wrapped in double quotes, or YAML reads
the item as a mapping rather than a string. A mapping that reaches the client
renders as nothing at all, so `scripts/check_content.py` rejects one.

Every field is required except `prereqs`, which can be omitted. `gpu` is `true`
only for the three lessons in [Lessons with a GPU run](#lessons-with-a-gpu-run). `minutes` is an honest reading and
working estimate.

The body is Markdown with these conventions:

- One `#` heading, matching `title`, as the first line after the front matter.
- `##` for sections, `###` for subsections. Never skip a level.
- Math in KaTeX: `$inline$` and `$$display$$`. Put each `$$` on its own line
  above and below the equation. A one-line `$$...$$` parses as inline math and
  renders at text size, so `scripts/check_content.py` rejects it. The renderer loads
  `remark-math` and `rehype-katex`, so use KaTeX-supported commands only.
  `\operatorname`, `\mathbb`, `\mathcal`, `\begin{bmatrix}`, `\begin{aligned}`,
  `\top`, `\lVert`, and `\argmin` via `\operatorname*{arg\,min}` all work.
- Fenced code blocks tagged with a language. Python blocks show the shape of an
  implementation, not a full solution to the exercise.
- Tables for genuinely two-dimensional data.
- Callouts, written as a blockquote whose first line is a type marker. The
  reader renders each type as a coloured box with its own label:

  ```markdown
  > [!KEY]
  > A trained model's error on its own training data is lower than its true
  > error, on average.
  ```

  | Marker | Label | Use it for |
  |---|---|---|
  | `[!KEY]` | Key idea | The one sentence to remember from a section. Two to four per lesson. |
  | `[!TIP]` | Tip | A practical shortcut or habit. At most two per lesson. |
  | `[!WARNING]` | Watch out | A trap the reader is about to fall into. At most two per lesson. |
  | `[!RECAP]` | Before you build | Three to five bullets, placed right before `## Build it step by step`. One per lesson. |

  `scripts/check_content.py` rejects any other marker, and a second recap.

Every lesson body includes, in this order:

1. An opening that says, in plain words, what problem the lesson solves and why
   it matters. Two or three short paragraphs. No heading, and no formulas.
2. `## What you already know` — a short list of the ideas the lesson builds on,
   each in one plain sentence. Keep it to five bullets.
3. The substance, in as many `##` sections as the material needs. Each section
   leads with intuition, then gives the formula, reads it aloud, and shows it
   as numpy. Include at least one small example worked by hand with real
   numbers.
4. `## Build it step by step` — a numbered walkthrough of the exercise, one
   step per function in the order the checks test them. Each step says what the
   function takes and returns, the numpy operations it needs, and the mistake
   people most often make. It gives the shape of the code, never the full
   answer.
5. `## Cost and scaling` — how the running time and memory grow as the data or
   the model gets bigger, in plain words. A short table works well.
6. `## Where this breaks` — the common ways the method fails in practice, and
   what people do about each one.
7. `## Further reading` — three to eight papers or book chapters, each a
   Markdown link to its arXiv abstract, DOI, or official page, with one
   sentence on what it adds. In Parts 7 to 10, end the section with two or
   three open questions a researcher is working on, each in one sentence.
8. `## The exercise` — one or two paragraphs that say what the reader builds
   and what the checks verify. Name the functions by their exact identifiers.

Aim for 2000 to 3500 words of body, plus math. Short sentences and short
paragraphs matter more than the word count.

## Exercise directory

Path: `content/exercises/<id>/`, where `<id>` matches the lesson's `exercise`
field. It holds exactly four files.

### `meta.json`

```json
{
  "id": "07-temporal-difference-learning",
  "title": "TD(0), SARSA, and Q-learning on the cliff",
  "brief": "Estimate values from single steps, then learn to control the cliff walk two ways and see them choose different paths.",
  "needs_gpu": false,
  "timeout": 60,
  "packages": ["numpy"],
  "hints": [
    "The TD error is r + gamma * V[s2] * (not terminated) - V[s]. Forgetting the terminal mask bootstraps from a state that has no future.",
    "SARSA's target uses the action you will actually take next; Q-learning's uses the max over actions, whatever you take."
  ]
}
```

`packages` is documentation only; the runtime image already carries `numpy`,
`scipy`, and `torch` on the CPU. `timeout` is in seconds and must leave headroom:
set it to about four times the reference solution's runtime, and never above 120
for a CPU exercise.

### `starter.py`

What the reader sees when the lesson opens. It must:

- Import only from the standard library, `numpy`, `rlenvs` (see below), and,
  when the lesson needs it, `torch`.
- Define every function the checks call, with the exact signature, a docstring
  that states the contract including array shapes, and a body that raises
  `NotImplementedError`.
- Carry a `if __name__ == "__main__":` block that builds a small example and
  prints something informative. The **Run** button executes this block, so it must
  work once the reader fills in the functions, and it must fail with a clear
  `NotImplementedError` before that.
- Never contain the answer, in a comment or anywhere else.

The **Run** button has the same time limit as the checks, so keep the
`__main__` block to a few seconds of work.

### `solution.py`

A correct, idiomatic reference implementation with the same public surface as
`starter.py`. It must pass every check in `tests.py`. Keep it clean enough to read
as the worked answer, because the reader can reveal it.

### `tests.py`

The checks. Import the harness helpers and register each check with the
`@check` decorator:

```python
"""Checks for the temporal-difference exercise."""

import numpy as np

import rlenvs
from rlcheck import check, close, fail, has


@check("td0_update moves V[s] a fraction alpha toward the one-step target")
def test_td0_update(sub):
    td0_update = has(sub, "td0_update")
    V = np.array([0.0, 0.5, 1.0])
    new = td0_update(V.copy(), s=1, r=1.0, s2=2, terminated=False, alpha=0.1, gamma=0.9)
    # Target 1 + 0.9 * 1.0 = 1.9, error 1.4, step 0.14.
    close(new, np.array([0.0, 0.64, 1.0]), atol=1e-12)


@check("Q-learning on the cliff finds the 13-step path along the edge")
def test_q_learning_cliff(sub):
    q_learning = has(sub, "q_learning")
    mdp = rlenvs.cliff_walking()
    Q = q_learning(mdp, episodes=500, alpha=0.5, epsilon=0.1, seed=0)
    v = rlenvs.policy_evaluation(mdp, rlenvs.greedy(Q))
    start = mdp.names["start"]
    # policy_evaluation returns nan for states that never terminate, so test
    # with isclose: a comparison with nan is always false.
    if not np.isclose(v[start], -13.0, atol=1e-9):
        fail(f"the greedy policy's value at the start should be -13, got {v[start]:.3f}")
```

The `rlcheck` module is on the path at run time. Its API:

| Name | Signature | Behaviour |
|---|---|---|
| `check` | `@check("description")` | Registers a check. The description is what the reader sees. |
| `close` | `close(got, want, rtol=1e-5, atol=1e-8, label=None)` | Fails with a readable diff when the arrays differ. |
| `equal` | `equal(got, want, label=None)` | Exact equality, for shapes, dtypes, and integers. |
| `fail` | `fail("message")` | Fails the check with that message. |
| `shape` | `shape(array, (3, 4), label=None)` | Asserts a shape. |
| `almost` | `almost(got, want, tol, label=None)` | Scalar comparison within `tol`. |
| `timed` | `with timed("matmul") as t:` | Records elapsed seconds, reported to the reader. |
| `has` | `has(sub, "name")` | Returns the reader's function, or fails with a message naming what is missing. |
| `finite` | `finite(array, label=None)` | Fails if any value is NaN or infinite. |
| `finite_difference` | `finite_difference(f, x, eps=1e-6)` | Central-difference gradient of a scalar function at `x`. |

Rules for checks:

- Each check receives `sub`, the reader's module, as its only argument.
- Seed every random number generator. `np.random.default_rng(0)` for numpy,
  `torch.manual_seed(0)` for torch. A check that passes intermittently is a bug.
- Write six to ten checks per exercise, ordered from the smallest piece to the
  whole. The first check must pass as soon as the reader gets one function right,
  so the pane shows progress early.
- Check the mathematics, not the implementation. Verify a gradient against a
  finite-difference approximation rather than against a particular loop.
- Never import `solution` from `tests.py`.
- Keep the whole suite fast. The reference solution must pass every check in
  under 20 seconds on a laptop, because the app container shares one and a half
  CPUs among every reader. That rules out long training runs: prefer exact
  answers on tiny MDPs, closed forms, and gradient checks, and keep any
  learning curve to a few thousand environment steps or a few hundred
  gradient updates on a network with fewer than ten thousand parameters.
- A check that trains something must use a threshold with real margin. Run the
  solution under at least five seeds while writing the check, and set the
  threshold so every one passes comfortably, then fix the seed in the check.
- Call `torch.set_num_threads(1)` at the top of any `tests.py` or `solution.py`
  that uses torch. Thread start-up costs more than a small network's work.
- Failure messages must say what was expected and what arrived. The reader cannot
  see your code.

### Lessons with a GPU run

Three lessons set `gpu: true` in their front matter and `"needs_gpu": true` in
`meta.json`: `35-rlhf-with-ppo`, `37-group-relative-policy-optimization`, and
`38-reasoning-with-verifiable-rewards`. The flag sends both **Run** and
**Check** to a RunPod worker with a CUDA GPU, when one is configured, and to the
local CPU runner otherwise. So in these lessons:

- `tests.py` is exactly like every other lesson's: CPU only, toy-sized, under
  20 seconds, and never imports `transformers`.
- The `__main__` block in `starter.py` and `solution.py` runs the reader's
  functions on a real model: `Qwen/Qwen3-0.6B` or `Qwen/Qwen3-0.6B-Base` from
  Hugging Face, loaded with `transformers` on `cuda`. Keep the trained
  policy's weights in float32 and run it under bfloat16 autocast, because
  bfloat16 weights round away updates at learning rates near 1e-6. A frozen
  reference model can be loaded in bfloat16. It must
  finish in under ten minutes on a 24 GB GPU, and print a short table of
  progress, such as mean reward and KL every few steps.
- The block must check `torch.cuda.is_available()` first. Without a GPU, or
  without `transformers` installed, it prints one line saying the real run
  needs the GPU worker, then runs the same functions on the toy policy from the
  checks, so **Run** still works on the CPU.
- Set `"timeout": 900` in `meta.json`. The local runner caps it at
  `LOCAL_TIMEOUT_CEILING`, 180 seconds, when no GPU is configured.
- The functions the checks test must work on plain tensors of log-probabilities,
  rewards, and masks, so that the same code serves the toy and the real model.

### The shared environments: `rlenvs`

`backend/app/runner/rlenvs.py` sits next to `rlcheck` on the import path, so a
starter, a solution, and a test suite can all `import rlenvs`. Use it rather than
writing a new grid world in every exercise, so that the reader meets the same
worlds again and again. Read its docstrings for the full contract. It provides:

| Name | What it is |
|---|---|
| `BernoulliBandit(means, seed)`, `GaussianBandit(means, sigma, seed)` | `k` arms with `.pull(arm)`, `.means`, `.best_mean`, and `.regret(arm)`. |
| `TabularMDP` | A finite MDP carrying `P` `(S, A, S)`, `R` `(S, A)`, `gamma`, `mu0`, and `terminal`, with Gym-style `reset(seed)` and `step(a)` returning `(s2, r, terminated, truncated)`, and `sample(s, a, rng)` for a model query. |
| `random_mdp(n_states, n_actions, gamma, seed, branching)` | A random MDP for exact checks. |
| `gridworld(layout, ...)` | A grid world from rows of characters: `.` open, `#` wall, `S` start, `G` goal, `H` hole, `C` cliff. Actions are `0` up, `1` right, `2` down, `3` left. |
| `cliff_walking()`, `frozen_lake()`, `river_swim()` | Three classic worlds. The optimal cliff walk from the start is worth exactly `-13`. |
| `policy_evaluation(mdp, pi)`, `q_from_v(mdp, v)`, `value_iteration(mdp)`, `greedy(q)`, `uniform_policy(mdp)`, `state_distribution(mdp, pi)` | Exact answers, for checks to compare against. With `gamma = 1`, `policy_evaluation` returns `nan` for states from which the policy never terminates, so compare with `np.isclose`, never with `abs(...) > tol`. |
| `CartPole(max_steps, seed)`, `Pendulum(max_steps, seed)` | The Gym physics in numpy. `CartPole.dynamics` also steps a batch of states at once. |

A policy for a tabular MDP is always an array of shape `(S, A)` whose rows sum
to one. When an exercise needs something `rlenvs` lacks, such as a toy
language model or a small diffusion model, define it in the starter file and
repeat the identical definition in `solution.py`, so that the checks can rely
on it. Do not edit `rlenvs.py` from an exercise.

## Mathematical accuracy

Every result, gradient, and complexity claim in a lesson must be correct. A
lesson that states a result without deriving it carries the same burden as one
that derives it: if it is on the page, it is right. When you write a gradient in
prose, the check that verifies it in `tests.py` must use finite differences, so
the prose and the code confirm each other. If those two disagree, the lesson is
wrong, not the reader.

Cutting a derivation must never change what a result says. Shortening the path
to a formula is editing; changing the formula is breaking the lesson.

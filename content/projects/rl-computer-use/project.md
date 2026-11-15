---
title: Teaching a vision-language model to use a browser
summary: Post-train Qwen3-VL-4B with GRPO on one rented GPU so it finishes more MiniWoB++ web tasks, then fix what the first run broke.
date: 2026-11-15
status: Finished
tags:
  - GRPO
  - Computer use
  - LoRA
lessons:
  - 071-group-relative-policy-optimization
  - 073-reasoning-with-verifiable-rewards
  - 065-sft-and-the-post-training-pipeline
  - 078-agents-and-multi-turn-rl
  - 084-capstone-post-training
  - 059-evaluating-rl
links:
  - label: Code (private GitHub repository)
    url: https://github.com/weights-and-wires/RL-ComputerUse
  - label: Adapters and full results (private Hugging Face repository)
    url: https://huggingface.co/weights-and-wires/qwen3vl-4b-miniwob-grpo
---

# Teaching a vision-language model to use a browser

This project takes a small open model that can read screenshots, puts it in
front of a real web browser, and trains it with reinforcement learning until it
finishes more simple web tasks. It ran twice. The first run raised success on
the tasks it practiced from 27% to 58%, but it quietly broke skills it never
practiced. The second run fixed that and went further: 72.6% on a broader set
of tasks, up from 46.6% for the original model.

Every idea here has a lesson in the course. This page links to each one where it
first comes up, so you can stop and read the lesson, then come back.

## The goal

*Computer use* means operating software the way a person does: look at the
screen, then click and type. The goal was to make a small model better at it,
using only one rented GPU and a few hours of training per run.

Three pieces make up the setup:

- **The model.** `Qwen/Qwen3-VL-4B-Instruct` is a *vision-language model*: a
  language model that also reads images. It has about 4 billion parameters,
  which is small enough to train on one 80 GB GPU.
- **The tasks.** MiniWoB++ is a collection of more than 100 tiny web pages, each
  with one job, such as "enter this date and press Submit." Each page checks by
  itself whether the job got done, so no person or judge model has to grade the
  attempt.
- **The browser.** A headless Chromium browser, which has no window on screen,
  runs each page. A program called BrowserGym drives it and reports what the
  page looks like after every action.

## How the model acts

The model plays a short game with the page, one turn at a time. Each turn it
receives three things: the task in words, a screenshot, and the page's
*accessibility tree*, a text list of the page's boxes and buttons with a number
for each. It answers with exactly one action that names an element by its
number, such as `click("12")` or `fill("7", "11/15/2026")`. The browser carries
out the action and sends back the new page.

One full attempt, from the first page to the end, is an *episode*. An episode
ends when the page reports success or failure, or after 8 turns.

Picking an element by number is much easier than guessing pixel positions, which
is why gains showed up within hours. Clicking by screen position instead is the
harder setting, left for a later run.

In reinforcement-learning terms, the model is the *policy*: the rule that maps
what it sees to what it does. [Language models as policies](/lessons/066-language-models-as-policies)
explains how a model that writes tokens can be treated as one.

## The reward

The *reward* is the score the model gets for an episode: 1 if the page reports
the task done and 0 if not. A small penalty applies to replies the program can't
parse as an action, and a small bonus rewards finishing in fewer turns, but only
on successful episodes. A bonus for short failed episodes would teach the model
to give up quickly.

A reward that a program checks exactly is called a *verifiable reward*. It's
hard to fool, because it measures the thing you care about. For more about
rewards like this, see [Verifiable rewards](/lessons/072-verifiable-rewards).

## How training works

Training has two stages: a warm start, then practice rounds with GRPO.

### The warm start

Before the main training, the model tries each practice task 32 times. Only the
successful attempts with no format errors are kept, at most 64 per task, and the
model is fine-tuned once on those, like copying its own best work. This is
*rejection-sampling fine-tuning*: draw samples, reject the bad ones, and train on
the rest. [Supervised fine-tuning and the post-training pipeline](/lessons/065-sft-and-the-post-training-pipeline#s-keep-the-highest-scoring-samples-then-copy-them)
builds it.

The warm start does two jobs. It teaches the answer format, and it moves tasks
from "almost never solved" toward "sometimes solved," where the next stage can
learn.

### Practice rounds with GRPO

*GRPO* (group-relative policy optimization) is the training method. For one
task page, the model makes 8 attempts, a *group*. Each attempt's *advantage* is
its reward minus the group's average reward, divided by the group's spread.
Attempts that beat the average become more likely, and the rest become less
likely. The group is its own baseline, so GRPO needs no second network to
predict rewards. [Group-relative policy optimization](/lessons/071-group-relative-policy-optimization)
derives the loss.

For example, if 3 of 8 attempts succeed, the average is 0.375. Each success gets
1 - 0.375 = +0.625 before scaling, and each failure gets 0 - 0.375 = -0.375.

Each *step*, or practice round, used 8 groups, so 64 episodes. The update clips
how far one step can move the policy, as PPO does; see
[Clip the ratio](/lessons/036-proximal-policy-optimization#s-clip-the-ratio).
This project used a lower clip of 0.2 and a higher upper clip of 0.28, a choice
called *clip-higher*, so rare good actions can grow faster.

Two more details matter for multi-turn episodes:

- The loss counts only the tokens the model wrote. The screenshot and page text
  come from the browser, so the loss masks them out. For more information, see
  [Mask what the agent did not write](/lessons/078-agents-and-multi-turn-rl#s-mask-what-the-agent-did-not-write).
- Every turn in an episode shares the episode's advantage. That's crude credit
  assignment: a mistake in turn 3 also penalizes turns 1 and 2. Short episodes
  limit the damage.

### A small add-on instead of the whole model

Training changes a *LoRA* adapter, not the model's own weights. LoRA (low-rank
adaptation) adds a small pair of matrices beside each weight matrix and trains
only those. Here the adapter had rank 32 and was under 2% of the model's size,
a 260 MB file. That keeps memory low enough for one GPU and makes it cheap to
copy the new weights to the fast generation engine after every step.
[LoRA: train a small change, not the matrix](/lessons/084-capstone-post-training#s-lora-train-a-small-change-not-the-matrix)
builds a LoRA layer by hand.

One GPU did both jobs. vLLM, an inference engine, generated the attempts, and
PyTorch computed the loss and gradients, taking turns on the same card. Thirty-two
browsers ran in parallel on the CPU.

### Fair tests

Every page is generated from a number called a *seed*, which decides details
such as the date to type. Practice used seeds 1000 and up. Tests used seeds 0 to
99, which training never saw, so the model can't pass by memorizing answers.

Tests use *greedy* decoding: the model always picks its most likely next token,
so the score doesn't depend on luck. A score of 72% means the model finished 72
of every 100 test pages. Some tasks were kept out of practice entirely, to show
whether skills carry over to new kinds of tasks.

## Run 1

Run 1 picked 10 practice tasks that the original model solved between about 10%
and 80% of the time, then ran the warm start and 158 GRPO steps in 5 hours.

The following table shows greedy success on run 1's 10 practice tasks, 50 test
pages each:

| Model | Success |
|---|---|
| Original model | 27.3% |
| After the warm start only | 32.0% |
| After run 1 | 57.8% |

The warm start added about 5 points and the GRPO steps about 26 more. Three
tasks went to nearly 100%: entering a time, finding a word in a paragraph, and
entering a date. The date task went from 0% to 98% during GRPO alone. The warm
start had kept no successful date episodes, so the model found the date format
that the page accepts by trial and error.

### Most attempts taught nothing

The training logs held a warning. If all 8 attempts in a group succeed, or all
8 fail, every advantage is 0 and the group teaches nothing. In run 1, 77% of
groups were like that. Each step trained on about 13 of its 64 episodes, and 20
of the 158 steps trained on none.

For a task the model solves with probability $p$, the chance that a group of
$G$ attempts has mixed results is $1 - p^G - (1 - p)^G$. The following table
shows that chance for $G = 8$:

| Task success rate | Chance a group of 8 is mixed |
|---|---|
| 5% or 95% | 34% |
| 10% or 90% | 57% |
| 20% or 80% | 83% |
| 50% | 99% |
| 98% | 15% |

Tasks drift out of the useful band as training works: once the model solves a
task almost every time, its groups stop teaching. [Count the groups that teach nothing](/lessons/073-reasoning-with-verifiable-rewards#s-count-the-groups-that-teach-nothing)
works this probability out, and [Keep the prompts that can teach](/lessons/072-verifiable-rewards#s-keep-the-prompts-that-can-teach)
shows how to filter for it.

### What run 1 broke

Later tests on a wider set of tasks showed a problem. Run 1 got much worse at
some tasks the original model already did well, even though it never practiced
them. Losing old skills while learning new ones is called *forgetting*.

The following table shows the four worst cases, with 50 test pages each:

| Task | Original | After run 1 | After run 2 |
|---|---|---|---|
| `guess-number` | 100% | 18% | 100% |
| `form-sequence-2` | 100% | 52% | 100% |
| `choose-list` | 98% | 62% | 98% |
| `use-spinner` | 86% | 52% | 96% |

Run 1's own check on never-practiced tasks used only 5 tasks, and it happened to
miss these. A narrow test can hide a model getting worse elsewhere.

## Run 2

Run 2 started from run 1's adapter and changed how it spent its compute,
because run 1 wasted most of it. It made five changes:

1. **Re-pick the practice tasks with the run-1 model.** The model being trained
   decides which tasks are informative, so the sampled baseline ran again with
   the run-1 adapter on about 60 candidate tasks. Run 2 kept up to 16 tasks it
   solved between 5% and 85% of the time, including the forgotten ones.
2. **Weight tasks by how much they can teach.** Instead of taking turns through
   the tasks, run 2 kept a running estimate of each task's success rate $p$ and
   picked tasks with weight $\max(1 - p^8 - (1 - p)^8, 0.05)$. The floor of 0.05
   keeps a little practice on mastered and unsolved tasks, so the trainer notices
   if one changes.
3. **Skip groups that teach nothing.** *Dynamic sampling*, from the DAPO paper,
   rolls out more groups than it needs and trains only on the mixed ones. Run 2
   rolled out rounds of 8 groups, up to 3 rounds per step, until it had 6 mixed
   groups. Each step then trained on up to 48 episodes instead of about 13.
4. **Make training faster and more stable.** The trainer computed the model's
   output scores only at the positions of the model's own tokens, which removed
   a 6 GB tensor per batch. Each step also made two updates instead of one, so
   the second update ran on weights that had already moved and the clipping in
   the loss took effect.
5. **Measure more carefully.** Run 2 doubled the never-practiced test set to 10
   tasks with 100 seeds each, and compared every pair of models on the same
   pages with a paired bootstrap interval. For how a bootstrap interval works, see
   [Put an interval on it](/lessons/059-evaluating-rl#s-put-an-interval-on-it-the-stratified-bootstrap).

Run 2 made 52 steps in 5 hours, about a third as many as run 1, but each step
took 4 to 7 minutes and trained on far more useful episodes.

## Results

All numbers are greedy success on test pages that training never used. Practice
tasks use 50 pages per task, and never-practiced tasks use 100.

| Task set | Original model | After run 1 | After run 2 |
|---|---|---|---|
| Run 2's 16 practice tasks | 46.6% | 37.8% | 72.6% |
| Run 1's 10 practice tasks | 27.3% | 57.8% | 61.6% |
| 10 never-practiced tasks | 62.9% | 66.6% | 67.3% |

Read the table one row at a time:

- **Run 2's practice tasks.** Run 2 beat run 1 by 34.9 points, with a 95%
  interval of +32.1 to +37.6. Run 1 sits below the original model on this row
  because these tasks include the ones it forgot.
- **Run 1's practice tasks.** Run 2 kept run 1's gains and added 3.8 points, with
  an interval of +1.8 to +6.0, mostly from two email and social-media tasks.
- **Never-practiced tasks.** Run 1 added a small, real gain of 3.7 points
  (+1.6 to +5.9). Run 2's extra 0.7 points has an interval of -1.0 to +2.3, which
  includes zero, so it can't be told apart from luck.

Some of run 2's biggest single-task gains over run 1 were `copy-paste-2` (10% to
100%), `email-inbox-important` (8% to 100%), `guess-number` (18% to 100%), and
`email-inbox-star-reply` (22% to 72%). A few tasks barely moved in either run,
such as `click-collapsible-2` at 4% and `click-menu` at 28%.

Run 1 cost about $26 and run 2 about $15, on one rented A100 80 GB GPU at $2.19
per hour.

## What we learned

- **Practice where the model is unsure.** A task the model always solves, or
  never solves, gives groups with no signal. Choosing tasks by their chance of a
  mixed group, and skipping unmixed groups, gave run 2 roughly four times as
  much learning per step.
- **Test broadly.** Run 1 looked like a clear win on its own tasks while it lost
  skills elsewhere. Only a wider test set showed the damage.
- **Pick the practice set with the model you're training.** Tasks that were
  hard for the original model were easy for the run-1 model, so the selection
  had to run again.
- **Skills carried over only a little.** Never-practiced tasks went from 63% to
  67%. Practice mainly improved the tasks it practiced, so a broader practice
  mix is the main lever for general skill.
- **A verifiable reward made this cheap.** The pages grade themselves, so no
  reward model or human labels were needed. For the ways a reward can still go
  wrong, see [Reward hacking, safety, and the frontier](/lessons/083-reward-hacking-and-safe-rl).

Possible next steps include a much wider task mix, and the harder setting where
the model clicks by screen position instead of by element number.

## How to reproduce it

The code lives in the private `weights-and-wires/RL-ComputerUse` repository. Its
`RUN_IT_YOURSELF.md` file has the full procedure, and `docs/RUNBOOK.md` has
recovery steps. The outline is as follows:

1. Rent one 80 GB GPU pod. The repository's `scripts/runpod_up.sh` creates one on
   Runpod and copies your Hugging Face and Weights & Biases tokens to it.
2. On the pod, inside `tmux`, run the one-time setup script
   `scripts/pod_bootstrap.sh`. It installs the dependencies, downloads the model,
   and runs the tests.
3. Run all of run 1: the smoke test, sampled baseline, task selection, base
   evaluations, warm start, GRPO, final evaluations, and report.

   ```bash
   PRICE_PER_HR=PRICE bash scripts/saturday.sh
   ```

   Replace `PRICE` with the pod's hourly price in dollars, which the script uses
   to report spend. The script skips phases that already finished, so if it
   stops, run the same command again.

4. Run 2 starts from run 1's adapters. Its launch command and settings are in
   `docs/PLAN_RUN2.md`, and `bash scripts/run2.sh --status` shows its progress.
5. Copy the results to your computer and delete the pod. The pod bills every
   hour until you stop or delete it.

To watch the trained model work without a GPU pod, run `scripts/demo.sh` on your
laptop. It downloads the adapters (it needs a Hugging Face token with access to
the private repository), opens a local page where you pick a task, and shows
each turn: the screenshot, the model's reasoning, its action, and the reward.
Pick **Original model**, **Run 1**, or **Run 2** to compare them.

## Further reading

These sources are the most useful companions to this project:

- [DeepSeekMath](https://arxiv.org/abs/2402.03300), section 4, introduces GRPO.
- [DAPO](https://arxiv.org/abs/2503.14476) describes clip-higher and dynamic
  sampling.
- [Dr. GRPO](https://arxiv.org/abs/2503.20783) explains a length bias in the
  original GRPO loss and the token-level fix this project used.
- [WebAgent-R1](https://arxiv.org/abs/2505.16421) applies multi-turn RL to web
  agents.
- [BrowserGym](https://browsergym.readthedocs.io/) documents the browser
  environment and the MiniWoB++ tasks.
- [LoRA](https://arxiv.org/abs/2106.09685) introduces low-rank adaptation.

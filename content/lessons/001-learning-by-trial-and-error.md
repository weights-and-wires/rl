---
title: Learning by trial and error
slug: 001-learning-by-trial-and-error
part: "Part 1 — Learning from reward"
summary: Meet the agent-environment loop, see why a reward is not a label, and run a random agent that logs every state, action, and reward.
minutes: 35
gpu: false
objectives:
  - Say how a reward differs from a supervised label, and why that difference makes learning harder.
  - Name the parts of the agent-environment loop and trace one step of it by hand.
  - Tell a terminated episode from a truncated one.
  - Run a random agent in a small room and log the state, action, and reward of every step.
exercise: 001-learning-by-trial-and-error
---

# Learning by trial and error

Picture a row of three slot machines. Each pull costs one coin, and each machine pays back one coin or nothing, at a rate that you cannot see. You have a thousand coins. Nobody tells you which machine is best, and nobody tells you which one to pull next. All you ever learn is what happened after the pulls you made.

That is a different kind of problem from the ones in a first machine learning course. There, every training example comes with the right answer, and a loss says how far off you were. Here, the only feedback is a score for what you did, and it says nothing about the pulls you did not make. Learning to act well from that kind of feedback is called *reinforcement learning*.

This chapter sets up the vocabulary and the loop that every later chapter runs inside. The three machines return as the running example in the bandit chapters, and a small room with a goal carries the chapters after them. You build a random agent, run it, and log every step it takes.

## What you already know

- A supervised model learns from pairs of an input and a correct output.
- A loss is a number that says how wrong a prediction was.
- An expected value is the long-run average of a random quantity.
- `np.random.default_rng(seed)` makes a reproducible random number generator.

## A row of slot machines

Make the row concrete. Machine 0 pays back with probability 0.1, machine 1 with 0.5, and machine 2 with 0.9. You will see these numbers in the code, but the player never does. Figure 1 shows what a player who knows nothing sees after ten pulls at random.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Three slot machines with hidden payout rates, and the outcomes of ten random pulls: machine 0 paid once in six pulls, machine 1 paid twice in two, machine 2 paid once in two">
<rect x="40" y="20" width="160" height="110" rx="10" class="fill-soft-a stroke-a"/>
<text x="120" y="46" text-anchor="middle" class="fig-bold">machine 0</text>
<text x="120" y="72" text-anchor="middle">payout rate: ?</text>
<text x="120" y="100" text-anchor="middle" class="fig-small">pulled 6 times</text>
<text x="120" y="118" text-anchor="middle" class="fig-small">paid 1 time</text>
<rect x="230" y="20" width="160" height="110" rx="10" class="fill-soft-a stroke-a"/>
<text x="310" y="46" text-anchor="middle" class="fig-bold">machine 1</text>
<text x="310" y="72" text-anchor="middle">payout rate: ?</text>
<text x="310" y="100" text-anchor="middle" class="fig-small">pulled 2 times</text>
<text x="310" y="118" text-anchor="middle" class="fig-small">paid 2 times</text>
<rect x="420" y="20" width="160" height="110" rx="10" class="fill-soft-a stroke-a"/>
<text x="500" y="46" text-anchor="middle" class="fig-bold">machine 2</text>
<text x="500" y="72" text-anchor="middle">payout rate: ?</text>
<text x="500" y="100" text-anchor="middle" class="fig-small">pulled 2 times</text>
<text x="500" y="118" text-anchor="middle" class="fig-small">paid 1 time</text>
<text x="40" y="170" class="fig-bold">The ten pulls of a random agent, in order</text>
<circle cx="62" cy="208" r="20" class="fill-soft-c stroke-c"/>
<text x="62" y="213" text-anchor="middle">2</text>
<text x="62" y="244" text-anchor="middle" class="fig-small">paid</text>
<circle cx="116" cy="208" r="20" class="fill-soft-c stroke-c"/>
<text x="116" y="213" text-anchor="middle">1</text>
<text x="116" y="244" text-anchor="middle" class="fig-small">paid</text>
<circle cx="170" cy="208" r="20" class="fill-soft-c stroke-c"/>
<text x="170" y="213" text-anchor="middle">1</text>
<text x="170" y="244" text-anchor="middle" class="fig-small">paid</text>
<circle cx="224" cy="208" r="20" class="fill-soft-c stroke-c"/>
<text x="224" y="213" text-anchor="middle">0</text>
<text x="224" y="244" text-anchor="middle" class="fig-small">paid</text>
<circle cx="278" cy="208" r="20" class="fill-sunk stroke-muted"/>
<text x="278" y="213" text-anchor="middle">0</text>
<text x="278" y="244" text-anchor="middle" class="fig-small">0</text>
<circle cx="332" cy="208" r="20" class="fill-sunk stroke-muted"/>
<text x="332" y="213" text-anchor="middle">0</text>
<text x="332" y="244" text-anchor="middle" class="fig-small">0</text>
<circle cx="386" cy="208" r="20" class="fill-sunk stroke-muted"/>
<text x="386" y="213" text-anchor="middle">0</text>
<text x="386" y="244" text-anchor="middle" class="fig-small">0</text>
<circle cx="440" cy="208" r="20" class="fill-sunk stroke-muted"/>
<text x="440" y="213" text-anchor="middle">0</text>
<text x="440" y="244" text-anchor="middle" class="fig-small">0</text>
<circle cx="494" cy="208" r="20" class="fill-sunk stroke-muted"/>
<text x="494" y="213" text-anchor="middle">0</text>
<text x="494" y="244" text-anchor="middle" class="fig-small">0</text>
<circle cx="548" cy="208" r="20" class="fill-sunk stroke-muted"/>
<text x="548" y="213" text-anchor="middle">2</text>
<text x="548" y="244" text-anchor="middle" class="fig-small">0</text>
</svg>
Figure 1. Ten random pulls on the row. Machine 1 paid both times it was pulled, so it looks perfect, yet machine 2 is the best. Ten pulls are too few to tell.
```

Machine 1 is 2 for 2 and machine 2 is only 1 for 2, so a player who trusts the log would call machine 1 the best. The truth is the other way around, because each machine pays at random.

> [!KEY]
> A reward tells you what happened after one action, once. It does not tell you what the best action was, and it is noisy.

## Why a reward is not a label

Why not treat the log as training data and fit a classifier? Because a classifier needs a label that says which machine was right, and the log has no such column. The log has outcomes, and an outcome depends on luck as well as on the choice.

The next table lists what changes when you move from labels to rewards.

| | Supervised learning | Reinforcement learning |
|---|---|---|
| Feedback | The correct answer for every example | A score for the action you took |
| Untried choices | The label covers them | You learn nothing about them |
| Noise | The label is fixed | The same action can pay differently each time |
| Where data comes from | A dataset someone collected | Your own choices decide what you see next |
| Timing | The label arrives with the example | The reward can arrive long after the action that earned it |

The last two rows matter most. Because your choices decide what you see, an agent that never pulls machine 2 never finds out that it is good. Because rewards can be late, a chess move can earn its reward thirty moves after the move itself. The slot machines have no delay, and the room below adds some.

## The agent-environment loop

Every reinforcement learning problem has the same shape. The *agent* is the program that chooses. The *environment* is everything it acts on. At each time step $t$, three things happen:

1. The agent observes the *state* $s_t$, a description of the situation it is in.
2. The agent picks an *action* $a_t$.
3. The environment answers with a *reward* $r_t$ and the next state $s_{t+1}$.

Figure 2 draws the loop.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="The agent-environment loop: the agent sends an action to the environment, and the environment sends back a reward and the next state">
<defs><marker id="c001-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
<rect x="30" y="60" width="190" height="100" rx="10" class="fill-soft-b stroke-b"/>
<text x="125" y="95" text-anchor="middle" class="fig-bold">agent</text>
<text x="125" y="120" text-anchor="middle" class="fig-small">picks a with policy</text>
<text x="125" y="140" text-anchor="middle" class="fig-small">π(a | s)</text>
<rect x="380" y="60" width="190" height="100" rx="10" class="fill-soft-a stroke-a"/>
<text x="475" y="95" text-anchor="middle" class="fig-bold">environment</text>
<text x="475" y="120" text-anchor="middle" class="fig-small">answers with a reward</text>
<text x="475" y="140" text-anchor="middle" class="fig-small">and the next state</text>
<line x1="222" y1="84" x2="378" y2="84" class="stroke-ink" stroke-width="2" marker-end="url(#c001-arrow)"/>
<text x="300" y="72" text-anchor="middle">action a_t</text>
<line x1="378" y1="138" x2="222" y2="138" class="stroke-ink" stroke-width="2" marker-end="url(#c001-arrow)"/>
<text x="300" y="158" text-anchor="middle">reward r_t</text>
<text x="300" y="177" text-anchor="middle">next state s_t+1</text>
<text x="300" y="30" text-anchor="middle" class="fig-small">the agent sees state s_t</text>
<text x="300" y="212" text-anchor="middle" class="fig-small">then the loop repeats from s_t+1, with t one larger</text>
</svg>
Figure 2. The agent-environment loop. The agent sends actions, the environment sends back rewards and states, and the loop runs until the episode ends.
```

In this course, $r_t$ is the reward for taking $a_t$ in $s_t$. Some books call the same number $R_{t+1}$. The meaning is the same, and only the label differs.

The agent's rule for choosing is its *policy*, written $\pi(a \mid s)$: the probability that it takes action $a$ in state $s$. A policy can be deterministic, always choosing the same action in a state, or random. For a small problem, a policy is an array `pi` of shape `(n_states, n_actions)` whose rows each sum to one. The row `pi[s]` is the distribution over actions in state `s`.

The slot machines are the loop with one situation. Whatever you do, the next state looks the same, so the state carries no information, and the only thing to learn is which action pays. That special case is called a *bandit*. The room below has many states.

Many tasks come in *episodes*: runs that start in a start state and end in a *terminal state*, such as a won game or a reached goal. One episode is the sequence $s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T$, where $s_T$ is terminal and $r_{T-1}$ is the last reward. In code, one episode looks like this:

```python
s = mdp.reset(seed=0)
while True:
    a = rng.choice(mdp.n_actions, p=pi[s])     # sample from the policy
    s2, r, terminated, truncated = mdp.step(a)
    # log (s, a, r) here: all three belong to step t
    s = s2
    if terminated or truncated:
        break
```

> [!WARNING]
> Log the state *before* you call `mdp.step`. If you log `s2`, every state in your log shifts one step late, and each reward lines up with the wrong state.

## A room to walk in

The exercise uses a 3 by 3 room with a pillar in the middle. You start at `S` in the top-left corner, and the episode ends when you reach the goal `G` in the bottom-right corner:

```text
S . .
. # .
. . G
```

The actions are up, right, down, and left, numbered 0 to 3. Walking into a wall or the pillar leaves you where you are. Every step pays $-1$, so the reward punishes wasted time. The room has eight states, one per open cell, numbered row by row: state 0 is `S`, state 1 is the top middle, and state 7 is `G`.

The reward says what you want, not how to get it. Here, $-1$ per step means "reach the goal quickly". The shortest route takes four steps: right, right, down, down.

Run an agent that knows nothing. The *uniform random policy* picks each of the four actions with probability $1/4$ in every state. The next table is its first five steps from one seeded run.

| $t$ | State $s_t$ | Action $a_t$ | Reward $r_t$ | What happened |
|---|---|---|---|---|
| 0 | 0 | 1 (right) | $-1$ | Moved to state 1 |
| 1 | 1 | 0 (up) | $-1$ | Hit the top wall, stayed in 1 |
| 2 | 1 | 0 (up) | $-1$ | Hit the top wall again, stayed in 1 |
| 3 | 1 | 3 (left) | $-1$ | Moved back to state 0 |
| 4 | 0 | 3 (left) | $-1$ | Hit the left wall, stayed in 0 |

Five steps cost five, and the agent has not left the top-left corner. This particular episode lasted 35 steps. Averaged over many episodes the random agent takes exactly 32 steps to reach the goal, against 4 for the shortest route. Figure 3 shows where those 32 steps are spent.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="The three by three room. A bar in each cell shows how many times per episode a random agent is in that cell: 8.1 at the start, shrinking toward the goal">
<rect x="20" y="14" width="110" height="80" class="fill-paper stroke-grid"/>
<rect x="30" y="48" width="90.0" height="16" class="fill-a"/>
<text x="30" y="38" class="fig-small">S  8.1 visits</text>
<rect x="130" y="14" width="110" height="80" class="fill-paper stroke-grid"/>
<rect x="140" y="48" width="68.8" height="16" class="fill-a"/>
<text x="140" y="38" class="fig-small">6.2 visits</text>
<rect x="240" y="14" width="110" height="80" class="fill-paper stroke-grid"/>
<rect x="250" y="48" width="44.7" height="16" class="fill-a"/>
<text x="250" y="38" class="fig-small">4.0 visits</text>
<rect x="20" y="94" width="110" height="80" class="fill-paper stroke-grid"/>
<rect x="30" y="128" width="67.1" height="16" class="fill-a"/>
<text x="30" y="118" class="fig-small">6.1 visits</text>
<rect x="240" y="94" width="110" height="80" class="fill-paper stroke-grid"/>
<rect x="250" y="128" width="20.6" height="16" class="fill-a"/>
<text x="250" y="118" class="fig-small">1.9 visits</text>
<rect x="20" y="174" width="110" height="80" class="fill-paper stroke-grid"/>
<rect x="30" y="208" width="44.6" height="16" class="fill-a"/>
<text x="30" y="198" class="fig-small">4.0 visits</text>
<rect x="130" y="174" width="110" height="80" class="fill-paper stroke-grid"/>
<rect x="140" y="208" width="22.6" height="16" class="fill-a"/>
<text x="140" y="198" class="fig-small">2.0 visits</text>
<rect x="240" y="174" width="110" height="80" class="fill-paper stroke-grid"/>
<rect x="240" y="174" width="110" height="80" class="fill-soft-c stroke-grid"/>
<text x="295.0" y="219.0" text-anchor="middle" class="fig-bold">G (goal)</text>
<rect x="130" y="94" width="110" height="80" class="fill-sunk stroke-grid"/>
<text x="185.0" y="139.0" text-anchor="middle" class="fig-bold">pillar</text>
<text x="380" y="40" class="fig-bold">Visits per episode</text>
<text x="380" y="62" class="fig-small">mean over 1000 episodes of the</text>
<text x="380" y="80" class="fig-small">random agent in the 3 by 3 room.</text>
<text x="380" y="98" class="fig-small">Bar length shows the count.</text>
<text x="380" y="132" class="fig-small">all cells together: 32.3 steps</text>
<text x="380" y="150" class="fig-small">per episode (exact: 32)</text>
<text x="380" y="184" class="fig-small">the shortest route takes 4 steps</text>
</svg>
Figure 3. Time spent in each cell by the random agent. It lingers near the start, because a walk with no direction keeps stepping back.
```

A better policy exists, and finding it from rewards alone is what the rest of the course is about. A log of $(s_t, a_t, r_t)$ triples, like the table above, is the raw material for every method you will meet.

### Termination and truncation

An episode can stop for two different reasons, and later chapters depend on telling them apart. It *terminates* when it reaches a terminal state, where nothing more can happen. It is *truncated* when a time limit cuts it off, even though the task could have continued. The room's environment truncates after 500 steps. A truncated episode stopped early, so its last state still has a future that the logged rewards do not include.

> [!KEY]
> An agent acts, the environment answers with a reward and a next state, and the loop repeats. A reward scores one action once, so the agent must find good actions by trying them, which is why random play is the starting point and not the goal.

> [!RECAP]
> - The loop is observe $s_t$, act $a_t$, receive $r_t$ and $s_{t+1}$, and repeat until the episode terminates or is truncated.
> - A policy `pi[s]` is a row of action probabilities, and a random policy has every entry equal to $1/n_\text{actions}$.
> - A log of $(s_t, a_t, r_t)$ records an episode, with the state taken before the step.
> - A reward is noisy feedback on the action taken, not the right answer.

## Build it step by step

The checks test the functions in this order.

1. **`uniform_random_policy(n_states, n_actions)`.** Returns an array of shape `(n_states, n_actions)` with every entry $1/n_\text{actions}$. Use `np.full`. The common mistake is a row that does not sum to one, which makes `rng.choice` raise an error.
2. **`run_episode(mdp, pi, rng)`.** Reset with `mdp.reset(seed=int(rng.integers(2**31)))`, so the whole episode follows from `rng`. Draw each action with `rng.choice(mdp.n_actions, p=pi[s])`. Return `(states, actions, rewards, terminated)`, where `states[t]` is the state before step $t$. The common mistake is appending the next state instead of the current one. The checks walk the shortest route, cut an episode off with a five-step time limit, and count how often each action is drawn.
3. **`visit_counts(states, n_states)`.** Count how often each state appears. `np.bincount` with `minlength=n_states` does it in one call. The common mistake is leaving out `minlength`, which drops the unvisited states at the end of the array.
4. **`mean_episode_length(mdp, pi, n_episodes, seed)`.** Make one generator from `seed` and reuse it for every episode. The common mistake is making a fresh generator with the same seed for each episode, which repeats one episode `n` times.

## Where this breaks

**A reward that says the wrong thing.** The agent maximizes the reward you wrote, not the one you meant. A reward for touching a ball can teach a robot to vibrate against it. [Reward design and shaping](/lessons/047-reward-design-and-shaping) returns to this.

**Episodes that never end.** A policy that walks into a wall forever never terminates, so the loop needs a time limit. Cutting the episode off leaves out the rewards that would have followed, which later chapters handle on purpose.

## Further reading

- [Sutton and Barto, *Reinforcement Learning: An Introduction*, Chapters 1 and 3](http://incompleteideas.net/book/the-book-2nd.html) introduce the agent-environment loop. The book is free online and underlies much of this course.
- [Mnih and colleagues, "Human-level control through deep reinforcement learning" (2015)](https://doi.org/10.1038/nature14236) trains one network design, separately on each of 49 Atari games, from pixels and score alone.
- [Ouyang and colleagues, "Training language models to follow instructions with human feedback" (2022)](https://arxiv.org/abs/2203.02155) uses reinforcement learning from human preferences to post-train a language model.
- [DeepSeek-AI, "DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning" (2025)](https://arxiv.org/abs/2501.12948) trains long step-by-step reasoning with rewards that a program can verify.

## The exercise

You build a random agent and the loop that runs it. Write `uniform_random_policy(n_states, n_actions)`, `run_episode(mdp, pi, rng)`, `visit_counts(states, n_states)`, and `mean_episode_length(mdp, pi, n_episodes, seed)`. The environment is the 3 by 3 room from this chapter, built with `rlenvs.gridworld`.

The checks verify that the random policy has equal rows that sum to one. They check that `run_episode` walks the four-step route exactly, reproduces an episode from a seeded generator, stops at a time limit with `terminated` set to `False`, and draws actions with the policy's probabilities. Last, they check that `visit_counts` includes states that were never visited, and that the random agent's mean episode length is close to 32.

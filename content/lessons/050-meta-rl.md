---
title: "Meta rl"
slug: 050-meta-rl
part: "Part 7 — Models, exploration, and fixed data"
summary: "A draft chapter for 050-meta-rl. It holds a place in the course while the lesson is written."
minutes: 20
gpu: false
draft: true
objectives:
  - State the question this draft chapter will answer.
  - Name the function the exercise asks you to write.
  - Run the checks and read the message from the starter file.
prereqs:
  - 049-options-and-hierarchical-rl
exercise: 050-meta-rl
---

# Meta rl

This chapter is a draft. It keeps a number in the course so the lessons after
it can move, and the work you have already finished stays attached to those
lessons.

The exercise beside this page is a placeholder. `draft_answer` takes a step
count and returns that same count. The starter raises `NotImplementedError`,
so the checks fail until you fill the function in.

## What you already know

Three facts carry over from the previous chapter:

- The previous chapter is [Options and hierarchical RL](/lessons/049-options-and-hierarchical-rl).
- A lesson file and its exercise directory share one slug.
- The checks call `draft_answer` and compare the return value with the number you pass in.

## A placeholder for the idea

The following figure marks this chapter as unfinished. The chapter that replaces
this draft draws the idea in its place.

```figure
<svg viewBox="0 0 560 140" role="img" aria-label="A single bar marking this chapter as a draft">
  <line x1="40" y1="110" x2="520" y2="110" class="stroke-muted"/>
  <rect x="80" y="50" width="70" height="60" class="fill-a"/>
  <text x="170" y="88">draft chapter</text>
</svg>
Figure 1. This figure holds the place of the picture the chapter will draw.
```

> [!KEY]
> A draft chapter keeps its neighbors numbered while its own text is still being written.

## Build it step by step

Fill the function in this order:

1. Open `starter.py` and find `draft_answer`. It takes one integer, `steps`, and returns that same integer.
2. Replace the `NotImplementedError` with `return steps`.
3. Run the checks. Each one calls `draft_answer` with a different number and expects that number back.

## The exercise

You write `draft_answer`. The checks call it with several step counts and expect each call to return the count it was given. The starter fails those checks until you fill the function in.

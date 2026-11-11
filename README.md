# Reinforcement learning

A hands-on course that teaches reinforcement learning by having you build it.
Eighty-four short chapters, each teaching one idea, run from a two-armed bandit to the methods that post-train
language, image, speech, and robot models. Each one explains an idea in plain
words, works a small example by hand, and then walks you through implementing
it, in an editor beside the text, against checks that verify the mathematics
rather than the shape of your code.

The reading pane is on the left and the editor is on the right. **Run**
executes your file and streams its output. **Check** runs the exercise's checks
and reports them one at a time as they finish.

## Who it is for

You have taken a first machine learning course, such as the one at
[ml.ifkash.dev](https://ml.ifkash.dev): you can fit a logistic regression,
train a small network with gradient descent, and write numpy and a little
PyTorch. You don't need any reinforcement learning background. The lessons
define every term and symbol when it first appears, show each formula next to
the code that computes it, and end with a step-by-step guide to the exercise
and a list of the papers behind it.

## What is in here

The repository is laid out as follows:

| Path | Contents |
|---|---|
| `content/lessons/` | Eighty-four chapters, in Markdown with YAML front matter. |
| `content/exercises/` | One directory per exercise: a starter file, checks, and a worked solution. |
| `backend/` | FastAPI: auth, content, progress, drafts, and the run stream. |
| `backend/app/runner/` | The harness, the `rlcheck` assertion library the checks use, the `rlenvs` environments every exercise shares, and the execution backends. |
| `frontend/` | React and Vite: the reader, the editor, and the check console. |
| `frontend/src/book/` | The print layout for the PDF book, and `frontend/scripts/build-book.mjs`, which prints it. |
| `docs/` | The content contract, the curriculum, and the research notes the lessons were written from. |
| `scripts/` | Content validation and the book's content dump. |

## The curriculum

**Part 1 — Learning from reward (001–007).** The agent-environment loop,
returns and discounting, epsilon-greedy, UCB, Thompson sampling, gradient
bandits, and contextual bandits.

**Part 2 — Planning with a known world (008–012).** Markov decision processes,
the Bellman equation, policy evaluation, policy iteration, and value iteration.

**Part 3 — Learning from experience (013–022).** Monte Carlo prediction and
control, off-policy importance sampling, temporal-difference learning, SARSA and
Q-learning, maximization bias, n-step returns, eligibility traces, Dyna, and
Monte Carlo tree search.

**Part 4 — Deep value methods (023–030).** Function approximation, control with
better features and average reward, the deadly triad, deep Q-networks, double and dueling DQN, prioritized replay,
distributional reinforcement learning, and learning from pixels.

**Part 5 — Policy gradients (031–039).** REINFORCE, the policy gradient
theorem, baselines and actor-critic, generalized advantage estimation, TRPO,
PPO, continuous PPO in practice, IMPALA and V-trace, and why policy
optimization works.

**Part 6 — Continuous control and maximum entropy (040–042).** DDPG and TD3,
maximum-entropy reinforcement learning, and soft actor-critic.

**Part 7 — Models, exploration, and fixed data (043–056).** Model-based
reinforcement learning, world models and MuZero, partial observability and
memory, exploration bonuses, sample complexity and regret, goal-conditioned
reinforcement learning and hindsight relabeling, options and hierarchical
reinforcement learning, meta-reinforcement learning, imitation learning, inverse reinforcement learning and GAIL, offline reinforcement
learning, advantage-weighted policy improvement, off-policy evaluation from
logs, and the decision transformer.

**Part 8 — Rewards, evaluation, and many agents (057–064).** Reward design and
shaping, learning a reward from preferences over trajectories, evaluating agents
honestly, debugging and diagnosing runs, multi-agent learning
and self-play, AlphaZero-style training from self-play, counterfactual regret
minimization on Kuhn poker, and cooperative multi-agent learning with VDN and
QMIX.

**Part 9 — RL for language models (065–079).** Supervised fine-tuning and the
post-training pipeline, language models as policies, reward models, reward-model
and judge evaluation, RLHF with PPO, DPO, GRPO and its successors, verifiable
rewards, reasoning with verifiable rewards, process rewards and test-time
search, credit assignment for language models, AI feedback and self-improvement, on-policy
distillation, multi-turn agents, and stale
samples and importance weights.

**Part 10 — Beyond text, and what can go wrong (080–084).** Diffusion and flow
models as policies, vision, speech, and robot learning, sim-to-real
transfer, reward hacking, safe
reinforcement learning and the frontier, and a capstone that chains SFT and DPO
with a LoRA adapter and a written evaluation.

If you started the earlier 52-lesson course, your progress carries over;
`content/renames.json` maps the old slugs to the new chapters.

[The full list, with what each exercise delivers](docs/CURRICULUM.md).

## Run it locally

You need Python 3.12 or later and Node 20 or later.

1. Copy the environment file. Local development can use the default empty SSO
   secret only for public endpoints; protected APIs require the trusted Caddy
   proxy, so use the deployed portal to exercise authenticated pages.

   ```bash
   cp .env.example .env
   ```

2. Install and start the backend:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r backend/requirements.txt pyyaml
   .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
   .venv/bin/uvicorn backend.app.main:app --reload --port 8000
   ```

3. In another terminal, start the frontend:

   ```bash
   cd frontend && npm install && npm run dev
   ```

Open `http://localhost:5173`. The Vite dev server proxies `/api` to the backend
on port 8000. To use a backend on another port, set `API_TARGET`, for example
`API_TARGET=http://127.0.0.1:8011 npm run dev`. Protected pages redirect to the
central sign-in portal, which returns to the exact page you opened.

## Write a lesson

`docs/CONTENT_CONTRACT.md` is the binding specification: the file layout, the
front matter, the required sections, the prose style, and the `mlcheck` API the
checks are written against.

Validate what you write:

```bash
.venv/bin/python scripts/check_content.py 016-temporal-difference-learning
```

```bash
node scripts/check_math.mjs content/lessons/016-temporal-difference-learning.md
```

That checks the front matter, the required sections, and the word count; runs
every check against `solution.py`, which must pass all of them; and runs them
against `starter.py`, which must fail at least one. Omit the slug to validate
everything. The same script runs in CI on every push.

`check_math.mjs` compiles every math segment with the frontend's own KaTeX, so
math that fails only at render time fails here instead. Run `npm install` in
`frontend/` first.

## How a run works

Pressing **Run** or **Check** posts your code to `/api/run`, which streams
server-sent events back until the run finishes. The event types are `begin`,
`status`, `plan`, `stdout`, `stderr`, `check`, `error`, and `done`.

The submission executes in a subprocess in its own process group, started with
`python -I`, under a CPU-time limit and a wall clock, with the container's
`mem_limit` bounding memory. When the wall clock runs out the whole process
group is killed. The harness wraps the reader's own prints as events, so a print
that happens to look like JSON cannot be confused with a harness event.

This is a resource boundary, not a security boundary. Anyone who can sign in can
execute code in the container. The app is for two named people on a private
host; do not open it wider without putting the runner in a sandbox that isolates
the filesystem and the network.

## Authentication

Authelia at `https://auth.ifkash.dev` protects the site through Caddy
`forward_auth`. The frontend sends unauthenticated visitors to the portal with
the exact current URL encoded as its `rd` return destination. Caddy copies
Authelia's `Remote-User` header and injects `X-SSO-Proxy`; the backend trusts
the identity only when that header matches `SSO_PROXY_SECRET`. The backend
rejects protected API and book requests without this vetted identity. Map
portal names to existing progress identities with `SSO_USERS`, for example
`ifkash:enderboi,ananya:michu`; unmapped usernames use their portal name.
Sign out using the shared Authelia portal.

## The GPU backend

Four lessons—`069-rlhf-with-ppo`, `071-group-relative-policy-optimization`,
`073-reasoning-with-verifiable-rewards`, and `084-capstone-post-training`—are marked `gpu: true`. Their checks
run on toy policies like every other exercise, and the GPU is there for the
**Run** block, which post-trains `Qwen/Qwen3-0.6B` with the reader's own loss
functions.

`backend/app/runner/runpod.py` sends a submission to a RunPod serverless
endpoint and streams the worker's events back. The worker in `gpu/` carries the
same harness, `rlcheck`, `rlenvs`, and exercise directories, plus
`transformers`, so a lesson behaves identically on either backend. With
`RUNPOD_ENDPOINT_ID` unset, the GPU lessons fall back to the local CPU runner,
and their **Run** block falls back to the toy policy.

To set up the endpoint, complete the following steps:

1. The publish workflow builds and pushes the worker on every push that
   touches `gpu/`, `content/exercises/`, or the harness, and tags
   `ghcr.io/weights-and-wires/rl-gpu-worker:latest`.

2. **Give RunPod read access to that image.** A package published from Actions
   starts private, and RunPod pulls anonymously. Make the package public under
   the organization's **Packages** settings, the same way `ml-gpu-worker` is
   public.

3. Create the endpoint and read back its ID:

   ```bash
   RUNPOD_API_KEY=... python3 scripts/runpod_endpoint.py
   ```

4. Put `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` in the deployed `.env` and
   restart:

   ```bash
   ssh ifkash@vm.ifkash.dev "cd ~/docs/courses/rl && docker compose up -d"
   ```

`/api/compute` reports what the endpoint has: whether it is reachable, how many
workers are warm, and how many jobs are queued.

## Deploy

The VM runs the app from a container image, behind the shared Caddy on the
external `edge` network. This bundle never publishes 80 or 443 itself.

A deploy has three steps:

1. Merge to `main`.
2. GitHub Actions builds the image and pushes it to
   `ghcr.io/weights-and-wires/rl`.
3. On the VM, run `deploy rl`. It pulls the new image and restarts the
   container.

The VM never builds from source and holds no copy of the tree. `.env` on the VM
holds the `SSO_PROXY_SECRET` matching `RL_SSO_PROXY_SECRET` in Caddy and any
RunPod key, and a deploy never touches it. `Caddyfile.example` documents the
vhost's Authelia forward-auth, trusted identity header, and run-stream
buffering and timeout settings. `docker-compose.yml` is an example of the
service definition.

The image also prints the PDF book, in a build stage that holds Chromium and
the print layout's dependencies. The final image carries only the PDF and its
manifest. The manifest's `commit` is `unknown` in the image, because the book
stage doesn't receive `GIT_SHA`. For more information, see
[The PDF book](#the-pdf-book).

To reach the app before its DNS record resolves, tunnel the container's
localhost port:

```bash
ssh -L 8097:127.0.0.1:8097 ifkash@vm.ifkash.dev
```

## The PDF book

The **PDF** button in the header downloads the whole course as one book: every
lesson, every exercise with its starter file and hints, and the solutions in an
appendix. The book has a contents page with page numbers, a bookmark for every
lesson and section, and running heads. Like the lessons, the download needs a
signed-in session.

To build the book, run the following command:

```bash
node frontend/scripts/build-book.mjs
```

The build needs Google Chrome or Chromium, found through `CHROME_PATH` or in
the usual install locations. It dumps the lessons with the backend's own loader
(`scripts/dump_book.py`), renders the print layout in `frontend/src/book/` with
the site's Markdown pipeline, and prints it with headless Chrome. It prints
more than once, so that the contents carry real page numbers. The PDF and its
`manifest.json` land in `book/`, which Git ignores.

The build fails on any formula KaTeX can't render and on anything wider than
the page, because Chrome shrinks every page to fit the widest element. It skips
itself when no lesson or layout file has changed; to rebuild anyway, pass
`--force`. The image build prints the book, so a deploy needs no local build.
CI also builds the book on every push and keeps it as a workflow artifact for
seven days.

The lockfile must come from npm 10, because the image's `npm ci` rejects a
lockfile that npm 11 writes. After you change a dependency, run
`npx npm@10 install` in `frontend/`.

## License

MIT. See [LICENSE](LICENSE).

# Sentinel

An autonomous incident response agent. It watches a production environment,
investigates when something breaks, criticises its own conclusions, and proposes
a fix that a human approves before anything is executed.

It is also a working tour of modern agentic AI: LangGraph orchestration,
LangChain, MCP, tool calling, hybrid RAG, multi-agent critique, guardrails,
evals, and a fine-tuned model doing the cheap high-volume classification.

---

## What it does

An alert fires. Sentinel triages it, retrieves relevant runbooks and past
incidents, pulls logs, metrics and deploy history, forms a hypothesis, and then
attacks that hypothesis with a critic. If the reasoning is weak, it loops back
for more evidence. When it is satisfied, it writes a remediation plan — and
stops, because the remediation tools refuse to run without an approval the
agent has no way to grant itself.

```
Alert → Triage → Retrieve → Investigate → Synthesize → Critique
                                 ↑                        │
                                 └──── revise ────────────┤
                                                          ↓ accept
                                                        Plan
                                                          ↓
                                        ╔═══════════════════════════════╗
                                        ║  INTERRUPT: human approves    ║
                                        ╚═══════════════════════════════╝
                                                          ↓
                                        Execute → Verify → Post-mortem
```

The production environment is simulated and lives in [`simulator/`](simulator/).
That is deliberate: it makes the repo runnable by anyone who clones it, and it
makes evaluation possible. Each scenario ships with a ground-truth root cause
the agent never sees, so "did it actually find the cause" has an answer.

## Quick start

Requires [uv](https://docs.astral.sh/uv/), Docker, and
[Ollama](https://ollama.com).

```bash
git clone https://github.com/GautamRaju18/Sentinel.git && cd Sentinel
```

```bash
cp .env.example .env    # then add your OPENROUTER_API_KEY
```

```bash
uv sync --extra dev && ollama pull llama3.2 && ollama pull nomic-embed-text
```

```bash
docker compose up -d && uv run python -m sentinel.cli ingest
```

Then run a full incident response:

```bash
uv run python -m sentinel.cli respond bad_deploy --show-answer
```

Or open the operator console:

```bash
uv run streamlit run ui/app.py
```

### Other commands

| Command | What it does |
|---|---|
| `python -m sentinel.cli scenarios` | List incident scenarios |
| `python -m sentinel.cli investigate <slug>` | Single-agent investigation, no graph |
| `python -m sentinel.cli graph` | Print the graph as Mermaid |
| `python -m sentinel.cli routing` | Show which model each tier resolves to |
| `python -m sentinel.cli tracing` | LangSmith tracing status + recent traces |
| `python scripts/smoke_phase1.py` | Verify the whole stack is wired up |
| `python evals/redteam.py` | Adversarial suite |
| `python evals/triage_eval.py` | Score the triage classifier |
| `pytest tests/ -q` | 83 unit tests |
| `uvicorn sentinel.api.app:app` | HTTP API with SSE streaming |
| `docker compose --profile app up` | Everything in containers |

## Scenarios

| slug | what broke | why it is interesting |
|---|---|---|
| `bad_deploy` | ORM refactor removed eager loading → N+1 queries | classic change-induced failure; a second recent deploy is a decoy |
| `memory_leak` | unbounded session cache → OOMKill loop | the culprit deploy is 3 days old, so recency heuristics fail |
| `pool_exhaustion` | configmap cut DB pool 40 → 10 | the metric *drops* instead of rising; errors blame the database |
| `cert_expiry` | internal TLS cert expired | nothing was deployed; CPU falls; **no available tool can fix it** |
| `dependency_cascade` | Redis cannot fork for RDB persistence | three services alert and none is the cause; **no tool can fix it** |

Two scenarios are deliberately unfixable with the agent's toolset. Diagnosing
correctly and escalating is the right answer there — an agent that keeps acting
until something appears to work has learned the wrong lesson.

## Results

QLoRA fine-tune of Qwen2.5-1.5B-Instruct on 1000 synthetic triage examples,
~30 minutes on a free Colab T4. Baseline is llama3.2:3b prompted with the same
schema. Both served through Ollama on the same machine.

### In-distribution (120 held-out alerts, unseen service names)

| metric | baseline | fine-tuned |
|---|---|---|
| valid JSON | frequently failed | **100%** |
| severity accuracy | 37.5% | **97.5%** |
| category accuracy | 25.0% | **100%** |
| category macro-F1 | 0.211 | **1.000** |
| critical underestimates (P1 filed as P3/P4) | 10.0% | **0%** |
| latency p50 | 2980 ms | **1622 ms** |

Half the latency at nearly double the accuracy, on a model half the size. The
JSON figure is the load-bearing one: the baseline's 25% category accuracy was
largely *because* it could not reliably emit parseable output, and a classifier
whose response will not parse does not work regardless of what it knew.

### Out-of-distribution — and this is the interesting part

A 100% score on a held-out split is a warning, not a triumph. That split varies
service names and numbers but reuses the same 21 templates as training, so it
catches a model that memorised *entities* and cannot catch one that memorised
*templates*.

[`evals/generalization.py`](evals/generalization.py) asks the harder question,
using alerts that share no template with the training data: the five
hand-written scenario alerts, plus six adversarial cases where the surface cues
point at the wrong label and the real signal is a detail.

| | baseline | fine-tuned | delta |
|---|---|---|---|
| realistic unseen alerts — severity | 40% | 80% | **+40pp** |
| realistic unseen alerts — category | 40% | 60% | **+20pp** |
| adversarial — severity | 67% | 67% | **±0** |
| adversarial — category | 50% | 50% | **±0** |

**Large gains on realistic alerts it had never seen. Exactly zero on the cases
designed to defeat surface-feature matching.**

That split is the honest headline. The fine-tune learned to map surface
features to labels far better than the baseline — genuinely useful, and worth
the 30 minutes. It did not learn the causal reasoning the adversarial cases
require. It still calls a cascading dependency failure a bad deploy when a
deploy happens to be nearby, and still rates an alert P1 because it contains
the words CRITICAL and FATAL despite the body saying zero users were affected.

This is what template-generated training data should be expected to produce,
and the reason the generalization suite exists. Fixing it needs training data
with genuine causal variety, not more of the same templates.

### Red-team

31/31 checks passing — 8 injection attacks neutralised, 5 secret classes
redacted, and the approval gate holding under every attempt.

## Design notes

**Guardrails are structural, not advisory.** The investigator is handed a
read-only tool list, so it *cannot* call a destructive tool regardless of what
it concludes. Write tools check an approval flag held in module state that no
prompt can set. Confidence is not authorization.

**Tool output is untrusted input.** Anyone who can write a log line can write
text the agent will read — a customer-supplied username is enough. Every tool
result passes through injection neutralisation, secret redaction and truncation
before entering a prompt. Detection is best-effort and will eventually miss
something, which is exactly why the gate does not depend on it.

**The critic is adversarial on purpose.** Its prompt asks what is *wrong* with
the hypothesis, not whether it is reasonable. Acting on a wrong hypothesis
during an outage makes the outage longer.

**Plans are validated before execution.** The first end-to-end run produced a
plan that applied an empty config key and scaled a failing service from 4
replicas down to 1. Both were schema-valid. Semantic errors need a semantic
check — see [`graph/validation.py`](sentinel/graph/validation.py).

**Models are addressed by tier, not by name.** Nodes ask for `PLANNER` or
`WORKER`; `.env` decides what that resolves to. That seam is what makes the
local-vs-hosted comparison and the fine-tuning result measurable rather than
anecdotal.

**Hybrid retrieval, because neither half is enough.** Vector search finds
"memory exhaustion" from "exit code 137"; BM25 finds the rare exact tokens
(`x509`, `HikariPool-1`, `MISCONF`) that embeddings blur. Fused with reciprocal
rank fusion, which needs no score normalisation and so has nothing to tune.

## Two environment problems, and what they forced

Neither was worked around by weakening anything, and both improved the design.

**Windows Smart App Control blocks `jiter`**, a compiled dependency of the
`openai` package that `langchain-openai` requires. Rather than disable an OS
security control, [`models/openrouter.py`](sentinel/models/openrouter.py)
implements the chat-completions protocol directly over httpx. That also bought
a model fallback chain, which the free tier badly needs, and exact token
accounting.

**psycopg's async mode requires a `SelectorEventLoop`; MCP's stdio transport
requires a `Proactor` loop.** One process cannot have both. So there are no
async database drivers at all — everything Postgres runs synchronously and
`asyncio.to_thread` bridges it, which is portable off Windows too. See
[`sentinel/db.py`](sentinel/db.py).

## Layout

```
sentinel/       agent, graph, tools, models, memory, rag, mcp, api
simulator/      the simulated production environment and its failure scenarios
data/runbooks/  the RAG corpus
finetune/       dataset generation, QLoRA training, export to Ollama
evals/          triage metrics, baseline/fine-tuned comparison, red-team
tests/          83 unit tests
ui/             Streamlit operator console
docs/           concept-index.md maps every concept to its implementation
```

[`docs/concept-index.md`](docs/concept-index.md) is the map from each concept to
the file implementing it.

## Status

- [x] **Phase 1** — simulator, tool layer, guardrails, single ReAct agent
- [x] **Phase 2** — MCP server (tools, resources, prompts) and multi-server client
- [x] **Phase 3** — LangGraph: typed state, conditional edges, cycles, checkpointing
- [x] **Phase 4** — hybrid RAG, incident memory, semantic cache
- [x] **Phase 5** — critic loop, human-in-the-loop gate, plan validation
- [x] **Phase 6** — dataset, QLoRA trainer, Ollama export, trained and measured
- [x] **Phase 7** — evals, red-team, API, UI, Docker

## Security

`.env` is gitignored and no secret is committed. If you clone this and add your
own keys, keep them there — and rotate anything that has ever been pasted into
a chat window or a terminal transcript.

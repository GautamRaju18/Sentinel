# Sentinel

An autonomous incident response agent. It watches a production environment,
investigates when something breaks, finds the root cause, and proposes a fix
that a human approves before anything is executed.

It is also a working tour of modern agentic AI: LangGraph orchestration,
LangChain, MCP, tool calling, RAG, multi-agent patterns, guardrails, evals, and
a fine-tuned model doing the cheap high-volume classification.

> **Status:** Phase 1 of 7 complete. See [the roadmap](#roadmap).

---

## What it does

An alert fires. Sentinel triages it, pulls logs, metrics and deploy history,
forms a hypothesis, criticises its own hypothesis, and loops back for more
evidence if the reasoning is weak. When it is confident, it writes a
remediation plan — and stops, because the remediation tools refuse to run
without a human approval that the agent cannot grant itself.

```
Alert → Triage (fine-tuned classifier) → Route
   ├─ Investigator  : logs, metrics, deploys        (read-only by construction)
   ├─ Code agent    : RAG over repo + git history
   └─ Runbook agent : RAG over past incidents
        ↓
   Hypothesis → Critic → (weak? loop back and gather more)
        ↓
   Remediation plan → HUMAN APPROVAL GATE → Execute → Verify → Post-mortem
```

The production environment being watched is simulated and lives in
[`simulator/`](simulator/). That is deliberate: it makes the repo runnable by
anyone who clones it, and it makes the eval suite deterministic. Each scenario
ships with a ground-truth root cause that the agent never sees, so
"did it actually find the cause" is a question with an answer.

## Quick start

Requires [uv](https://docs.astral.sh/uv/), Docker, and
[Ollama](https://ollama.com).

```bash
git clone https://github.com/GautamRaju18/Sentinel.git && cd Sentinel
cp .env.example .env          # then fill in OPENROUTER_API_KEY
uv sync --extra dev
ollama pull llama3.2 && ollama pull nomic-embed-text
docker compose up -d
uv run python scripts/smoke_phase1.py
```

Then investigate an incident:

```bash
uv run python -m sentinel.cli investigate bad_deploy --tier planner --show-answer
```

Other commands:

```bash
uv run python -m sentinel.cli scenarios
```

```bash
uv run python -m sentinel.cli routing
```

## Scenarios

| slug | what broke | why it is interesting |
|---|---|---|
| `bad_deploy` | ORM refactor removed eager loading → N+1 queries | classic change-induced failure; a second recent deploy is a decoy |
| `memory_leak` | unbounded session cache → OOMKill loop | the culprit deploy is 3 days old, so recency heuristics fail |
| `pool_exhaustion` | configmap cut DB pool 40 → 10 | the metric *drops* instead of rising; errors blame the database |
| `cert_expiry` | internal TLS cert expired | nothing was deployed; CPU falls; a downstream service also alerts |
| `dependency_cascade` | Redis cannot fork for RDB persistence | three services alert at once and none of them is the cause |

## Design notes

**Guardrails are structural, not advisory.** The investigator is given a
read-only tool list, so it *cannot* call a destructive tool regardless of what
it decides. Write tools check an approval flag held in module state that the
model has no way to set. Confidence is not authorization.

**Tool output is untrusted.** Logs are attacker-influenced: anyone who can
write a log line can try to write an instruction. Every tool result passes
through neutralisation, PII redaction and truncation before it enters a prompt,
and injection attempts are flagged rather than silently swallowed. See
[`sentinel/tools/guardrails.py`](sentinel/tools/guardrails.py).

**Models are addressed by tier, not by name.** Nodes ask for `PLANNER` or
`WORKER`; `.env` decides what that resolves to. That is what makes the local
vs hosted comparison — and the Phase 6 fine-tuning result — measurable rather
than anecdotal.

**No `langchain-openai`.** It depends on `openai` → `jiter`, whose compiled
extension is blocked by Windows Smart App Control on the development machine.
Rather than weaken an OS security control,
[`sentinel/models/openrouter.py`](sentinel/models/openrouter.py) speaks the
chat-completions protocol directly over httpx. It also adds a model fallback
chain, which the free tier badly needs.

## Roadmap

- [x] **Phase 1** — simulator, tool layer, guardrails, single ReAct agent
- [ ] **Phase 2** — MCP server exposing the tools; external MCP clients
- [ ] **Phase 3** — LangGraph: state, conditional edges, cycles, checkpointing
- [ ] **Phase 4** — RAG over runbooks and past incidents; semantic cache
- [ ] **Phase 5** — multi-agent supervisor, critic loop, human-in-the-loop gate
- [ ] **Phase 6** — QLoRA fine-tune of the triage classifier
- [ ] **Phase 7** — evals, red-team suite, observability, Streamlit UI

[`docs/concept-index.md`](docs/concept-index.md) maps each concept to the file
that implements it.

## Layout

```
sentinel/       agent, graph, tools, models, memory, api
simulator/      the fake production environment and its failure scenarios
finetune/       dataset generation, QLoRA training, export to Ollama
evals/          datasets, LLM-as-judge, trajectory scoring, red-team
ui/             Streamlit front end
```

## Security

`.env` is gitignored and no secret is committed. If you clone this and add your
own keys, keep them there.

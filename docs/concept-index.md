# Concept index

Every concept this project set out to cover, mapped to the file that
implements it. If a row has no file, it is not done.

## Agent orchestration — LangGraph

| Concept | Where |
|---|---|
| StateGraph, typed state | [`sentinel/graph/state.py`](../sentinel/graph/state.py) |
| State reducers (`add_messages`, `operator.add`, custom merge) | [`state.py:merge_usage`](../sentinel/graph/state.py) |
| Nodes as pure functions | [`sentinel/graph/nodes.py`](../sentinel/graph/nodes.py) |
| Conditional edges / routing | [`nodes.py:route_after_critique`](../sentinel/graph/nodes.py) |
| **Cycles** (investigate → critique → investigate) | [`builder.py`](../sentinel/graph/builder.py) |
| Loop bounding | `MAX_INVESTIGATION_LOOPS` in [`config.py`](../sentinel/config.py) |
| Interrupts / human-in-the-loop | `INTERRUPT_BEFORE` in [`builder.py`](../sentinel/graph/builder.py) |
| Checkpointing (Postgres) | [`sentinel/db.py`](../sentinel/db.py) |
| Resume after interrupt | [`runner.py:resume_with_decision`](../sentinel/runner.py) |
| Time travel / checkpoint forking | [`runner.py:fork_from`](../sentinel/runner.py) |
| Streaming node updates | [`runner.py:run_until_pause`](../sentinel/runner.py) |
| Graph visualisation | [`builder.py:render_mermaid`](../sentinel/graph/builder.py) |

## LangChain

| Concept | Where |
|---|---|
| Custom `BaseChatModel` | [`models/openrouter.py`](../sentinel/models/openrouter.py) |
| Tool definitions with Pydantic args schemas | [`tools/observability.py`](../sentinel/tools/observability.py) |
| Structured output | [`models/structured.py`](../sentinel/models/structured.py) |
| Output repair and retry on validation errors | [`structured.py:generate_structured`](../sentinel/models/structured.py) |
| Message types and conversion | [`openrouter.py:_lc_to_wire`](../sentinel/models/openrouter.py) |
| Embeddings | [`models/router.py:get_embeddings`](../sentinel/models/router.py) |
| Document loaders and splitters | [`rag/store.py:chunk_markdown`](../sentinel/rag/store.py) |
| Vector store (pgvector) | [`rag/store.py`](../sentinel/rag/store.py) |
| Streaming (SSE) | [`openrouter.py:_astream`](../sentinel/models/openrouter.py) |

## Tool calling

| Concept | Where |
|---|---|
| The agentic loop, written out | [`sentinel/agents/__init__.py`](../sentinel/agents/__init__.py) |
| Parallel tool execution | `asyncio.gather` in [`agents/__init__.py`](../sentinel/agents/__init__.py) |
| Tool errors returned as observations | [`agents/__init__.py:_run_tool`](../sentinel/agents/__init__.py) |
| Per-tool timeouts | `tool_timeout_seconds` in [`config.py`](../sentinel/config.py) |
| Tool output truncation | [`guardrails.py:truncate`](../sentinel/tools/guardrails.py) |
| Tool descriptions as prompt surface | docstrings in [`observability.py`](../sentinel/tools/observability.py) |
| Capability isolation (read-only toolset) | [`tools/__init__.py:get_tools`](../sentinel/tools/__init__.py) |
| Trajectory capture for eval | `AgentRun.tool_sequence` |

## MCP

| Concept | Where |
|---|---|
| MCP server — tools | [`mcp_server/server.py`](../sentinel/mcp_server/server.py) |
| MCP server — resources (`incident://`) | same file |
| MCP server — prompts | same file |
| stdio and streamable-http transports | `server.py:main` |
| MCP client, multi-server | [`sentinel/mcp_client.py`](../sentinel/mcp_client.py) |
| External server config with env expansion | [`mcp.config.json`](../mcp.config.json) |
| Graceful degradation when a server is absent | `mcp_client.py:_is_available` |
| Tool namespacing across servers | `mcp_client.py:load_mcp_tools` |

## Retrieval

| Concept | Where |
|---|---|
| Heading-aware chunking | [`rag/store.py:chunk_markdown`](../sentinel/rag/store.py) |
| Contextual retrieval (breadcrumb in chunk) | same function |
| Vector search | [`rag/retriever.py:vector_search`](../sentinel/rag/retriever.py) |
| BM25 lexical search | `retriever.py:bm25_search` |
| **Hybrid search with reciprocal rank fusion** | `retriever.py:reciprocal_rank_fusion` |
| Query expansion from triage output | `retriever.py:expand_query` |
| Corpus | [`data/runbooks/`](../data/runbooks/) |

## Memory

| Concept | Where |
|---|---|
| Long-term incident memory | [`memory/incidents.py`](../sentinel/memory/incidents.py) |
| Distilling a run into a retrievable document | `incidents.py:_to_document` |
| Semantic cache | [`memory/cache.py`](../sentinel/memory/cache.py) |
| TTL and LRU eviction | `cache.py:_evict` |
| Short-term memory (message state) | `messages` channel in [`state.py`](../sentinel/graph/state.py) |

## Multi-agent

| Concept | Where |
|---|---|
| Specialist agents with distinct prompts and toolsets | [`agents/prompts.py`](../sentinel/agents/prompts.py) |
| **Critic / reflection loop** | `nodes.py:critique_node` |
| Critic-directed re-investigation | `open_questions` in state, consumed by `investigate_node` |
| Deterministic execution of an approved plan | `nodes.py:execute_node` |

## Safety

| Concept | Where |
|---|---|
| Blast-radius classification | [`tools/guardrails.py`](../sentinel/tools/guardrails.py) |
| Approval gate the model cannot bypass | `remediation.py:_gate` + `_APPROVAL_GRANTED` |
| Prompt-injection detection and neutralisation | `guardrails.py:neutralize` |
| PII and secret redaction | `guardrails.py:redact` |
| Plan validation before execution | [`graph/validation.py`](../sentinel/graph/validation.py) |
| Single-use approval | `revoke_all_approvals()` in `execute_node` |
| Red-team suite | [`evals/redteam.py`](../evals/redteam.py) |

## Fine-tuning

| Concept | Where |
|---|---|
| Dataset generation | [`finetune/generate_dataset.py`](../finetune/generate_dataset.py) |
| Held-out split with unseen entities | `TEST_SERVICES` in the same file |
| Chat-template formatting | `to_chat_format` |
| QLoRA (4-bit NF4, double quantisation) | [`finetune/train_qlora.py`](../finetune/train_qlora.py) |
| LoRA rank/alpha/target modules | `peft_config` in the same file |
| Gradient checkpointing, paged optimiser | `SFTConfig` |
| Completion-only loss | `DataCollatorForCompletionOnlyLM` |
| Adapter merge and GGUF export | [`finetune/export_ollama.py`](../finetune/export_ollama.py) |
| Colab training notebook | [`finetune/colab_train.ipynb`](../finetune/colab_train.ipynb) |
| One-command install of the trained model | [`finetune/install_gguf.py`](../finetune/install_gguf.py) |
| Backend swap behind one interface | [`sentinel/triage.py`](../sentinel/triage.py) |

## Evaluation

| Concept | Where |
|---|---|
| Ground-truth datasets | `GroundTruth` in [`simulator/scenarios.py`](../simulator/scenarios.py) |
| Classification metrics, macro-F1 | [`evals/triage_eval.py`](../evals/triage_eval.py) |
| Domain-weighted error metric | `critical_underestimates` |
| Confusion matrices | same file |
| Baseline vs fine-tuned comparison | [`evals/compare.py`](../evals/compare.py) |
| **Out-of-distribution / generalization eval** | [`evals/generalization.py`](../evals/generalization.py) |
| Adversarial cases that defeat surface matching | `ADVERSARIAL` in the same file |
| Adversarial security evaluation | [`evals/redteam.py`](../evals/redteam.py) |
| Unit tests | [`tests/`](../tests/) |
| Running CI locally | [`scripts/check.py`](../scripts/check.py) |

## Production concerns

| Concept | Where |
|---|---|
| Model routing by tier | [`models/router.py`](../sentinel/models/router.py) |
| Provider fallback chains | `ChatOpenRouter.fallback_models` |
| Token accounting | `usage_metadata` → `token_usage` in state |
| Structured logging | [`logging_setup.py`](../sentinel/logging_setup.py) |
| LangSmith tracing | [`sentinel/observability.py`](../sentinel/observability.py) |
| Verifying tracing is live, not just configured | `observability.py:verify_tracing` |
| Settings from environment | [`config.py`](../sentinel/config.py) |
| HTTP API with SSE streaming | [`api/app.py`](../sentinel/api/app.py) |
| Operator UI | [`ui/app.py`](../ui/app.py) |
| Containerisation, non-root | [`Dockerfile`](../Dockerfile) |
| Graceful degradation | checkpointer fallback in `db.py`, retrieval in `retrieve_node` |

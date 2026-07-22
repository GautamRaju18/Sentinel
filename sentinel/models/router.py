"""Model routing.

Nodes ask for a *tier*, not a model. That indirection is what lets the whole
system swap between local Ollama and hosted OpenRouter by editing .env, and
it is what makes the cost story legible: cheap models do classification,
expensive ones do planning.

Spec format is "<provider>:<model>":
    ollama:llama3.2:latest
    openrouter:deepseek/deepseek-chat-v3-0324:free
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from sentinel.config import get_settings
from sentinel.logging_setup import get_logger

log = get_logger(__name__)


class ModelTier(StrEnum):
    # Hardest reasoning: remediation plans, root-cause synthesis.
    PLANNER = "planner"
    # Mid: evidence synthesis, critique, summarisation.
    REASONER = "reasoner"
    # Cheap and local: tool-heavy loops, extraction, formatting.
    WORKER = "worker"
    # Classification only — the tier Phase 6 replaces with a fine-tuned model.
    TRIAGE = "triage"


def _spec_for(tier: ModelTier) -> str:
    s = get_settings()
    return {
        ModelTier.PLANNER: s.model_planner,
        ModelTier.REASONER: s.model_reasoner,
        ModelTier.WORKER: s.model_worker,
        ModelTier.TRIAGE: s.model_triage,
    }[tier]


def parse_spec(spec: str) -> tuple[str, str]:
    """'ollama:llama3.2:latest' -> ('ollama', 'llama3.2:latest')"""
    if ":" not in spec:
        raise ValueError(f"model spec must be '<provider>:<model>', got {spec!r}")
    provider, model = spec.split(":", 1)
    provider = provider.strip().lower()
    if provider not in {"ollama", "openrouter"}:
        raise ValueError(f"unknown provider {provider!r} in spec {spec!r}")
    return provider, model.strip()


@lru_cache(maxsize=16)
def _build(spec: str, temperature: float, num_ctx: int) -> BaseChatModel:
    provider, model = parse_spec(spec)
    settings = get_settings()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_ctx=num_ctx,
            # Ollama defaults to a short context; incident evidence is long.
            reasoning=False,
        )

    from sentinel.models.openrouter import ChatOpenRouter

    if not settings.openrouter_api_key:
        raise RuntimeError(
            f"spec {spec!r} needs OPENROUTER_API_KEY. Set it in .env, or point this "
            f"tier at an ollama: model."
        )
    # Free-tier capacity comes and goes minute to minute, so every hosted call
    # gets a fallback chain rather than a single model.
    fallbacks = [
        m.strip()
        for m in settings.openrouter_fallbacks.split(",")
        if m.strip() and m.strip() != model
    ]
    return ChatOpenRouter(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=temperature,
        fallback_models=fallbacks,
    )


def get_model(
    tier: ModelTier = ModelTier.WORKER,
    *,
    temperature: float = 0.0,
    num_ctx: int = 8192,
) -> BaseChatModel:
    spec = _spec_for(tier)
    log.debug("model.resolve", tier=str(tier), spec=spec)
    return _build(spec, temperature, num_ctx)


@lru_cache(maxsize=2)
def get_embeddings() -> Embeddings:
    provider, model = parse_spec(get_settings().model_embed)
    if provider != "ollama":
        raise ValueError("embeddings must use an ollama: spec — OpenRouter has no embedding API")
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=model, base_url=get_settings().ollama_base_url)


def describe_routing() -> str:
    """Human-readable view of which model each tier resolves to."""
    lines = ["Model routing:"]
    for tier in ModelTier:
        lines.append(f"  {tier.value:<9} -> {_spec_for(tier)}")
    lines.append(f"  {'embed':<9} -> {get_settings().model_embed}")
    return "\n".join(lines)

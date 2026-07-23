"""Central settings. Everything tunable lives here and comes from .env.

Nothing else in the codebase reads os.environ directly — that keeps the
knobs discoverable and makes tests able to override cleanly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BlastRadius = Literal["low", "medium", "high", "critical"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- providers ---
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Comma-separated. Tried in order when the primary model is saturated —
    # free-tier availability fluctuates by the minute.
    openrouter_fallbacks: str = (
        "nvidia/nemotron-3-super-120b-a12b:free,"
        "openai/gpt-oss-20b:free,"
        "nvidia/nemotron-nano-9b-v2:free"
    )
    ollama_base_url: str = "http://localhost:11434"

    # --- model routing ---
    # Format: "<provider>:<model>" where provider is ollama | openrouter.
    model_planner: str = "openrouter:deepseek/deepseek-chat-v3-0324:free"
    model_reasoner: str = "openrouter:deepseek/deepseek-chat-v3-0324:free"
    model_worker: str = "ollama:llama3.2:latest"
    model_triage: str = "ollama:llama3.2:latest"
    model_embed: str = "ollama:nomic-embed-text:latest"
    triage_backend: Literal["baseline", "finetuned"] = "baseline"

    # --- storage ---
    database_url: str = "postgresql://sentinel:sentinel@localhost:5433/sentinel"

    # --- observability ---
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "sentinel"

    # --- integrations ---
    github_token: str = ""
    huggingface_token: str = ""
    # Repository monitoring. Owner whose repos we watch, and an optional
    # comma-separated allowlist (empty = every repo the token can see).
    github_owner: str = "GautamRaju18"
    github_watch: str = ""

    # --- safety rails ---
    # The investigate->reflect cycle is a loop in the graph; without a hard cap
    # a confused agent will burn tokens forever. This is the backstop.
    max_investigation_loops: int = 4
    # Remediations above this blast radius never auto-execute, whatever the
    # agent's confidence. Confidence is not authorization.
    max_auto_blast_radius: BlastRadius = "low"
    always_require_approval: bool = True

    # --- tool behaviour ---
    tool_timeout_seconds: float = 20.0
    max_tool_output_chars: int = 6_000

    @property
    def async_database_url(self) -> str:
        return self.database_url.replace("postgresql://", "postgresql+psycopg://", 1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

"""Shared UI helpers and styling.

Every page imports from here so the console looks like one product rather than
five scripts that happen to live in a folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEVERITY_COLOUR = {"P1": "#ef4444", "P2": "#f97316", "P3": "#eab308", "P4": "#64748b"}
BLAST_COLOUR = {
    "low": "#22c55e",
    "medium": "#eab308",
    "high": "#f97316",
    "critical": "#ef4444",
}
STAGE_ICON = {
    "triage": "🏷",
    "retrieve": "📚",
    "investigate": "🔍",
    "synthesize": "🧩",
    "critique": "⚖️",
    "plan": "📋",
    "approval": "🔐",
    "execute": "⚡",
    "verify": "✅",
    "postmortem": "📝",
}

CSS = """
<style>
  /* Tighten Streamlit's very generous default spacing */
  .block-container { padding-top: 2.2rem; max-width: 1200px; }

  .sentinel-hero {
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 1.4rem 1.6rem;
      margin-bottom: 1.2rem;
  }
  .sentinel-hero h1 { color: #f1f5f9; margin: 0; font-size: 1.6rem; }
  .sentinel-hero p  { color: #94a3b8; margin: .35rem 0 0; font-size: .92rem; }

  .pill {
      display: inline-block; padding: .18rem .6rem; border-radius: 999px;
      font-size: .75rem; font-weight: 600; letter-spacing: .02em;
      border: 1px solid currentColor;
  }
  .stat-card {
      border: 1px solid rgba(128,128,128,.25); border-radius: 10px;
      padding: .9rem 1rem; height: 100%;
  }
  .stat-card .label { font-size: .75rem; opacity: .7; text-transform: uppercase;
                      letter-spacing: .05em; }
  .stat-card .value { font-size: 1.55rem; font-weight: 700; line-height: 1.2; }
  .stat-card .delta { font-size: .8rem; }

  /* The approval gate should look like a stop sign, not a form */
  .gate {
      border: 2px solid #f59e0b; border-radius: 12px;
      padding: 1.1rem 1.3rem; background: rgba(245,158,11,.07);
  }
  code { font-size: .85em; }
</style>
"""


def page_setup(title: str, icon: str = "🛡️") -> None:
    st.set_page_config(page_title=f"Sentinel · {title}", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="sentinel-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def pill(text: str, colour: str) -> str:
    return f'<span class="pill" style="color:{colour}">{text}</span>'


def stat(label: str, value: str, delta: str = "", colour: str = "") -> str:
    delta_html = f'<div class="delta" style="color:{colour}">{delta}</div>' if delta else ""
    return (
        f'<div class="stat-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{delta_html}</div>'
    )


def load_results(name: str) -> dict | None:
    """Read a recorded eval result, or None if that eval has not been run."""
    path = ROOT / "evals" / f"results_{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def sidebar_status() -> None:
    """Compact system status, shown on every page."""
    from sentinel.config import get_settings

    settings = get_settings()
    with st.sidebar:
        st.markdown("### 🛡️ Sentinel")
        st.caption("Autonomous incident response")
        st.divider()

        backend = settings.triage_backend
        st.markdown(
            f"**Triage** {pill(backend, '#22c55e' if backend == 'finetuned' else '#64748b')}",
            unsafe_allow_html=True,
        )
        tracing = settings.langsmith_tracing
        st.markdown(
            f"**Tracing** {pill('on' if tracing else 'off', '#22c55e' if tracing else '#64748b')}",
            unsafe_allow_html=True,
        )
        st.caption(f"Approval required: {settings.always_require_approval}")
        st.caption(f"Max loops: {settings.max_investigation_loops}")


def no_data(message: str, command: str) -> None:
    """Consistent empty state that tells the user how to populate it."""
    st.info(message)
    st.code(command, language="bash")

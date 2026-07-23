"""A read-only GitHub client for repository monitoring.

Every method here is a GET. This client never writes — no pushing, no closing
issues, no cancelling runs. That is a deliberate constraint: monitoring should
not be able to mutate the thing it watches, and it means the token's write
scope is irrelevant to anything in this file. The one thing that acts on the
world (opening an incident from a failed run) lives behind the same human gate
as everything else in Sentinel.

The GitHub REST API is well specified and the surface we need is small, so we
speak it directly over httpx rather than pulling in a heavy SDK — the same
choice made for the model client.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from sentinel.config import get_settings
from sentinel.logging_setup import get_logger

log = get_logger(__name__)

API = "https://api.github.com"


@dataclass
class WorkflowRun:
    id: int
    name: str
    status: str  # queued | in_progress | completed
    conclusion: str | None  # success | failure | cancelled | ...
    branch: str
    event: str
    created_at: str
    url: str
    commit_message: str = ""
    actor: str = ""

    @property
    def failed(self) -> bool:
        return self.conclusion in {"failure", "timed_out", "startup_failure"}

    @property
    def ok(self) -> bool:
        return self.conclusion == "success"


@dataclass
class RepoHealth:
    name: str
    full_name: str
    private: bool
    default_branch: str
    pushed_at: str
    stars: int
    open_issues: int  # GitHub counts PRs as issues; we split them below
    open_prs: int
    description: str = ""
    latest_run: WorkflowRun | None = None
    has_ci: bool = False
    error: str = ""

    @property
    def ci_status(self) -> str:
        if self.error:
            return "error"
        if not self.has_ci:
            return "no-ci"
        if self.latest_run is None:
            return "no-runs"
        if self.latest_run.status != "completed":
            return "running"
        return "passing" if self.latest_run.ok else "failing"

    @property
    def days_since_push(self) -> int:
        try:
            when = datetime.fromisoformat(self.pushed_at.replace("Z", "+00:00"))
            return (datetime.now(UTC) - when).days
        except (ValueError, AttributeError):
            return -1

    @property
    def needs_attention(self) -> bool:
        """The one boolean the alert path keys on."""
        return self.ci_status in {"failing", "error"} or self.open_prs > 0


class GitHubMonitor:
    def __init__(self, token: str | None = None, owner: str | None = None):
        settings = get_settings()
        self.token = token or settings.github_token
        self.owner = owner or settings.github_owner
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    # --- low level --------------------------------------------------------

    def _get(self, path: str, **params: Any) -> httpx.Response:
        with httpx.Client(timeout=30) as client:
            return client.get(f"{API}{path}", headers=self._headers, params=params or None)

    def rate_limit(self) -> dict[str, Any]:
        r = self._get("/rate_limit")
        core = r.json().get("resources", {}).get("core", {}) if r.status_code == 200 else {}
        return {
            "remaining": core.get("remaining"),
            "limit": core.get("limit"),
            "reset_in_s": max(0, (core.get("reset", 0) - int(time.time()))),
        }

    # --- discovery --------------------------------------------------------

    def list_repos(self, limit: int = 100) -> list[dict[str, Any]]:
        """Repos the token can see, most-recently-pushed first.

        Uses /user/repos so private repos the token was granted also appear;
        falls back to the public listing if that is unavailable.
        """
        r = self._get("/user/repos", per_page=limit, sort="pushed", affiliation="owner")
        if r.status_code != 200:
            r = self._get(f"/users/{self.owner}/repos", per_page=limit, sort="pushed")
        r.raise_for_status()
        return r.json()

    def watched_repo_names(self) -> list[str]:
        """Resolve the configured watchlist, or every visible repo."""
        watch = get_settings().github_watch.strip()
        if watch:
            return [w.strip() for w in watch.split(",") if w.strip()]
        return [repo["name"] for repo in self.list_repos()]

    # --- per repo ---------------------------------------------------------

    def latest_run(self, repo: str, branch: str | None = None) -> WorkflowRun | None:
        params = {"per_page": 1}
        if branch:
            params["branch"] = branch
        r = self._get(f"/repos/{self.owner}/{repo}/actions/runs", **params)
        if r.status_code != 200:
            return None
        runs = r.json().get("workflow_runs", [])
        if not runs:
            return None
        w = runs[0]
        head = w.get("head_commit") or {}
        return WorkflowRun(
            id=w["id"],
            name=w.get("name") or w.get("display_title", "workflow"),
            status=w["status"],
            conclusion=w.get("conclusion"),
            branch=w.get("head_branch", ""),
            event=w.get("event", ""),
            created_at=w.get("created_at", ""),
            url=w.get("html_url", ""),
            commit_message=(head.get("message") or "").split("\n")[0],
            actor=(w.get("actor") or {}).get("login", ""),
        )

    def open_pr_count(self, repo: str) -> int:
        r = self._get(f"/repos/{self.owner}/{repo}/pulls", state="open", per_page=100)
        return len(r.json()) if r.status_code == 200 else 0

    def health(self, repo: str) -> RepoHealth:
        try:
            meta_r = self._get(f"/repos/{self.owner}/{repo}")
            meta_r.raise_for_status()
            meta = meta_r.json()

            run = self.latest_run(repo, branch=meta.get("default_branch"))
            prs = self.open_pr_count(repo)
            # GitHub's open_issues_count includes PRs; subtract them.
            issues = max(0, meta.get("open_issues_count", 0) - prs)

            return RepoHealth(
                name=meta["name"],
                full_name=meta["full_name"],
                private=meta["private"],
                default_branch=meta.get("default_branch", "main"),
                pushed_at=meta.get("pushed_at", ""),
                stars=meta.get("stargazers_count", 0),
                open_issues=issues,
                open_prs=prs,
                description=meta.get("description") or "",
                latest_run=run,
                has_ci=run is not None,
                error="",
            )
        except httpx.HTTPStatusError as e:
            return RepoHealth(
                name=repo,
                full_name=f"{self.owner}/{repo}",
                private=False,
                default_branch="",
                pushed_at="",
                stars=0,
                open_issues=0,
                open_prs=0,
                error=f"{e.response.status_code}",
            )
        except Exception as e:
            return RepoHealth(
                name=repo,
                full_name=f"{self.owner}/{repo}",
                private=False,
                default_branch="",
                pushed_at="",
                stars=0,
                open_issues=0,
                open_prs=0,
                error=f"{type(e).__name__}",
            )

    def health_all(self, repos: list[str] | None = None) -> list[RepoHealth]:
        names = repos or self.watched_repo_names()
        return [self.health(name) for name in names]

    # --- CI failure detail (feeds the investigator) -----------------------

    def failed_jobs(self, repo: str, run_id: int) -> list[dict[str, Any]]:
        r = self._get(f"/repos/{self.owner}/{repo}/actions/runs/{run_id}/jobs", per_page=100)
        if r.status_code != 200:
            return []
        return [
            {
                "name": j["name"],
                "conclusion": j.get("conclusion"),
                "failed_steps": [
                    s["name"] for s in j.get("steps", []) if s.get("conclusion") == "failure"
                ],
            }
            for j in r.json().get("jobs", [])
            if j.get("conclusion") == "failure"
        ]

    def run_log_excerpt(self, repo: str, run_id: int, max_chars: int = 4000) -> str:
        """The tail of a failed run's logs — where the actual error usually is.

        The logs endpoint returns a zip; we pull it, concatenate, and keep the
        end. Best-effort: on any failure we return a marker rather than raising,
        because a missing log should degrade the investigation, not kill it.
        """
        import io
        import zipfile

        try:
            with httpx.Client(timeout=45, follow_redirects=True) as client:
                r = client.get(
                    f"{API}/repos/{self.owner}/{repo}/actions/runs/{run_id}/logs",
                    headers=self._headers,
                )
            if r.status_code != 200:
                return f"[logs unavailable: HTTP {r.status_code}]"
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            text = "\n".join(
                zf.read(name).decode("utf-8", "replace")
                for name in zf.namelist()
                if name.endswith(".txt")
            )
            return text[-max_chars:] if len(text) > max_chars else text
        except Exception as e:
            return f"[could not read logs: {type(e).__name__}]"

    def recent_commits(self, repo: str, branch: str, n: int = 5) -> list[dict[str, str]]:
        r = self._get(f"/repos/{self.owner}/{repo}/commits", sha=branch, per_page=n)
        if r.status_code != 200:
            return []
        return [
            {
                "sha": c["sha"][:8],
                "message": (c["commit"]["message"] or "").split("\n")[0],
                "author": (c["commit"]["author"] or {}).get("name", ""),
                "date": (c["commit"]["author"] or {}).get("date", "")[:19],
            }
            for c in r.json()
        ]

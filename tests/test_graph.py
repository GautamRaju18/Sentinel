"""Graph topology, state reducers, structured output and plan validation.

These are the parts that break quietly. A missing reducer does not raise; it
silently drops half the evidence.
"""

from __future__ import annotations

import operator
from typing import get_args, get_type_hints

import pytest

from sentinel.graph.builder import INTERRUPT_BEFORE, build_graph, render_mermaid
from sentinel.graph.schemas import (
    Category,
    Confidence,
    Hypothesis,
    RemediationPlan,
    RemediationStep,
)
from sentinel.graph.state import IncidentState, merge_usage, new_state
from sentinel.graph.validation import validate_plan
from sentinel.models.structured import extract_json
from simulator.world import load_world


class TestTopology:
    def test_compiles(self):
        assert build_graph().compile()

    def test_has_the_reflection_cycle(self):
        mermaid = render_mermaid()
        assert "critique" in mermaid and "investigate" in mermaid
        # The cycle: critique must be able to send us back to investigate.
        assert "critique -.-> investigate" in mermaid

    def test_interrupts_before_approval(self):
        assert INTERRUPT_BEFORE == ["approval"]

    def test_approval_branches_both_ways(self):
        mermaid = render_mermaid()
        assert "approval -.-> execute" in mermaid
        assert "approval -.-> postmortem" in mermaid


class TestState:
    def test_evidence_accumulates(self):
        # get_type_hints, not __annotations__: `from __future__ import
        # annotations` leaves the raw strings behind.
        hints = get_type_hints(IncidentState, include_extras=True)
        assert operator.add in get_args(hints["evidence"])[1:]

    def test_usage_merges_by_addition(self):
        assert merge_usage({"input": 10}, {"input": 5, "output": 2}) == {
            "input": 15,
            "output": 2,
        }

    def test_new_state_is_complete(self):
        s = new_state("inc-1", "bad_deploy", "alert text")
        assert s["stage"] == "triage"
        assert s["loop_count"] == 0
        assert s["approved"] is None, "approval must start unset, never True"


class TestStructuredExtraction:
    @pytest.mark.parametrize(
        "raw",
        [
            '{"severity": "P1"}',
            'Here you go:\n```json\n{"severity": "P1"}\n```',
            'Some prose. {"severity": "P1"} And more prose.',
            '```\n{"severity": "P1"}\n```',
        ],
    )
    def test_finds_the_object(self, raw):
        assert extract_json(raw) is not None
        assert "P1" in extract_json(raw)

    def test_handles_nested_braces(self):
        raw = 'text {"a": {"b": {"c": 1}}, "d": 2} tail'
        assert extract_json(raw) == '{"a": {"b": {"c": 1}}, "d": 2}'

    def test_ignores_braces_inside_strings(self):
        raw = '{"msg": "this } is not the end", "ok": true}'
        assert extract_json(raw) == raw

    def test_returns_none_without_json(self):
        assert extract_json("no object here") is None


def _plan(**step_kwargs) -> RemediationPlan:
    defaults = dict(action="restart_service", target="checkout-api", rationale="because")
    defaults.update(step_kwargs)
    return RemediationPlan(
        summary="s",
        steps=[RemediationStep(**defaults)],
        expected_effect="e",
        rollback_plan="r",
        do_nothing_option="d",
    )


class TestPlanValidation:
    def test_accepts_a_sane_plan(self):
        world = load_world("bad_deploy")
        assert validate_plan(_plan(action="rollback_deploy", target="dpl-8814"), world).ok

    def test_rejects_unknown_deploy(self):
        world = load_world("bad_deploy")
        assert not validate_plan(_plan(action="rollback_deploy", target="dpl-9999"), world).ok

    def test_rejects_apply_config_without_key(self):
        """The exact bug the first end-to-end run shipped to execution."""
        world = load_world("pool_exhaustion")
        result = validate_plan(
            _plan(action="apply_config", target="payment-service", parameters={}), world
        )
        assert not result.ok
        assert any("key" in e for e in result.errors)

    def test_rejects_scale_down_during_incident(self):
        """The other one: 4 replicas -> 1 while the service was failing."""
        world = load_world("pool_exhaustion")
        result = validate_plan(
            _plan(
                action="scale_service",
                target="payment-service",
                parameters={"replicas": "1"},
            ),
            world,
        )
        assert not result.ok
        assert any("REDUCE" in e for e in result.errors)

    def test_allows_scale_up(self):
        world = load_world("pool_exhaustion")
        assert validate_plan(
            _plan(
                action="scale_service",
                target="payment-service",
                parameters={"replicas": "8"},
            ),
            world,
        ).ok

    def test_rejects_two_rollbacks_at_once(self):
        world = load_world("bad_deploy")
        plan = RemediationPlan(
            summary="s",
            steps=[
                RemediationStep(action="rollback_deploy", target="dpl-8814", rationale="a"),
                RemediationStep(action="rollback_deploy", target="dpl-8813", rationale="b"),
            ],
            expected_effect="e",
            rollback_plan="r",
            do_nothing_option="d",
        )
        assert not validate_plan(plan, world).ok

    def test_rejects_unknown_service(self):
        world = load_world("bad_deploy")
        assert not validate_plan(_plan(target="not-a-service"), world).ok


class TestSchemas:
    def test_hypothesis_requires_confidence(self):
        with pytest.raises(Exception):
            Hypothesis(root_cause="x", category=Category.UNKNOWN, affected_service="y")

    def test_hypothesis_roundtrips(self):
        h = Hypothesis(
            root_cause="x",
            category=Category.BAD_DEPLOY,
            affected_service="checkout-api",
            confidence=Confidence.HIGH,
        )
        assert Hypothesis.model_validate_json(h.model_dump_json()) == h

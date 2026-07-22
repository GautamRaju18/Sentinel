"""The simulator must be deterministic — the eval suite depends on it."""

from __future__ import annotations

import pytest

from simulator.scenarios import list_scenarios, load_scenario
from simulator.world import SimulatedWorld, load_world


@pytest.mark.parametrize("slug", list_scenarios())
class TestScenarios:
    def test_produces_signal(self, slug):
        s = load_scenario(slug)
        assert s.logs, f"{slug} has no logs"
        assert s.metrics, f"{slug} has no metrics"
        assert s.alert.body

    def test_is_deterministic(self, slug):
        a, b = load_scenario(slug), load_scenario(slug)
        assert [e.render() for e in a.logs] == [e.render() for e in b.logs]
        assert {k: v.summarize() for k, v in a.metrics.items()} == {
            k: v.summarize() for k, v in b.metrics.items()
        }

    def test_ground_truth_is_complete(self, slug):
        gt = load_scenario(slug).ground_truth
        assert gt.root_cause and gt.correct_remediation
        assert len(gt.key_evidence) >= 3

    def test_ground_truth_not_reachable_through_queries(self, slug):
        """The answer key must not leak through any tool-facing method."""
        w = load_world(slug)
        gt = w.scenario.ground_truth
        surface = "\n".join(
            [
                *(e.render() for e in w.logs),
                *(d.render() for d in w.deploys),
                *(h.render() for h in w.get_health()),
                w.alert.render(),
            ]
        )
        # A distinctive phrase from the answer should not appear verbatim.
        assert gt.root_cause[:60] not in surface


class TestWorldActions:
    def test_correct_rollback_heals(self):
        w = load_world("bad_deploy")
        assert not w.remediated
        result = w.rollback_deploy("dpl-8814")
        assert result.ok and result.changed.get("healed")
        assert w.remediated

    def test_wrong_rollback_does_not_heal(self):
        w = load_world("bad_deploy")
        result = w.rollback_deploy("dpl-8813")
        assert result.ok
        assert not result.changed.get("healed")
        assert not w.remediated

    def test_unknown_deploy_fails(self):
        w = load_world("bad_deploy")
        assert not w.rollback_deploy("dpl-0000").ok

    def test_double_rollback_refused(self):
        w = load_world("bad_deploy")
        w.rollback_deploy("dpl-8814")
        assert not w.rollback_deploy("dpl-8814").ok

    def test_restart_does_not_fix_expired_cert(self):
        w = load_world("cert_expiry")
        result = w.restart_service("auth-service")
        assert result.ok
        assert not result.changed.get("healed")

    def test_config_fix_heals_config_incident(self):
        w = load_world("pool_exhaustion")
        result = w.apply_config("payment-service", "maximumPoolSize", "40")
        assert result.changed.get("healed")

    def test_scale_bounds_enforced(self):
        w = load_world("bad_deploy")
        assert not w.scale_service("checkout-api", 0).ok
        assert not w.scale_service("checkout-api", 999).ok


class TestQueries:
    def test_level_filter_is_minimum_not_exact(self):
        w = load_world("memory_leak")
        errors = w.query_logs(level="ERROR", limit=200)
        assert errors
        # FATAL is above ERROR, so it must be included.
        assert any(e.level == "FATAL" for e in errors)

    def test_contains_filter(self):
        w = load_world("cert_expiry")
        assert w.query_logs(contains="x509", limit=50)

    def test_unknown_service_returns_nothing(self):
        assert load_world("bad_deploy").get_health("nope") == []

    def test_isolation_between_loads(self):
        w1 = load_world("bad_deploy")
        w1.rollback_deploy("dpl-8814")
        w2 = load_world("bad_deploy")
        assert not w2.remediated, "worlds share mutable state between loads"


def test_world_is_a_world():
    assert isinstance(load_world("bad_deploy"), SimulatedWorld)

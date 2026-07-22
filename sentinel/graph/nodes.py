"""Graph nodes.

Each node is a plain async function: state in, partial state out. Keeping them
free of graph wiring means they can be unit-tested directly, and it keeps the
topology in one readable place (builder.py) instead of smeared across the code.

The interesting nodes are `critique` and `route_after_critique`: together they
form the reflection cycle that lets the agent decide it has not finished
thinking. That loop is bounded by MAX_INVESTIGATION_LOOPS, because an agent
that is confused does not become less confused by iterating.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from sentinel.agents import run_agent
from sentinel.agents.prompts import INVESTIGATOR_SYSTEM
from sentinel.config import get_settings
from sentinel.graph.schemas import (
    Category,
    Confidence,
    Critique,
    Evidence,
    Hypothesis,
    PostMortem,
    RemediationPlan,
    Verification,
)
from sentinel.graph.state import IncidentState
from sentinel.logging_setup import get_logger
from sentinel.models.router import ModelTier, get_model
from sentinel.models.structured import generate_structured
from sentinel.tools import get_tools
from sentinel.tools.remediation import grant_approval, revoke_all_approvals
from simulator.world import get_world

log = get_logger(__name__)


def _usage(inp: int = 0, out: int = 0) -> dict[str, int]:
    return {"input": inp, "output": out}


# ---------------------------------------------------------------------------
# 1. Triage
# ---------------------------------------------------------------------------


async def triage_node(state: IncidentState) -> dict[str, Any]:
    """Classify the alert. Cheap tier — this is the Phase 6 fine-tune target."""
    from sentinel.triage import classify

    result = await classify(state["alert"])
    log.info(
        "node.triage",
        severity=str(result.severity),
        category=str(result.category),
        service=result.affected_service,
    )
    return {
        "triage": result,
        "stage": "investigate",
        "messages": [
            AIMessage(f"Triage: {result.severity} {result.category} — {result.reasoning}")
        ],
    }


# ---------------------------------------------------------------------------
# 2. Investigate — a bounded ReAct loop with read-only tools
# ---------------------------------------------------------------------------


async def investigate_node(state: IncidentState) -> dict[str, Any]:
    """Gather evidence. On later loops, steered by the critic's questions."""
    loop = state.get("loop_count", 0)
    triage = state.get("triage")
    questions = state.get("open_questions") or []

    parts = [f"Investigate this production alert.\n\n{state['alert']}"]
    if triage:
        parts.append(
            f"\nInitial triage: {triage.severity} / {triage.category} "
            f"(service: {triage.affected_service or 'unknown'})"
        )
    if runbook := state.get("runbook_context"):
        parts.append(f"\nRelevant runbook guidance:\n{runbook}")
    if similar := state.get("similar_incidents"):
        parts.append(f"\nSimilar past incidents:\n{similar}")
    if questions:
        parts.append(
            "\nA previous investigation round was judged incomplete. Answer these "
            "specific questions with tool calls — do not repeat work already done:\n"
            + "\n".join(f"  {i}. {q}" for i, q in enumerate(questions, 1))
        )

    run = await run_agent(
        task="\n".join(parts),
        tools=get_tools(read_only=True),
        system_prompt=INVESTIGATOR_SYSTEM,
        tier=ModelTier.REASONER,
        max_steps=10 if loop == 0 else 6,
    )

    evidence = [
        Evidence(
            observation=inv.result[:600],
            source=inv.name,
            supports=f"round {loop + 1} investigation",
            strength="moderate",
        )
        for inv in run.invocations
        if not inv.error
    ]

    flags: list[str] = []
    for inv in run.invocations:
        if "⚠ SECURITY" in inv.result:
            flags.append(f"injection attempt observed via {inv.name}")

    log.info("node.investigate", loop=loop + 1, tools=len(run.invocations), evidence=len(evidence))

    return {
        "evidence": evidence,
        "loop_count": loop + 1,
        "stage": "synthesize",
        "tool_trajectory": run.tool_sequence,
        "security_flags": flags,
        "token_usage": _usage(run.input_tokens, run.output_tokens),
        "messages": [AIMessage(run.final_text or "(investigation produced no summary)")],
    }


# ---------------------------------------------------------------------------
# 3. Synthesize a hypothesis
# ---------------------------------------------------------------------------

_SYNTH_SYSTEM = """\
You are a senior SRE forming a root-cause hypothesis from collected evidence.

Rules:
- Only assert what the evidence supports. If the mechanism is unclear, say so in
  `unknowns` rather than inventing a plausible story.
- The trigger is a specific deploy id, config id, or event — or null when
  nothing changed. Do not name a deploy you have not seen in the evidence.
- Confidence is about EVIDENCE, not about how reasonable your story sounds.
  high   = the causal chain is observed end to end
  medium = the chain is likely but one link is inferred
  low    = several competing explanations still fit
"""


async def synthesize_node(state: IncidentState) -> dict[str, Any]:
    evidence = state.get("evidence") or []
    critique = state.get("critique")

    bundle = "\n\n".join(
        f"[{i}] via {e.source}\n{e.observation}" for i, e in enumerate(evidence[-40:])
    )
    prompt = f"ALERT:\n{state['alert']}\n\nEVIDENCE:\n{bundle}"
    if critique and critique.verdict == "revise":
        prompt += (
            f"\n\nYour previous hypothesis was rejected (score {critique.score}/10).\n"
            f"Gaps: {'; '.join(critique.gaps)}\n"
            f"Alternatives to rule out: {'; '.join(critique.alternative_causes)}\n"
            f"Produce a revised hypothesis that addresses these."
        )

    fallback = Hypothesis(
        root_cause="Insufficient evidence to determine a root cause.",
        category=Category.UNKNOWN,
        affected_service=(state.get("triage").affected_service if state.get("triage") else "")
        or "unknown",
        confidence=Confidence.LOW,
        unknowns=["synthesis failed to produce a valid hypothesis"],
    )
    hypothesis = await generate_structured(
        get_model(ModelTier.PLANNER),
        Hypothesis,
        prompt,
        system=_SYNTH_SYSTEM,
        default=fallback,
    )
    log.info(
        "node.synthesize",
        confidence=str(hypothesis.confidence),
        trigger=hypothesis.trigger,
        service=hypothesis.affected_service,
    )
    return {
        "hypothesis": hypothesis,
        "stage": "critique",
        "messages": [AIMessage(f"Hypothesis: {hypothesis.root_cause}")],
    }


# ---------------------------------------------------------------------------
# 4. Critic — an adversarial reader, not a cheerleader
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM = """\
You are a skeptical incident reviewer. Your job is to find what is WRONG with a
colleague's root-cause hypothesis before anyone acts on it. Acting on a wrong
hypothesis during an outage makes the outage longer.

Attack it on these axes:
1. Grounding — is every claim traceable to an actual observation? A hypothesis
   that names a mechanism nobody measured is a story, not a diagnosis.
2. Correlation vs causation — a deploy near the onset is suggestive, not proof.
   Was the mechanism actually verified, or just assumed from the commit message?
3. Direction — could the named service be a VICTIM rather than the cause? Which
   component degraded first?
4. Alternatives — does another explanation fit this same evidence?
5. Sufficiency — would this explain the MAGNITUDE observed, not just the direction?

Verdict 'accept' only when the causal chain is observed end to end and no
alternative fits as well. Otherwise 'revise', and write next_questions as
specific, answerable tool queries — "check whether request rate rose on
checkout-api", not "investigate further".

Be demanding but not obstructive: if the chain is genuinely well evidenced,
accept it. Endless revision during an outage is its own failure mode.
"""


async def critique_node(state: IncidentState) -> dict[str, Any]:
    hypothesis = state.get("hypothesis")
    evidence = state.get("evidence") or []
    loop = state.get("loop_count", 0)
    max_loops = get_settings().max_investigation_loops

    if hypothesis is None:
        return {"stage": "plan", "errors": ["critique ran with no hypothesis"]}

    bundle = "\n\n".join(
        f"[{i}] via {e.source}\n{e.observation[:400]}" for i, e in enumerate(evidence[-30:])
    )
    prompt = (
        f"ALERT:\n{state['alert']}\n\n"
        f"HYPOTHESIS:\n{hypothesis.model_dump_json(indent=2)}\n\n"
        f"EVIDENCE AVAILABLE:\n{bundle}\n\n"
        f"This is investigation round {loop} of at most {max_loops}."
    )
    critique = await generate_structured(
        get_model(ModelTier.PLANNER),
        Critique,
        prompt,
        system=_CRITIC_SYSTEM,
        default=Critique(
            verdict="accept",
            score=5,
            reasoning="critic failed to produce a verdict; accepting to avoid stalling",
        ),
    )
    log.info("node.critique", verdict=critique.verdict, score=critique.score, loop=loop)
    return {
        "critique": critique,
        "open_questions": critique.next_questions,
        "messages": [
            AIMessage(f"Critique [{critique.verdict}, {critique.score}/10]: {critique.reasoning}")
        ],
    }


def route_after_critique(state: IncidentState) -> str:
    """The cycle. Loop back to investigate, or move on to planning."""
    critique = state.get("critique")
    loop = state.get("loop_count", 0)
    max_loops = get_settings().max_investigation_loops

    if loop >= max_loops:
        log.warning("route.loop_cap", loop=loop, cap=max_loops)
        return "plan"
    if critique and critique.verdict == "revise" and critique.next_questions:
        log.info("route.reinvestigate", score=critique.score, loop=loop)
        return "investigate"
    return "plan"


# ---------------------------------------------------------------------------
# 5. Plan
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """\
You write remediation plans for a human operator to approve.

Constraints:
- Every step maps to exactly one available action: rollback_deploy,
  restart_service, scale_service, apply_config.
- `target` must be a real deploy id or service name observed in the evidence.
  Never invent one.
- REQUIRED PARAMETERS. A step missing these will be rejected:
    apply_config   -> parameters MUST contain "key" and "value", both non-empty
                      (e.g. {"key": "maximumPoolSize", "value": "40"})
    scale_service  -> parameters MUST contain "replicas" as an integer string
- NEVER scale a service DOWN during an active incident. Removing capacity from
  a failing system makes it fail harder.
- Prefer the SMALLEST plan that addresses the root cause. One well-targeted step
  beats four hopeful ones — when four things change at once and the incident
  resolves, nobody learns which one worked.
- Restarting a service to clear a symptom is a mitigation, not a fix. Say so in
  the rationale if you propose it.
- Roll back at most one deploy per plan, so its effect is observable.
- Always populate do_nothing_option. Sometimes waiting is correct, and the
  operator deserves that comparison.
- State risks plainly, including the risk that your hypothesis is wrong.
"""


async def plan_node(state: IncidentState) -> dict[str, Any]:
    hypothesis = state.get("hypothesis")
    critique = state.get("critique")

    prompt = (
        f"ALERT:\n{state['alert']}\n\n"
        f"ROOT CAUSE HYPOTHESIS:\n{hypothesis.model_dump_json(indent=2) if hypothesis else 'none'}\n\n"
        f"REVIEWER NOTES:\n{critique.reasoning if critique else 'none'}\n"
        f"Remaining doubts: {'; '.join(critique.gaps) if critique and critique.gaps else 'none'}\n\n"
        f"Write the remediation plan."
    )
    plan = await generate_structured(
        get_model(ModelTier.PLANNER),
        RemediationPlan,
        prompt,
        system=_PLAN_SYSTEM,
        default=RemediationPlan(
            summary="No automated remediation proposed — escalate to a human.",
            steps=[
                {
                    "action": "restart_service",
                    "target": (hypothesis.affected_service if hypothesis else "unknown"),
                    "rationale": "placeholder; planning failed",
                    "blast_radius": "high",
                }
            ],
            expected_effect="unknown",
            rollback_plan="n/a",
            do_nothing_option="Escalate to the on-call engineer.",
        ),
    )
    # Validate before showing it to a human. An operator should never be asked
    # to approve a plan with an empty config key in it; catching that here also
    # gives the model a precise repair signal rather than a vague retry.
    from sentinel.graph.validation import validate_plan

    validation = validate_plan(plan, get_world())
    if not validation.ok:
        log.warning("node.plan.invalid_retrying", errors=validation.errors)
        plan = await generate_structured(
            get_model(ModelTier.PLANNER),
            RemediationPlan,
            f"{prompt}\n\nYour previous plan was REJECTED by validation:\n"
            + "\n".join(f"- {e}" for e in validation.errors)
            + "\n\nProduce a corrected plan that fixes exactly these problems.",
            system=_PLAN_SYSTEM,
            default=plan,
        )
        validation = validate_plan(plan, get_world())

    log.info(
        "node.plan",
        steps=len(plan.steps),
        valid=validation.ok,
        summary=plan.summary[:80],
    )
    return {
        "plan": plan,
        "stage": "await_approval",
        "errors": [f"plan validation: {e}" for e in validation.errors],
        "messages": [AIMessage(f"Plan: {plan.summary}")],
    }


# ---------------------------------------------------------------------------
# 6. Approval gate — the graph interrupts here
# ---------------------------------------------------------------------------


async def approval_node(state: IncidentState) -> dict[str, Any]:
    """Runs only AFTER a human resumed the graph with a decision.

    The interrupt itself is configured on the graph (interrupt_before), so
    execution stops before this node ever runs. When it does run, `approved`
    has been written into state from outside the agent.
    """
    approved = state.get("approved")
    plan = state.get("plan")

    if not approved:
        log.info("node.approval.rejected", note=state.get("approval_note", ""))
        return {
            "stage": "postmortem",
            "messages": [
                HumanMessage(
                    f"Operator REJECTED the plan. {state.get('approval_note') or ''}".strip()
                )
            ],
        }

    # Grant only the specific actions this plan asked for — not a blanket unlock.
    revoke_all_approvals()
    for step in plan.steps if plan else []:
        grant_approval(step.action)

    log.info("node.approval.granted", actions=[s.action for s in (plan.steps if plan else [])])
    return {
        "stage": "execute",
        "messages": [
            HumanMessage(f"Operator APPROVED the plan. {state.get('approval_note') or ''}".strip())
        ],
    }


def route_after_approval(state: IncidentState) -> str:
    return "execute" if state.get("approved") else "postmortem"


# ---------------------------------------------------------------------------
# 7. Execute
# ---------------------------------------------------------------------------


async def execute_node(state: IncidentState) -> dict[str, Any]:
    """Run the approved steps directly — no model in the loop.

    The plan was already reviewed by a human; re-deriving the tool calls from a
    model here would let it drift from what was approved. Deterministic
    execution of an approved plan is the whole point of the gate.
    """
    from sentinel.tools import get_tool

    plan = state.get("plan")
    if plan is None:
        return {"stage": "verify", "errors": ["execute ran with no plan"]}

    from sentinel.graph.validation import validate_plan, validate_step

    world = get_world()

    # Last line of defence. A plan can be schema-valid and still incoherent —
    # an apply_config with no key, or a scale-down during an outage. Neither is
    # something we want to discover by executing it.
    validation = validate_plan(plan, world)
    if not validation.ok:
        log.error("execute.plan_rejected", errors=validation.errors)
        return {
            "stage": "verify",
            "errors": [f"plan rejected before execution: {e}" for e in validation.errors],
            "messages": [
                AIMessage(
                    "Execution aborted — the approved plan failed validation:\n"
                    + validation.render()
                )
            ],
        }
    for w in validation.warnings:
        log.warning("execute.plan_warning", warning=w)

    performed: list[str] = []
    errors: list[str] = []

    for step in plan.steps:
        # Re-validate per step: earlier steps mutate the world, so a step that
        # was valid when the plan was written may not be valid by its turn.
        step_check = validate_step(step, world)
        if not step_check.ok:
            errors.extend(step_check.errors)
            log.warning("execute.step_skipped", action=step.action, errors=step_check.errors)
            continue
        try:
            tool = get_tool(step.action)
            args: dict[str, Any] = {"reason": step.rationale}
            if step.action == "rollback_deploy":
                args["deploy_id"] = step.target
            else:
                args["service"] = step.target
            if step.action == "scale_service":
                args["replicas"] = int(step.parameters["replicas"])
            if step.action == "apply_config":
                args["key"] = step.parameters["key"]
                args["value"] = step.parameters["value"]

            result = await tool.ainvoke(args)
            performed.append(f"{step.action}({step.target}) -> {result}")
            log.info("node.execute", action=step.action, target=step.target)
        except Exception as e:
            msg = f"{step.action}({step.target}) failed: {type(e).__name__}: {e}"
            errors.append(msg)
            log.error("node.execute.failed", action=step.action, error=str(e))

    revoke_all_approvals()  # approval is single-use
    return {
        "executed_actions": performed,
        "errors": errors,
        "stage": "verify",
        "messages": [AIMessage("Executed:\n" + "\n".join(performed))],
    }


# ---------------------------------------------------------------------------
# 8. Verify
# ---------------------------------------------------------------------------


async def verify_node(state: IncidentState) -> dict[str, Any]:
    world = get_world()
    health = "\n".join(h.render() for h in world.get_health())
    actions = "\n".join(state.get("executed_actions") or []) or "none"

    verification = await generate_structured(
        get_model(ModelTier.REASONER),
        Verification,
        (
            f"Actions taken:\n{actions}\n\n"
            f"Service health now:\n{health}\n\n"
            f"Did the remediation resolve the incident? Be strict: a service that is "
            f"merely restarted but will fail again is NOT resolved."
        ),
        system="You verify whether a remediation worked. Report what you observe, not what you hoped.",
        default=Verification(
            resolved=world.remediated,
            observations=[health],
            next_action="Manual verification required.",
        ),
    )
    log.info("node.verify", resolved=verification.resolved)
    return {
        "verification": verification,
        "stage": "postmortem",
        "messages": [AIMessage(f"Verification: resolved={verification.resolved}")],
    }


# ---------------------------------------------------------------------------
# 9. Post-mortem — and the write into long-term memory
# ---------------------------------------------------------------------------


async def postmortem_node(state: IncidentState) -> dict[str, Any]:
    hypothesis = state.get("hypothesis")
    verification = state.get("verification")

    prompt = (
        f"ALERT:\n{state['alert']}\n\n"
        f"ROOT CAUSE:\n{hypothesis.root_cause if hypothesis else 'not determined'}\n\n"
        f"ACTIONS TAKEN:\n{chr(10).join(state.get('executed_actions') or ['none'])}\n\n"
        f"OUTCOME: {'resolved' if verification and verification.resolved else 'unresolved'}\n\n"
        f"Write the post-mortem."
    )
    pm = await generate_structured(
        get_model(ModelTier.REASONER),
        PostMortem,
        prompt,
        system=(
            "You write blameless post-mortems. Describe systems and processes, never "
            "individuals: 'the pipeline had no query-count regression check', not "
            "'the author forgot'. The `lesson` field is stored in long-term memory and "
            "retrieved during future incidents, so make it generalisable rather than "
            "specific to these exact service names."
        ),
        default=PostMortem(
            title="Incident post-mortem",
            timeline=["post-mortem generation failed"],
            impact="unknown",
            root_cause=hypothesis.root_cause if hypothesis else "not determined",
            action_items=["Write this post-mortem manually."],
            lesson="n/a",
        ),
    )

    # Persist to the incident memory store so future investigations can find it.
    try:
        from sentinel.memory.incidents import remember_incident

        await remember_incident(state, pm)
    except Exception as e:
        log.warning("postmortem.memory_write_failed", error=str(e))

    log.info("node.postmortem", title=pm.title)
    return {
        "postmortem": pm,
        "stage": "done",
        "messages": [AIMessage(f"Post-mortem: {pm.title}")],
    }


# ---------------------------------------------------------------------------
# Retrieval node (wired in Phase 4)
# ---------------------------------------------------------------------------


async def retrieve_node(state: IncidentState) -> dict[str, Any]:
    """Pull runbooks and similar past incidents before investigating."""
    try:
        from sentinel.rag.retriever import retrieve_context

        runbooks, similar = await retrieve_context(state["alert"], state.get("triage"))
        log.info("node.retrieve", runbook_chars=len(runbooks), similar_chars=len(similar))
        return {"runbook_context": runbooks, "similar_incidents": similar}
    except Exception as e:
        # Retrieval is an enhancement, never a hard dependency. Degrade quietly.
        log.warning("node.retrieve.failed", error=str(e))
        return {"runbook_context": "", "similar_incidents": ""}

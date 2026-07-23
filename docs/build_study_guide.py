"""Generate the Sentinel study guide PDF.

A from-scratch explainer: plain English, analogies, every concept, and prep for
explaining it out loud. Built with reportlab Platypus so it needs no system
libraries (weasyprint's GTK stack is not installable under Smart App Control).

    uv run python docs/build_study_guide.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).parent / "Sentinel_Study_Guide.pdf"

# --- palette --------------------------------------------------------------
INK = colors.HexColor("#0f172a")
SLATE = colors.HexColor("#334155")
MUTED = colors.HexColor("#64748b")
INDIGO = colors.HexColor("#4f46e5")
INDIGO_L = colors.HexColor("#eef2ff")
GREEN = colors.HexColor("#16a34a")
GREEN_L = colors.HexColor("#f0fdf4")
AMBER = colors.HexColor("#d97706")
AMBER_L = colors.HexColor("#fffbeb")
RED = colors.HexColor("#dc2626")
RED_L = colors.HexColor("#fef2f2")
CODE_BG = colors.HexColor("#0f172a")
CODE_FG = colors.HexColor("#e2e8f0")
RULE = colors.HexColor("#e2e8f0")

# --- styles ---------------------------------------------------------------
ss = getSampleStyleSheet()


def style(name, **kw):
    return ParagraphStyle(name, parent=ss["Normal"], **kw)


S = {
    "title": style(
        "t", fontName="Helvetica-Bold", fontSize=30, leading=34, textColor=INK, spaceAfter=6
    ),
    "subtitle": style(
        "st", fontName="Helvetica", fontSize=13, leading=18, textColor=MUTED, spaceAfter=4
    ),
    "h1": style(
        "h1",
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        textColor=INDIGO,
        spaceBefore=6,
        spaceAfter=8,
    ),
    "h2": style(
        "h2",
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=INK,
        spaceBefore=12,
        spaceAfter=5,
    ),
    "h3": style(
        "h3",
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=SLATE,
        spaceBefore=8,
        spaceAfter=3,
    ),
    "body": style(
        "b",
        fontName="Helvetica",
        fontSize=10.5,
        leading=15.5,
        textColor=INK,
        spaceAfter=7,
        alignment=TA_LEFT,
    ),
    "small": style(
        "sm", fontName="Helvetica", fontSize=9, leading=13, textColor=MUTED, spaceAfter=4
    ),
    "bullet": style("bl", fontName="Helvetica", fontSize=10.5, leading=15, textColor=INK),
    "code": style("c", fontName="Courier", fontSize=8.6, leading=12.2, textColor=CODE_FG),
    "kicker": style(
        "k", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=INDIGO, spaceAfter=2
    ),
    "toc": style("toc", fontName="Helvetica", fontSize=11, leading=19, textColor=INK),
    "tochead": style("toch", fontName="Helvetica-Bold", fontSize=11, leading=19, textColor=INDIGO),
    "cardtitle": style(
        "ct", fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=INK, spaceAfter=3
    ),
    "callht": style(
        "ch", fontName="Helvetica-Bold", fontSize=9.5, leading=13, textColor=INK, spaceAfter=2
    ),
    "callbody": style("cb", fontName="Helvetica", fontSize=9.8, leading=14, textColor=INK),
}


def P(text, s="body"):
    return Paragraph(text, S[s])


def bullets(items, s="bullet", bullet="•"):
    return ListFlowable(
        [ListItem(Paragraph(t, S[s]), value=bullet, leftIndent=6) for t in items],
        bulletType="bullet",
        start=bullet,
        leftIndent=14,
        spaceAfter=6,
    )


def code(lines):
    body = "<br/>".join(
        line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
        for line in lines
    )
    t = Table([[Paragraph(body, S["code"])]], colWidths=[16.4 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("ROUNDEDCORNERS", [4, 4, 4, 4]),
            ]
        )
    )
    return t


def callout(kind, heading, body_html):
    bg, bar = {
        "info": (INDIGO_L, INDIGO),
        "good": (GREEN_L, GREEN),
        "warn": (AMBER_L, AMBER),
        "bad": (RED_L, RED),
    }[kind]
    inner = [Paragraph(heading, S["callht"])] if heading else []
    inner.append(Paragraph(body_html, S["callbody"]))
    t = Table([[inner]], colWidths=[16.0 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return t


def kv_table(rows, col0=5.2, col1=11.0, header=None):
    data = []
    if header:
        data.append(
            [
                Paragraph(f"<b>{header[0]}</b>", S["small"]),
                Paragraph(f"<b>{header[1]}</b>", S["small"]),
            ]
        )
    for a, b in rows:
        data.append([Paragraph(a, S["small"]), Paragraph(b, S["small"])])
    t = Table(data, colWidths=[col0 * cm, col1 * cm])
    style_cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
    ]
    if header:
        style_cmds.append(("LINEBELOW", (0, 0), (-1, 0), 1, INDIGO))
    t.setStyle(TableStyle(style_cmds))
    return t


def compare_table(header, rows):
    """3-col comparison with coloured cells."""
    data = [[Paragraph(f"<b>{h}</b>", S["small"]) for h in header]]
    for r in rows:
        data.append([Paragraph(str(x), S["small"]) for x in r])
    t = Table(data, colWidths=[6.4 * cm, 4.8 * cm, 4.8 * cm])
    t.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    return t


def rule():
    return HRFlowable(width="100%", thickness=0.7, color=RULE, spaceBefore=8, spaceAfter=8)


def concept_card(num, title, tag):
    head = Table(
        [
            [
                Paragraph(
                    f"<b>{num}</b>",
                    style("cn", fontName="Helvetica-Bold", fontSize=15, textColor=colors.white),
                ),
                Paragraph(f"{title}<br/><font size=8 color='#64748b'>{tag}</font>", S["cardtitle"]),
            ]
        ],
        colWidths=[1.1 * cm, 15.0 * cm],
    )
    head.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), INDIGO),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (0, 0), 8),
                ("BOTTOMPADDING", (0, 0), (0, 0), 8),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
                ("LEFTPADDING", (1, 0), (1, 0), 10),
            ]
        )
    )
    return head


# --- page furniture -------------------------------------------------------


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "Sentinel — Autonomous Incident Response Agent")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.line(2 * cm, 1.55 * cm, A4[0] - 2 * cm, 1.55 * cm)
    canvas.restoreState()


def on_cover(canvas, doc):
    canvas.saveState()
    # Dark banner across the top third.
    banner_h = 9.5 * cm
    canvas.setFillColor(INK)
    canvas.rect(0, A4[1] - banner_h, A4[0], banner_h, fill=1, stroke=0)
    canvas.setFillColor(INDIGO)
    canvas.rect(0, A4[1] - banner_h - 0.18 * cm, A4[0], 0.18 * cm, fill=1, stroke=0)
    # Title text drawn directly on the banner so it sits on the dark fill.
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 46)
    canvas.drawString(2 * cm, A4[1] - 4.6 * cm, "SENTINEL")
    canvas.setFillColor(colors.HexColor("#c7d2fe"))
    canvas.setFont("Helvetica", 16)
    canvas.drawString(2 * cm, A4[1] - 5.6 * cm, "An Autonomous Incident-Response Agent")
    canvas.setFillColor(colors.HexColor("#94a3b8"))
    canvas.setFont("Helvetica", 10)
    canvas.drawString(2 * cm, A4[1] - 8.6 * cm, "A complete, from-scratch study guide")
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Sentinel — Study Guide",
        author="Sentinel project",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    cover_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="cover")
    doc.addPageTemplates(
        [
            PageTemplate(id="cover", frames=[cover_frame], onPage=on_cover),
            PageTemplate(id="body", frames=[frame], onPage=on_page),
        ]
    )

    e = []  # elements
    e += cover()
    e.append(PageBreak())
    e.append(_next_template())
    e += toc()
    e.append(PageBreak())
    e += part1_big_picture()
    e.append(PageBreak())
    e += part2_flow()
    e.append(PageBreak())
    e += part3_concepts()
    e.append(PageBreak())
    e += part4_results()
    e.append(PageBreak())
    e += part5_explain()
    e.append(PageBreak())
    e += part6_glossary()

    doc.build(e)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


class _NextTemplate(PageBreak):
    """Switch to the body page template after the cover."""


def _next_template():
    from reportlab.platypus import NextPageTemplate

    return NextPageTemplate("body")


# ==========================================================================
# CONTENT
# ==========================================================================


def cover():
    # The banner + title are drawn in on_cover(); flowed content starts below it.
    return [
        Spacer(1, 8.6 * cm),
        Paragraph(
            "The Complete Study Guide",
            style(
                "cvt",
                fontName="Helvetica-Bold",
                fontSize=24,
                leading=30,
                textColor=INK,
                spaceAfter=10,
            ),
        ),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "Everything in this project, explained from scratch — in plain English, "
            "with analogies, and with the exact words to use when you explain it out "
            "loud. Read it cold; teach it after.",
            style("cvd", fontName="Helvetica", fontSize=12.5, leading=18, textColor=SLATE),
        ),
        Spacer(1, 1.2 * cm),
        kv_table(
            [
                (
                    "What it is",
                    "An AI agent that investigates production incidents and proposes fixes for a human to approve.",
                ),
                (
                    "Core stack",
                    "LangGraph · LangChain · MCP · QLoRA fine-tuning · pgvector RAG · FastAPI · Streamlit",
                ),
                (
                    "The headline",
                    "Fine-tuned a 1.5B model to 100% category accuracy — then proved with a harder test that it had learned patterns, not reasoning.",
                ),
            ],
            col0=3.4,
            col1=12.8,
        ),
    ]


def toc():
    rows = [
        ("PART 1", "The Big Picture — what it is and why it exists", True),
        ("", "The elevator pitch, the problem, the factory-floor analogy", False),
        ("PART 2", "The Flow — what happens, step by step", True),
        ("", "Alert in, ten stages, human gate, fix out", False),
        ("PART 3", "Every Concept, From Scratch", True),
        ("", "1  Tool calling / the agentic loop", False),
        ("", "2  LangGraph — state machines for agents", False),
        ("", "3  Human-in-the-loop, done properly", False),
        ("", "4  Structured output", False),
        ("", "5  RAG and hybrid retrieval", False),
        ("", "6  The critic / multi-agent pattern", False),
        ("", "7  MCP — the Model Context Protocol", False),
        ("", "8  Fine-tuning with QLoRA", False),
        ("", "9  Evaluation — the part most projects skip", False),
        ("", "10 Prompt-injection defence", False),
        ("", "11 Model routing by tier", False),
        ("", "12 Observability", False),
        ("PART 4", "The Results — and the honest story", True),
        ("", "The numbers, and why 100% was a warning sign", False),
        ("PART 5", "How to Explain It", True),
        ("", "Resume bullets, tech stack, interview Q&A", False),
        ("PART 6", "Glossary — every term in one place", True),
    ]
    out = [P("Contents", "h1"), Spacer(1, 4)]
    data = []
    for tag, text, head in rows:
        st = "tochead" if head else "toc"
        data.append([Paragraph(f"<b>{tag}</b>" if tag else "", S["small"]), Paragraph(text, S[st])])
    t = Table(data, colWidths=[2.2 * cm, 14.2 * cm])
    t.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]
        )
    )
    out.append(t)
    return out


def part1_big_picture():
    e = [P("Part 1", "kicker"), P("The Big Picture", "h1"), rule()]

    e.append(P("The 30-second pitch", "h2"))
    e.append(
        P(
            "Sentinel is an AI agent that does what a sleepy on-call engineer does at "
            "3 a.m. when a pager goes off. An alert fires — say, <i>checkout is slow</i>. "
            "Sentinel reads the alert, decides which logs and metrics to pull, looks at "
            "what changed recently, forms a theory of what broke, <b>argues against its "
            "own theory</b>, and if the theory holds up, writes a step-by-step fix. Then "
            "it stops and asks a human to approve before it touches anything. A person "
            "clicks approve; it executes the fix and checks whether the system recovered; "
            "then it writes a short post-mortem and remembers the lesson for next time."
        )
    )

    e.append(
        callout(
            "info",
            "The one sentence to memorise",
            "&ldquo;It&rsquo;s an autonomous agent that investigates production incidents "
            "end-to-end, but it physically cannot change anything without a human "
            "approving first — safety is built into the structure, not asked for "
            "politely.&rdquo;",
        )
    )

    e.append(P("What problem does it solve?", "h2"))
    e.append(
        P(
            "When something breaks in production, the slow and expensive part is not "
            "fixing it — it&rsquo;s the <b>investigation</b>. A human has to figure out "
            "which of a hundred services is actually the culprit, dig through logs, "
            "correlate a deploy against a latency spike, and rule out red herrings. That "
            "takes minutes to hours, and every minute is downtime. Sentinel automates the "
            "investigation and hands a human a well-reasoned plan to approve — turning "
            "&ldquo;figure it out from scratch&rdquo; into &ldquo;check this and click "
            "yes.&rdquo;"
        )
    )

    e.append(P("The analogy that makes it click", "h2"))
    e.append(
        P(
            "Think of a <b>hospital emergency room</b>. A patient arrives (the alert). A "
            "triage nurse rates how urgent it is (triage). A doctor orders tests — blood "
            "work, X-rays (the read-only tools: logs, metrics, deploys). The doctor forms "
            "a diagnosis (hypothesis), and a second senior doctor challenges it to catch "
            "mistakes (the critic). They agree on a treatment (the plan) — but a "
            "controlled drug requires a second signature before it&rsquo;s administered "
            "(the human approval gate). After treatment, they check the patient recovered "
            "(verify) and write it up (post-mortem). Sentinel is that ER, for software."
        )
    )

    e.append(
        callout(
            "good",
            "Why the simulated environment is a feature, not a shortcut",
            "The &ldquo;production system&rdquo; Sentinel watches is faked in code. That is "
            "<b>deliberate</b>: because we control the fake world, we know the true cause "
            "of every incident — the answer key. That is what makes it possible to "
            "<i>measure</i> whether the agent was actually right, instead of just "
            "plausible. Against real infrastructure you have no answer key, so you can "
            "never score the agent. Same tool shapes as Prometheus/Loki/Kubernetes; "
            "swapping in real backends is a plumbing detail.",
        )
    )

    e.append(P("The five things it strings together", "h2"))
    e.append(
        bullets(
            [
                "<b>An agent brain</b> (LangGraph) that decides what to do next, can loop back to think more, and can pause.",
                "<b>Tools</b> it can call to look at the world (logs, metrics, deploys) and — only with permission — change it.",
                "<b>A knowledge base</b> (RAG) of runbooks it searches for relevant advice.",
                "<b>A small custom-trained model</b> (fine-tuned) that does the cheap, high-volume classification.",
                "<b>A safety layer</b> that makes destructive actions impossible without a human.",
            ]
        )
    )
    return e


def part2_flow():
    e = [P("Part 2", "kicker"), P("The Flow, Step by Step", "h1"), rule()]
    e.append(
        P(
            "Everything Sentinel does is a path through ten stages. Here is the whole "
            "journey in order. The two special moves — the <b>loop</b> and the "
            "<b>pause</b> — are what make it an agent rather than a script.",
            "body",
        )
    )

    steps = [
        (
            "1",
            "Triage",
            "A fast classifier reads the alert and tags it: how severe (P1–P4), what kind of failure, which service, does a human need paging now? This is the job the fine-tuned model does.",
        ),
        (
            "2",
            "Retrieve",
            "Search the runbook library for advice relevant to this alert (this is RAG).",
        ),
        (
            "3",
            "Investigate",
            "An agent with READ-ONLY tools pulls logs, metrics and deploy history. It decides what to look at — it is not following a fixed script.",
        ),
        (
            "4",
            "Synthesize",
            "Turn the gathered evidence into a single root-cause hypothesis, as structured data.",
        ),
        (
            "5",
            "Critique",
            "A second agent attacks that hypothesis: is every claim backed by evidence? Could the blamed service be a victim? Correlation or causation?",
        ),
        (
            "↺",
            "Loop",
            "If the critic is unconvinced, go BACK to Investigate with specific new questions. Capped at 4 rounds — a confused agent doesn't improve by spinning.",
        ),
        (
            "6",
            "Plan",
            "Once the hypothesis holds, write a concrete remediation plan: which action, on which target, why, how risky, how to undo it.",
        ),
        (
            "⏸",
            "PAUSE",
            "The graph HALTS here. A human sees the plan and clicks approve or reject. The agent cannot proceed on its own. This is the safety gate.",
        ),
        (
            "7",
            "Execute",
            "Only after approval: run the approved steps exactly, no model improvising.",
        ),
        (
            "8",
            "Verify",
            "Check the metrics actually recovered. A service that was merely restarted but will fail again is NOT resolved.",
        ),
        (
            "9",
            "Post-mortem",
            "Write a blameless write-up and save the lesson to long-term memory, so the next similar incident starts smarter.",
        ),
    ]
    data = []
    for n, title, desc in steps:
        special = n in ("↺", "⏸")
        badge_bg = AMBER if n == "⏸" else (INDIGO if not special else SLATE)
        badge = Table(
            [
                [
                    Paragraph(
                        f"<b>{n}</b>",
                        style(
                            "bn",
                            fontName="Helvetica-Bold",
                            fontSize=12,
                            textColor=colors.white,
                            alignment=TA_CENTER,
                        ),
                    )
                ]
            ],
            colWidths=[0.9 * cm],
            rowHeights=[0.9 * cm],
        )
        badge.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), badge_bg),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]
            )
        )
        body = Paragraph(f"<b>{title}</b> &nbsp; {desc}", S["small"])
        data.append([badge, body])
    t = Table(data, colWidths=[1.1 * cm, 15.3 * cm])
    t.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
            ]
        )
    )
    e.append(t)

    e.append(Spacer(1, 6))
    e.append(
        callout(
            "warn",
            "The two moves that make it &ldquo;agentic&rdquo;",
            "<b>The loop (↺)</b> — the system can decide it hasn&rsquo;t thought hard "
            "enough and go gather more evidence. A normal program runs top to bottom; an "
            "agent can revisit. <b>The pause (⏸)</b> — the system can stop mid-run, wait "
            "for a human, and resume later, even on a different machine, because its state "
            "is saved to a database.",
        )
    )
    return e


def part3_concepts():
    e = [P("Part 3", "kicker"), P("Every Concept, From Scratch", "h1"), rule()]
    e.append(
        P(
            "Twelve ideas. Each one follows the same shape: <b>what it is</b> in plain "
            "words, an <b>analogy</b>, <b>why it matters</b>, <b>what we built</b>, and "
            "<b>the gotcha</b> — the non-obvious thing that trips people up and that you "
            "can mention to sound like you actually did it.",
            "body",
        )
    )
    e.append(Spacer(1, 4))

    # 1 Tool calling
    e.append(
        KeepTogether(
            [
                concept_card(
                    "1", "Tool calling / the agentic loop", "the beating heart of every agent"
                ),
                Spacer(1, 6),
                P(
                    "<b>What it is.</b> You hand the model a menu of tools, each with a "
                    "description. Instead of answering from memory, the model replies "
                    "&ldquo;call <font face='Courier'>query_logs</font> with these "
                    "arguments.&rdquo; Your code runs that, hands back the result, and the model "
                    "decides the next move. Repeat until it stops asking."
                ),
                P(
                    "<b>Analogy.</b> A detective who can&rsquo;t solve the case from the armchair. "
                    "They send for the autopsy report, read it, then decide to interview a "
                    "witness based on what it said. Each result shapes the next request."
                ),
                P(
                    "<b>Why it matters.</b> This is the exact line between a chatbot and an "
                    "agent. A chatbot talks; an agent <i>acts</i>. The model doesn&rsquo;t know "
                    "your logs — it decides to go and look."
                ),
                P(
                    "<b>What we built.</b> We wrote the loop by hand rather than importing a "
                    "ready-made one, because the loop <i>is</i> the concept. It runs several "
                    "tool calls at once when they&rsquo;re independent, times each one out, and "
                    "feeds tool <i>errors</i> back to the model as observations — a good agent "
                    "recovers from a bad argument by trying again."
                ),
                callout(
                    "warn",
                    "The gotcha",
                    "Tool descriptions are part of the prompt. The model picks tools by reading "
                    "those docstrings, so ours say not just <i>what</i> a tool does but "
                    "<i>when to reach for it</i> and <i>how to read the result</i>. Vague tool "
                    "descriptions are the #1 reason agents call the wrong tool.",
                ),
            ]
        )
    )
    e.append(Spacer(1, 10))

    # 2 LangGraph
    e.append(
        KeepTogether(
            [
                concept_card(
                    "2", "LangGraph — state machines for agents", "the orchestration layer"
                ),
                Spacer(1, 6),
                P(
                    "<b>What it is.</b> Instead of one big loop, you model the agent as a "
                    "<b>graph</b>: boxes (nodes) are steps, arrows (edges) are transitions, and "
                    "a shared &ldquo;state&rdquo; object flows through."
                ),
                P(
                    "<b>Analogy.</b> A board game. Each square does something; some squares say "
                    "&ldquo;go back three spaces&rdquo; (a loop); one square says &ldquo;wait "
                    "for the other player&rdquo; (the pause). The rules are printed on the board, "
                    "not hidden in your head."
                ),
                P(
                    "<b>Why not just a loop?</b> Real workflows branch, cycle, and pause. A "
                    "plain loop buries those in tangled if-statements. In a graph they&rsquo;re "
                    "<i>declared</i> — you can see the whole control flow on one screen."
                ),
            ]
        )
    )
    e.append(P("Four sub-ideas worth naming separately:", "h3"))
    e.append(
        kv_table(
            [
                (
                    "Typed state + reducers",
                    "State is a typed dictionary where each field has a merge rule. Evidence is set to <i>add</i>, so investigation rounds accumulate instead of overwriting. THE classic bug: a missing merge rule silently throws away half your data — no error, just wrong results.",
                ),
                (
                    "Conditional edges",
                    "After the critic runs, a routing function chooses the next node: loop back, or move on.",
                ),
                (
                    "Cycles",
                    "investigate → synthesize → critique → investigate. The reflection loop. Bounded, because iterating doesn&rsquo;t cure confusion.",
                ),
                (
                    "Checkpointing + interrupts",
                    "The whole state saves to Postgres at each step, so the graph can pause before the approval node, the process can exit, and someone resumes tomorrow from another machine.",
                ),
            ],
            col0=4.2,
            col1=12.0,
        )
    )
    e.append(Spacer(1, 10))

    # 3 HITL
    e.append(
        KeepTogether(
            [
                concept_card("3", "Human-in-the-loop, done properly", "the safety story"),
                Spacer(1, 6),
                P(
                    "<b>The naive version.</b> Ask the model &ldquo;should I get approval?&rdquo; "
                    "and trust its answer. That&rsquo;s theatre — the model can just say no."
                ),
                P("<b>What we built — three independent walls:</b>"),
                bullets(
                    [
                        "<b>The interrupt.</b> The graph structurally halts before the approval node. It cannot pass.",
                        "<b>The approval flag lives outside the model&rsquo;s reach</b> — it&rsquo;s set in program memory by the graph AFTER a human decides. There is no prompt field or tool argument the model could use to set it.",
                        "<b>The investigator gets a read-only tool list</b> — it literally does not have a destructive tool to call, whatever it concludes.",
                    ]
                ),
                callout(
                    "bad",
                    "The line to remember",
                    "<b>Confidence is not authorization.</b> A model being sure it&rsquo;s right "
                    "does not make it right, and certainty is never permission to change "
                    "production. Safety is structural — the agent can propose a rollback, but it "
                    "has no mechanism to grant itself the right to run one.",
                ),
            ]
        )
    )
    e.append(Spacer(1, 10))

    # 4 structured output
    e.append(
        KeepTogether(
            [
                concept_card("4", "Structured output", "making models return data, not prose"),
                Spacer(1, 6),
                P(
                    "<b>What it is.</b> Forcing the model&rsquo;s answer into a strict shape (a "
                    "schema) — specific fields with specific types — instead of free-form text."
                ),
                P(
                    "<b>Analogy.</b> A form with labelled boxes versus a blank sheet. If a "
                    "colleague fills in boxes, the next person can process it. If they write a "
                    "paragraph, someone has to re-read and interpret it every time."
                ),
                P(
                    "<b>Why it matters.</b> Free text between agent steps is where these systems "
                    "rot. If step A returns prose, step B has to re-parse it, and a small "
                    "wording change silently breaks a later branch."
                ),
                P(
                    "<b>What we built.</b> Small and free models often break strict JSON mode — "
                    "markdown fences, trailing commas, prose wrapped around the answer. So our "
                    "helper escalates: ask with the schema → salvage the JSON out of whatever "
                    "came back → <b>show the model its own validation errors and ask again</b> → "
                    "fall back to a safe default."
                ),
                callout(
                    "info",
                    "The gotcha",
                    "Step three is the clever one. Telling the model &ldquo;field &lsquo;severity&rsquo; "
                    "must be one of P1&ndash;P4&rdquo; is a far better repair instruction than "
                    "&ldquo;try again.&rdquo; You feed the machine its exact mistake.",
                ),
            ]
        )
    )
    e.append(Spacer(1, 10))

    # 5 RAG
    e.append(
        KeepTogether(
            [
                concept_card(
                    "5", "RAG &amp; hybrid retrieval", "giving the model a library to consult"
                ),
                Spacer(1, 6),
                P(
                    "<b>What it is.</b> RAG = Retrieval-Augmented Generation. Before the model "
                    "answers, fetch the most relevant documents and paste them into the prompt. "
                    "The model &ldquo;reads the manual&rdquo; instead of relying on memory."
                ),
                P(
                    "<b>Analogy.</b> An open-book exam. You don&rsquo;t memorise the textbook; "
                    "you look up the right page at the right moment."
                ),
                P("<b>The key upgrade — using TWO search methods together:</b>"),
                bullets(
                    [
                        "<b>Vector search</b> understands meaning. Ask for &ldquo;exit code 137&rdquo; and it finds a section titled &ldquo;memory exhaustion&rdquo; even with no shared words.",
                        "<b>Keyword search (BM25)</b> nails rare exact strings — <font face='Courier'>x509</font>, <font face='Courier'>HikariPool-1</font>, <font face='Courier'>MISCONF</font>. Meaning-based search BLURS these, because they&rsquo;re near-unique tokens with little semantic content.",
                    ]
                ),
                P(
                    "We run both and fuse the rankings (a method called Reciprocal Rank Fusion) "
                    "rather than trying to average their scores — the two scoring systems "
                    "aren&rsquo;t on the same scale, and forcing them onto one requires tuning "
                    "that silently goes stale."
                ),
                callout(
                    "info",
                    "Two smaller tricks",
                    "<b>Chunking on headings</b>, not fixed character counts — a runbook section "
                    "is already one coherent unit of advice. <b>Contextual chunks</b> — each "
                    "piece carries its heading path (&ldquo;Database errors &gt; Fix&rdquo;), "
                    "which makes its meaning-fingerprint far more specific.",
                ),
            ]
        )
    )
    e.append(Spacer(1, 10))

    # 6 critic
    e.append(
        KeepTogether(
            [
                concept_card(
                    "6", "The critic / multi-agent pattern", "an agent that attacks its own work"
                ),
                Spacer(1, 6),
                P(
                    "<b>What it is.</b> A second agent whose only job is to find what&rsquo;s "
                    "<i>wrong</i> with the first agent&rsquo;s conclusion."
                ),
                P(
                    "<b>Analogy.</b> A newspaper. One journalist writes the story; a sceptical "
                    "editor tears it apart before it prints. The editor&rsquo;s job isn&rsquo;t "
                    "to say &ldquo;nice article&rdquo; — it&rsquo;s to catch what&rsquo;s "
                    "unsupported."
                ),
                P(
                    "<b>The crucial prompt choice.</b> We ask the critic <i>what is wrong</i> "
                    "with the hypothesis, not <i>whether it&rsquo;s reasonable</i>. A model "
                    "asked &ldquo;is this good?&rdquo; says yes. It&rsquo;s told to check: is "
                    "every claim tied to a real observation? Is this correlation or causation? "
                    "Could the blamed service be a <i>victim</i> of the real cause?"
                ),
                callout(
                    "good",
                    "Why it matters",
                    "Acting on a wrong diagnosis during an outage makes the outage longer. In "
                    "real runs the critic regularly rejected the first hypothesis and sent the "
                    "agent back with <i>specific, answerable questions</i> — not a vague "
                    "&ldquo;look harder.&rdquo;",
                ),
            ]
        )
    )
    e.append(Spacer(1, 10))

    # 7 MCP
    e.append(
        KeepTogether(
            [
                concept_card(
                    "7", "MCP — the Model Context Protocol", "a universal plug for AI tools"
                ),
                Spacer(1, 6),
                P(
                    "<b>What it is.</b> An open standard for exposing tools and data to any AI "
                    "system. The common phrase: &ldquo;USB-C for AI tools.&rdquo;"
                ),
                P(
                    "<b>Analogy.</b> Before USB, every device had its own connector. USB meant "
                    "any device worked with any computer. MCP is that standard plug — expose "
                    "your tools once and any MCP-speaking AI (Claude Desktop, another team&rsquo;s "
                    "agent) can use them without importing your code."
                ),
                P(
                    "<b>What we built — both ends.</b> A <b>server</b> that publishes our tools "
                    "using all three MCP building blocks: tools (the model calls them), resources "
                    "(read-only context addressed by a URL, like "
                    "<font face='Courier'>incident://current/alert</font>), and prompts "
                    "(reusable templates). And a <b>client</b> that loads other people&rsquo;s "
                    "MCP servers from a config file."
                ),
                callout(
                    "bad",
                    "The security decision",
                    "Over MCP we expose remediation only as <font face='Courier'>propose_"
                    "remediation</font>, which records a suggestion and runs nothing. An outside "
                    "client can&rsquo;t prove a human approved anything, so actual execution "
                    "stays locked inside our graph behind the approval gate.",
                ),
            ]
        )
    )
    e.append(Spacer(1, 10))

    # 8 fine-tuning
    e.append(
        KeepTogether(
            [
                concept_card("8", "Fine-tuning with QLoRA", "training a small model for one job"),
                Spacer(1, 6),
                P(
                    "<b>What it is.</b> Taking a small general model and specialising it for one "
                    "narrow task by showing it many examples."
                ),
                P("<b>The two acronyms, decoded:</b>"),
                bullets(
                    [
                        "<b>LoRA</b> — instead of retraining the whole (huge) model, freeze it and train tiny &ldquo;adapter&rdquo; patches — about 1% of the parameters. Cheap and fast.",
                        "<b>QLoRA</b> — the Q is &ldquo;quantised.&rdquo; Load the frozen model in 4-bit (low precision) to shrink it, and train the adapters on top. This is what let a 1.5-billion-parameter model train on a free Colab GPU.",
                    ]
                ),
                P(
                    "<b>Analogy.</b> Hiring a smart generalist and giving them one week of "
                    "training on your company&rsquo;s specific ticket system. You don&rsquo;t "
                    "re-educate them from birth; you add a thin layer of job-specific skill."
                ),
                P(
                    "<b>When it&rsquo;s the right tool.</b> Narrow, high-volume, fixed-output "
                    "jobs. Alert triage is exactly that: it runs on every alert, outputs a fixed "
                    "form, and needs pattern-recognition, not deep reasoning."
                ),
                callout(
                    "info",
                    "The choices that mattered",
                    "<b>Train on the answer only</b>, not the alert text — otherwise the model "
                    "wastes its effort learning to echo input it&rsquo;s always given. "
                    "<b>Adapters on attention AND the feed-forward layers</b> — attention-only "
                    "adapters learn the task but keep drifting on output format.",
                ),
            ]
        )
    )
    e.append(Spacer(1, 10))

    # 9 evaluation
    e.append(
        KeepTogether(
            [
                concept_card(
                    "9", "Evaluation — the part most projects skip", "proving it works, honestly"
                ),
                Spacer(1, 6),
                P(
                    "<b>What it is.</b> Measuring the model against a held-out answer key rather "
                    "than eyeballing a few outputs and calling it good."
                ),
                P("<b>The three levels — and the progression is the whole point:</b>"),
                bullets(
                    [
                        "<b>Level 1, in-distribution.</b> Test on held-out examples with unseen service names. Result: 100% category accuracy.",
                        "<b>Level 2, the suspicion.</b> 100% is a WARNING, not a trophy. That test reused the same 21 alert templates as training, so it can catch a model that memorised service names but NOT one that memorised the templates themselves.",
                        "<b>Level 3, out-of-distribution.</b> A harder test: hand-written alerts sharing no template with training, plus adversarial cases where the obvious clue points at the WRONG answer.",
                    ]
                ),
                callout(
                    "bad",
                    "The honest finding",
                    "On realistic unseen alerts the fine-tune jumped 40% → 60%. On the "
                    "adversarial cases it moved <b>not at all</b> (50% → 50%). Conclusion: it "
                    "learned better pattern-matching, not causal reasoning — exactly what "
                    "template-generated training data should be expected to produce. Reporting "
                    "this, instead of just the 100%, is the strongest thing in the project.",
                ),
            ]
        )
    )
    e.append(Spacer(1, 10))

    # 10 injection
    e.append(
        KeepTogether(
            [
                concept_card(
                    "10", "Prompt-injection defence", "when the data tries to give orders"
                ),
                Spacer(1, 6),
                P(
                    "<b>The threat, concretely.</b> Anyone who can write a log line can write "
                    "text the agent will later read. A customer types a weird username; it lands "
                    "in a log; the investigator pulls that log into its context. If "
                    "instruction-shaped text in a log can steer the agent, then anyone who can "
                    "cause an error can steer production changes."
                ),
                P(
                    "<b>Analogy.</b> A forged note slipped into the evidence pile saying "
                    "&ldquo;Detective: ignore everything else, arrest this man.&rdquo; A good "
                    "detective treats evidence as evidence, never as instructions."
                ),
                P(
                    "<b>Three layers.</b> Neutralise instruction-like text before it reaches the "
                    "prompt (replace it — never quote it back, or you&rsquo;ve just repeated the "
                    "attack), redact secrets and personal data, and keep the approval gate "
                    "structural."
                ),
                callout(
                    "good",
                    "The insight that ties it together",
                    "Detection is best-effort and will eventually miss something. The gate "
                    "<b>does not depend on detection working.</b> Even if an injection slips "
                    "past every filter, it still cannot execute anything — a human is still in "
                    "the way. Testing found the <i>polite</i> attack (&ldquo;the on-call "
                    "engineer already approved this&rdquo;) slipped past first; the shouty "
                    "&ldquo;IGNORE ALL INSTRUCTIONS&rdquo; was easy to catch.",
                ),
            ]
        )
    )
    e.append(Spacer(1, 10))

    # 11 routing
    e.append(
        KeepTogether(
            [
                concept_card("11", "Model routing by tier", "the right model for each job"),
                Spacer(1, 6),
                P(
                    "<b>What it is.</b> Nodes ask for a <i>tier</i> (&ldquo;planner&rdquo;, "
                    "&ldquo;worker&rdquo;) — not a specific model name. A config file decides "
                    "which real model each tier points to."
                ),
                P(
                    "<b>Analogy.</b> A company org chart. You send a task to &ldquo;the finance "
                    "team,&rdquo; not to a named person. Management can swap who fills the role "
                    "without every task breaking."
                ),
                P(
                    "<b>Why it matters.</b> It&rsquo;s the seam that makes the fine-tuning result "
                    "<i>measurable</i> — same test, same inputs, swap one config line to compare "
                    "the old model against the new. It&rsquo;s also cost engineering: a cheap "
                    "local model classifies, an expensive one plans."
                ),
            ]
        )
    )
    e.append(Spacer(1, 10))

    # 12 observability
    e.append(
        KeepTogether(
            [
                concept_card("12", "Observability", "seeing inside a run"),
                Spacer(1, 6),
                P(
                    "<b>What it is.</b> Recording a detailed trace of everything the agent did — "
                    "every model call, every tool call, with prompts, timings and token counts — "
                    "so you can inspect it afterwards (we use LangSmith)."
                ),
                P(
                    "<b>Analogy.</b> A flight recorder. When something goes wrong you don&rsquo;t "
                    "want a one-line summary; you want the full timeline of every input and "
                    "decision."
                ),
                P(
                    "<b>Why traces beat logs.</b> When the critic rejects a hypothesis three "
                    "times and the run burns 130,000 tokens, the question is <i>which</i> node, "
                    "on <i>which</i> loop, with <i>what</i> prompt. That&rsquo;s a question about "
                    "a tree, not a flat log line."
                ),
                callout(
                    "warn",
                    "A real trap we hit",
                    "The settings loader reads the config file into an object, but the tracing "
                    "library reads raw environment variables — two different places. Flipping "
                    "the switch in the config file looked enabled, reported itself enabled, and "
                    "traced nothing. The fix verifies the tracer actually started, instead of "
                    "trusting the flag.",
                ),
            ]
        )
    )
    return e


def part4_results():
    e = [P("Part 4", "kicker"), P("The Results — and the Honest Story", "h1"), rule()]

    e.append(P("The fine-tuned model vs the baseline", "h2"))
    e.append(
        P(
            "Both models were given the exact same held-out alerts. &ldquo;Baseline&rdquo; "
            "is a general 3-billion-parameter model just prompted with instructions; "
            "&ldquo;fine-tuned&rdquo; is our 1.5-billion model trained for this one job — "
            "half the size, better numbers, half the latency.",
            "body",
        )
    )
    e.append(
        compare_table(
            ["Metric", "Baseline", "Fine-tuned"],
            [
                ["Valid JSON (parseable output)", "often failed", "100%"],
                ["Severity accuracy", "37.5%", "97.5%"],
                ["Category accuracy", "25.0%", "100%"],
                ["Critical misses (P1 seen as minor)", "10.0%", "0%"],
                ["Latency (p50)", "2980 ms", "1622 ms"],
            ],
        )
    )
    e.append(Spacer(1, 6))
    e.append(
        callout(
            "info",
            "Read the &ldquo;valid JSON&rdquo; row first",
            "The baseline&rsquo;s category accuracy was low largely because it "
            "couldn&rsquo;t reliably produce parseable output at all — and a classifier "
            "whose answer won&rsquo;t parse is a broken classifier, no matter what it "
            "&lsquo;knew.&rsquo; Getting a small model to 100% clean structured output is "
            "the quiet, real win.",
        )
    )

    e.append(P("The test that told the truth", "h2"))
    e.append(
        P(
            "A 100% score should make you suspicious, not proud. That test reused the same "
            "21 alert templates as the training data — so it can&rsquo;t tell "
            "&ldquo;learned the task&rdquo; apart from &ldquo;memorised the templates.&rdquo; "
            "So we built a harder test: brand-new alerts sharing no template, plus "
            "adversarial ones where the obvious surface clue points at the wrong answer.",
            "body",
        )
    )
    e.append(
        compare_table(
            ["Generalization test", "Baseline", "Fine-tuned"],
            [
                ["Realistic unseen alerts", "40%", "60%  (+20)"],
                ["Adversarial (traps)", "50%", "50%  (+0)"],
            ],
        )
    )
    e.append(Spacer(1, 6))
    e.append(
        callout(
            "bad",
            "This is the headline finding — lead with it, don&rsquo;t hide it",
            "Big gains on realistic new alerts; <b>zero movement</b> on cases built to "
            "fool surface-level pattern matching. That means the model learned to match "
            "patterns better, but did <i>not</i> learn the underlying cause-and-effect "
            "reasoning — precisely what you&rsquo;d expect from template-generated data. "
            "&ldquo;I got 100%, found it suspicious, built a harder test, and proved the "
            "model had memorised patterns&rdquo; shows scientific judgment. That sentence "
            "is worth more than the 100% ever was.",
        )
    )

    e.append(P("Why the adversarial score didn&rsquo;t move — and how to fix it", "h2"))
    e.append(
        P(
            "The training data had no <i>causal</i> variety: every "
            "&ldquo;resource-exhaustion&rdquo; example literally contained &ldquo;exit "
            "code 137,&rdquo; so the model learned the <i>string</i>, not the "
            "<i>concept</i>. The fix is a data generator that varies the mechanism (same "
            "cause, different symptoms; same symptoms, different cause) rather than just "
            "the surface words. Naming this yourself, before anyone asks, is more "
            "impressive than the perfect score.",
            "body",
        )
    )
    return e


def part5_explain():
    e = [P("Part 5", "kicker"), P("How to Explain It", "h1"), rule()]

    e.append(P("Three résumé bullets", "h2"))
    e.append(
        callout(
            "good",
            "Bullet 1 — the system",
            "Built <b>Sentinel</b>, an autonomous incident-response agent using LangGraph "
            "and LangChain — a 10-node state machine with a cyclic self-critique loop, "
            "Postgres checkpointing, and a human-in-the-loop interrupt gating all "
            "destructive actions. Hybrid RAG (BM25 + pgvector, fused via reciprocal rank "
            "fusion) grounds investigations in a runbook corpus, and a custom MCP server "
            "exposes the toolset to any protocol-compatible client.",
        )
    )
    e.append(
        callout(
            "good",
            "Bullet 2 — the fine-tune",
            "Fine-tuned <b>Qwen2.5-1.5B with QLoRA</b> for alert triage, lifting category "
            "accuracy from 25% → 100% and severity accuracy 37.5% → 97.5% while halving "
            "p50 latency (2980ms → 1622ms) on a model half the baseline&rsquo;s size; "
            "eliminated critical-severity underestimates entirely (10% → 0%).",
        )
    )
    e.append(
        callout(
            "good",
            "Bullet 3 — the judgment (your strongest)",
            "Designed a <b>three-tier evaluation harness that exposed the model&rsquo;s "
            "limits rather than hiding them</b> — an out-of-distribution suite with "
            "adversarial cases showed the fine-tune gained +20pp on unseen realistic "
            "alerts but 0 on cases engineered to defeat surface-feature matching, proving "
            "it learned pattern-matching, not causal reasoning. Also built a 31-check "
            "red-team suite for prompt injection, PII redaction, and privilege escalation.",
        )
    )

    e.append(P("The tech stack, grouped", "h2"))
    e.append(
        kv_table(
            [
                ("Agent &amp; orchestration", "LangGraph · LangChain · MCP (FastMCP) · Pydantic"),
                (
                    "Models &amp; training",
                    "Qwen2.5-1.5B (QLoRA) · Ollama · OpenRouter · PEFT · TRL · bitsandbytes · llama.cpp/GGUF",
                ),
                (
                    "Data &amp; retrieval",
                    "PostgreSQL + pgvector · BM25 (rank-bm25) · Reciprocal Rank Fusion",
                ),
                ("Backend", "Python 3.11 · FastAPI · SSE streaming · asyncio · httpx · psycopg3"),
                ("Frontend", "Streamlit (5-page console)"),
                (
                    "Observability &amp; eval",
                    "LangSmith · structlog · custom eval harness (macro-F1, confusion matrices)",
                ),
                (
                    "Infra &amp; tooling",
                    "Docker (multi-stage, non-root) · Compose · GitHub Actions CI · uv · pytest (83 tests) · ruff",
                ),
            ],
            col0=4.6,
            col1=11.6,
        )
    )

    e.append(PageBreak())
    e.append(P("Interview Q&amp;A — the questions you WILL get", "h2"))

    qa = [
        (
            "&ldquo;Walk me through what happens when an alert fires.&rdquo;",
            "Triage classifies it; retrieve pulls relevant runbooks; an investigator with "
            "read-only tools gathers logs, metrics and deploys; synthesize forms a "
            "hypothesis; a critic attacks it and can loop back for more evidence; once "
            "solid, plan writes remediation steps; the graph PAUSES for human approval; "
            "on approval it executes, verifies recovery, and writes a post-mortem to memory.",
        ),
        (
            "&ldquo;Isn&rsquo;t the environment fake? Does it count?&rdquo;",
            "Fake on purpose. A simulated world with a known ground truth is the only way "
            "to actually SCORE whether the agent found the right cause. Real infrastructure "
            "has no answer key. The tool interfaces match real ones (Prometheus, logs, k8s), "
            "so swapping in real backends is plumbing, not redesign.",
        ),
        (
            "&ldquo;How do you stop the agent doing something destructive?&rdquo;",
            "Three structural walls, not a polite request: the graph physically halts "
            "before execution; the approval flag is set outside the model&rsquo;s reach; "
            "and the investigator is handed a read-only tool list so it has no destructive "
            "tool to call. Confidence is never authorization.",
        ),
        (
            "&ldquo;Your model hits 100% — isn&rsquo;t that overfitting?&rdquo;",
            "Yes, and I caught it. The held-out test reused training templates, so I built "
            "an out-of-distribution suite with adversarial cases. The fine-tune gained on "
            "realistic new alerts but zero on the traps — it learned patterns, not "
            "causal reasoning. The fix is training data with real causal variety.",
        ),
        (
            "&ldquo;Why fine-tune instead of just prompting a bigger model?&rdquo;",
            "Triage is narrow, high-volume, and fixed-output — the textbook case for "
            "fine-tuning. A 1.5B specialised model beat a 3B general one on accuracy at "
            "half the latency, and runs locally for free. You wouldn&rsquo;t fine-tune for "
            "the open-ended planning step — that stays on a bigger model.",
        ),
        (
            "&ldquo;What was the hardest bug?&rdquo;",
            "A dtype mismatch during fine-tuning: the model loaded in bf16 while training "
            "ran fp16, because a library renamed a parameter and silently ignored the old "
            "one. It only surfaced minutes into a training run. The fix picks precision "
            "from the actual hardware and ASSERTS the dtype applied instead of trusting it.",
        ),
    ]
    for q, a in qa:
        e.append(
            KeepTogether(
                [
                    Paragraph(
                        f"Q&nbsp;&nbsp;{q}",
                        style(
                            "q",
                            fontName="Helvetica-Bold",
                            fontSize=10.5,
                            leading=14,
                            textColor=INDIGO,
                            spaceBefore=6,
                            spaceAfter=2,
                        ),
                    ),
                    Paragraph(f"A&nbsp;&nbsp;{a}", S["body"]),
                ]
            )
        )
    return e


def part6_glossary():
    e = [P("Part 6", "kicker"), P("Glossary — Every Term in One Place", "h1"), rule()]
    terms = [
        (
            "Agent",
            "An AI system that doesn&rsquo;t just talk but takes actions — calling tools, deciding its own next step, looping until done.",
        ),
        (
            "Agentic loop",
            "The core cycle: model requests a tool → you run it → feed the result back → repeat until the model stops asking.",
        ),
        (
            "Tool calling",
            "The mechanism by which a model asks your code to run a named function with arguments.",
        ),
        (
            "LangChain",
            "A library of building blocks for LLM apps — models, tools, prompts, retrievers — with one common interface.",
        ),
        (
            "LangGraph",
            "A library for building agents as graphs (nodes + edges + shared state), supporting branches, loops and pauses.",
        ),
        (
            "Node / Edge",
            "A node is one step (a box); an edge is a transition (an arrow) between steps.",
        ),
        (
            "State",
            "The shared data object that flows through the graph and accumulates results as it goes.",
        ),
        (
            "Reducer",
            "A merge rule on a state field. &ldquo;Add&rdquo; means new values append; without one, they overwrite.",
        ),
        (
            "Checkpointing",
            "Saving the graph&rsquo;s state to a database at each step, so a run can pause and resume — even after a restart.",
        ),
        (
            "Interrupt",
            "A configured point where the graph halts and waits (here: for human approval) before continuing.",
        ),
        (
            "Human-in-the-loop (HITL)",
            "A design where a person must approve certain steps before the system proceeds.",
        ),
        (
            "Structured output",
            "Forcing model output into a strict schema (typed fields) rather than free-form prose.",
        ),
        (
            "Schema / Pydantic",
            "A definition of the exact shape data must take. Pydantic is the Python library that enforces it.",
        ),
        (
            "RAG",
            "Retrieval-Augmented Generation — fetch relevant documents and add them to the prompt before the model answers.",
        ),
        (
            "Embedding / Vector",
            "A list of numbers representing a text&rsquo;s meaning, so similar meanings sit close together.",
        ),
        ("Vector search", "Finding documents by meaning-similarity rather than exact words."),
        (
            "BM25",
            "A classic keyword-ranking algorithm — great at exact rare tokens that embeddings blur.",
        ),
        (
            "Hybrid search / RRF",
            "Combining keyword and vector search. Reciprocal Rank Fusion merges their rankings without rescaling.",
        ),
        (
            "pgvector",
            "A PostgreSQL extension that stores embeddings and does vector search inside the database.",
        ),
        (
            "Chunking",
            "Splitting documents into pieces for retrieval. We split on headings, keeping each piece coherent.",
        ),
        (
            "Critic / reflection",
            "A second agent that attacks the first&rsquo;s output; a loop back for more work if it&rsquo;s weak.",
        ),
        (
            "Multi-agent",
            "Multiple specialised agents (investigator, critic, planner) each with their own prompt and tools.",
        ),
        (
            "MCP",
            "Model Context Protocol — an open standard for exposing tools/data to any AI system. &ldquo;USB-C for AI tools.&rdquo;",
        ),
        (
            "MCP server / client",
            "A server publishes tools; a client consumes tools published by others.",
        ),
        (
            "Fine-tuning",
            "Specialising a model for one task by training it on task-specific examples.",
        ),
        (
            "LoRA",
            "Low-Rank Adaptation — freeze the base model, train tiny adapter matrices (~1% of parameters).",
        ),
        (
            "QLoRA",
            "LoRA with the base model quantised to 4-bit, so a bigger model fits on a small GPU.",
        ),
        ("Quantisation", "Storing model weights in lower precision (e.g. 4-bit) to save memory."),
        ("Adapter", "The small set of trained weights LoRA adds on top of the frozen base model."),
        (
            "GGUF / Ollama",
            "GGUF is a portable model file format; Ollama is a tool that runs such models locally.",
        ),
        (
            "Baseline",
            "The comparison point — here, a general model just prompted, before fine-tuning.",
        ),
        (
            "Held-out / OOD",
            "Held-out data isn&rsquo;t used in training. OOD (out-of-distribution) is deliberately unlike training data.",
        ),
        (
            "Macro-F1",
            "An accuracy metric that treats every class equally, so rare classes still count.",
        ),
        (
            "Confusion matrix",
            "A grid showing what got predicted vs what was true — the diagonal is correct.",
        ),
        (
            "Prompt injection",
            "An attack where malicious instructions hidden in data try to hijack the model.",
        ),
        ("Blast radius", "How much damage an action can do — used to decide what needs approval."),
        (
            "Model routing / tier",
            "Sending each task to a role (&ldquo;planner&rdquo;), with config deciding the actual model.",
        ),
        (
            "Observability / tracing",
            "Recording a full trace of a run (every call, prompt, token) for later inspection.",
        ),
        (
            "Token",
            "The unit models read and write text in — roughly a word-piece; usage is measured in tokens.",
        ),
        (
            "Latency (p50/p95)",
            "Response time. p50 = median; p95 = the slow tail (95% are faster than this).",
        ),
        ("SSE", "Server-Sent Events — a way to stream updates to a browser as they happen."),
    ]
    data = []
    for term, desc in terms:
        data.append([Paragraph(f"<b>{term}</b>", S["small"]), Paragraph(desc, S["small"])])
    t = Table(data, colWidths=[4.2 * cm, 12.0 * cm])
    t.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    e.append(t)
    e.append(Spacer(1, 10))
    e.append(
        callout(
            "info",
            "One last thing to internalise",
            "The strongest signal in this whole project is not any single technology — "
            "it&rsquo;s that you measured your own work honestly, found where it fell "
            "short, and could explain exactly why. Anyone can wire up a demo. Knowing what "
            "your demo does NOT prove is the senior skill.",
        )
    )
    return e


if __name__ == "__main__":
    build()

"""Agent registry.

Mirrors the OpenCode model: each agent is a specialist with its own
description (which the model reads to decide whether to delegate), its own
system prompt, its own tool subset, and its own budget. The model picks the
agent that fits the request.
"""

from tools import TOOLS as _BASE_TOOLS, build_tool_list

def _tool_subset(*names: str) -> list:
    """The TOOLS entries for the named tools, in the order given."""
    by_name = {t["function"]["name"]: t for t in _BASE_TOOLS}
    return [by_name[n] for n in names if n in by_name]


RESEARCH_TOOLS = ("web_search", "read_file", "write_file", "edit_file", "code",
                  "wolfram_alpha", "verify_claims", "draw_diagram")
ANALYZE_TOOLS = ("code", "wolfram_alpha", "web_search", "read_file", "write_file",
                 "edit_file", "verify_claims", "draw_diagram")
CODE_TOOLS = ("code", "read_file", "write_file", "edit_file", "web_search",
              "draw_diagram")
EXPLORE_TOOLS = ("web_search", "read_file", "code", "wolfram_alpha", "draw_diagram")

AGENTS: dict[str, dict] = {
    "research": {
        "description": (
            "Deep multi-source research that produces a sourced report. Use for "
            "questions needing many sources, a literature review, a company or "
            "market analysis, regulatory or competitive landscape research, and "
            "anything explicitly framed as 'research', 'deep dive', 'investigate' "
            "or 'do a thorough job on'."
        ),
        "label": "Deep research",
        "tools": _tool_subset(*RESEARCH_TOOLS),
        "max_steps": 100,
        "max_tokens": 8192,
        "prompt": (
            "You are a research agent producing a sourced report. Work "
            "autonomously: plan, gather evidence with tools, analyse it, and "
            "produce a thorough, well-structured report.\n"
            "Ground every figure in what a tool actually returned. If a tool did "
            "not give you a number, do not state it - flag it as unverified. If "
            "sources disagree, show the conflict rather than resolving it "
            "silently. Distinguish reported figures from your own estimates.\n"
            "Recompute every derived figure with verify_claims before you write "
            "it down. If a source's own percentage disagrees with your "
            "recomputation, trust the recomputation and flag the discrepancy.\n"
            "Date every figure and prefer primary sources - filings, official "
            "announcements, the regulator - over coverage of them.\n"
            "IMPORTANT: write your deliverable to files as you go, not only at the "
            "end. Use write_file to save each major section as its own markdown "
            "file (e.g. report_1_summary.md, report_2_findings.md) as soon as you "
            "have finished that section, and revise it later if new evidence "
            "arrives. The files are what gets delivered to the user, so they must "
            "contain the real work. Keep your final chat summary short and point "
            "at the files."
        ),
    },
    "analyze": {
        "description": (
            "Quantitative analysis of data: models, simulations, statistics, "
            "scenarios and forecasts. Use when the request needs numbers "
            "computed, a model built, a probability estimated, a dataset "
            "processed, or a decision quantified. Prefer this over 'research' "
            "when the deliverable is a calculation rather than a written survey."
        ),
        "label": "Quantitative analysis",
        "tools": _tool_subset(*ANALYZE_TOOLS),
        "max_steps": 100,
        "max_tokens": 8192,
        "prompt": (
            "You are a quantitative analyst. Build the model or run the "
            "computation, then report the result with its assumptions stated.\n"
            "Use the code tool to actually compute - run simulations, fit "
            "models, process data - rather than estimating in your head. Use "
            "wolfram_alpha for closed-form maths. Verify your numbers before "
            "reporting them, and say explicitly which inputs were assumed "
            "versus measured.\n"
            "Run every derived figure through verify_claims before you publish "
            "it, and correct any mismatch - a simulation that reports a total "
            "that does not match its parts is a bug, not a rounding issue.\n"
            "Write the model, its assumptions, and the results to files "
            "(e.g. model.md, results.md) with the code, so the user can "
            "reproduce them. Keep your chat summary to the headline numbers and "
            "the caveats."
        ),
    },
    "code": {
        "description": (
            "Write, run and verify complete programs: scripts, CLI tools, "
            "utilities, data pipelines, and debug or refactor passes over "
            "existing code. Use this whenever the deliverable is a program or a "
            "file the user will run - anything more than a one-off snippet. The "
            "agent writes the code to files, runs it, fixes what breaks, and "
            "delivers the working files. Prefer this over the inline code tool "
            "for anything substantial."
        ),
        "label": "Coding",
        "tools": _tool_subset(*CODE_TOOLS),
        "max_steps": 100,
        "max_tokens": 8192,
        "prompt": (
            "You are a senior engineer. Write working code, run it, and check "
            "the output before you report back.\n"
            "Save the finished code to files in the workspace with write_file so "
            "the user gets the actual program, and include the exact command "
            "needed to run it. If something fails, debug it rather than "
            "reporting the failure - iterate until it works or until you are "
            "genuinely blocked, and then say precisely what is blocking it."
        ),
    },
    "explore": {
        "description": (
            "Fast read-only survey of a topic or a codebase: find and summarise "
            "what exists, how things work, what the key facts are. Use when the "
            "question is 'what is here' or 'how does this work' rather than "
            "'build me a report'. Cheaper and faster than a full research job."
        ),
        "label": "Exploration",
        "tools": _tool_subset(*EXPLORE_TOOLS),
        "max_steps": 100,
        "max_tokens": 4096,
        "prompt": (
            "You are exploring a topic or a codebase to answer a question. Be "
            "efficient: a handful of targeted searches or reads, then answer. "
            "Do not pad the answer with caveats or a long report - the user "
            "asked a focused question. Cite what you found, and date anything "
            "time-sensitive so the user knows how current it is."
        ),
    },
}


def agent_names() -> list[str]:
    return list(AGENTS)


def get_agent(name: str) -> dict | None:
    return AGENTS.get(name)


def agent_catalogue() -> str:
    """Human/model readable catalogue of the available agents."""
    lines = []
    for name, spec in AGENTS.items():
        lines.append(f"- {name} ({spec['label']}): {spec['description']}")
    return "\n".join(lines)


# The full tool list the model sees, with the agent catalogue filled in.
TOOLS = build_tool_list()

# Agents must not spawn further agents.
def _strip_delegation(tools: list) -> list:
    return [t for t in tools if t["function"]["name"] != "start_agent_task"]

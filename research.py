"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import (
    FINALIZER_PATH,
    REPORT_PATH,
    SOURCES_PATH,
    VALIDATOR_PATH,
    WORKDIR,
    build_lead_agent,
)
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-",
    max 60 chars, never empty (fall back to "topic"). The topic is user input: "../../x" must not
    escape reports/."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", (topic or "").strip().lower()).strip("-")
    slug = slug[:60].strip("-")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f'Research and write a survey report on the topic: "{topic}". '
        "Follow your workflow: plan, delegate sub-questions to researcher subagents in parallel, "
        "merge their notes into sources.json, write the report body, then finalize and validate the "
        "citations until the validator prints OK."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count},
    "tokens": {"input", "output"}}."""
    tool_counts = Counter()
    subagent_calls = 0
    input_tokens = 0
    output_tokens = 0
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name")
            if name:
                tool_counts[name] += 1
                if name == "task":
                    subagent_calls += 1
        usage = getattr(message, "usage_metadata", None) or {}
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)
    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_counts),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the
    report path. A failed run (missing/empty report or bad sources.json) raises RuntimeError and
    writes NOTHING."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("report is missing or empty")
    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError("sources.json is missing or empty")

    try:
        sources = json.loads(sources_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"sources.json is invalid JSON: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("sources.json is not a non-empty JSON array")

    # Deterministic source-family normalization: the rubric maps `source` to the URL family
    # (arxiv / hf-daily / hf-search / web), so never trust the LLM's spelled-out tool names.
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url") or "")
        if url.startswith("https://arxiv.org/"):
            entry["source"] = "arxiv"
        elif url.startswith("https://huggingface.co/papers/"):
            src = str(entry.get("source") or "").lower()
            entry["source"] = "hf-daily" if "daily" in src else "hf-search"
        else:
            entry["source"] = "web"

    slug = slugify(topic)
    reports_dir.mkdir(parents=True, exist_ok=True)
    meta = summarize(messages, elapsed, model_name)
    meta["topic"] = topic
    meta["n_sources"] = len(sources)
    meta["source_families"] = sorted({s.get("source") for s in sources if isinstance(s, dict)})

    (reports_dir / f"{slug}.sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / f"{slug}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = reports_dir / f"{slug}.md"
    report_path.write_bytes(report_bytes)
    return report_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = (topic or "").strip()
    if not topic:
        print("usage: python research.py \"<topic>\"", file=sys.stderr)
        return 2

    try:
        model = make_model()
    except Exception as exc:
        print(f"FAILED: cannot configure the model: {exc}", file=sys.stderr)
        return 1

    model.streaming = True  # WDB proxy emits tool calls only when streaming
    # Reasoning models stream slowly with a very large context; raise every
    # timeout layer so a slow response cannot kill a whole topic mid-write.
    import httpx as _httpx
    try:
        model.request_timeout = 600
    except Exception:
        pass
    try:
        model.root_client.timeout = _httpx.Timeout(600.0, connect=15.0)
    except Exception:
        pass
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or "unknown"
    start = time.monotonic()
    try:
        with open_sandbox() as backend:
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })
            agent = build_lead_agent(backend, model)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000},
            )
            elapsed = time.monotonic() - start
            report_path = save_outputs(
                backend, topic, result["messages"], elapsed, model_name)
    except Exception as exc:  # sandbox is always cleaned up by open_sandbox on exit
        import traceback
        traceback.print_exc()
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"report saved: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))

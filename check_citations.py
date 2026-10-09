"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_CITATION = re.compile(r"\[(\d+)\](?!\()")


def _body_text(report_text):
    """Report text before the `## References` heading (citations in the list do not count)."""
    parts = re.split(r"(?m)^##[ \t]+References[ \t]*$", report_text, maxsplit=1)
    return parts[0] if len(parts) > 1 else ""


def _code_stripped(text):
    """Remove fenced code blocks and inline code spans so citations inside code do not count."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    return re.sub(r"`[^`\n]*`", " ", text)


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not sources:
        return ["no sources in sources.json"]

    by_n = {}
    for entry in sources:
        if not isinstance(entry, dict) or not isinstance(entry.get("n"), int):
            problems.append("source entry has an invalid (non-integer) n")
            continue
        n = entry["n"]
        if n in by_n:
            problems.append(f"duplicate source number {n}")
        by_n[n] = entry

    urls = {}
    for n, entry in by_n.items():
        url = entry.get("url")
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] url missing or not http(s)")
            continue
        if url in urls:
            problems.append(f"duplicate url {url} (sources [{urls[url]}] and [{n}])")
        urls[url] = n

    if not re.search(r"(?m)^##[ \t]+References[ \t]*$", report_text):
        problems.append("missing `## References` heading")

    body = _code_stripped(_body_text(report_text))
    cited = {int(m.group(1)) for m in _CITATION.finditer(body)}

    for n in cited:
        if n not in by_n:
            problems.append(f"[{n}] cited in body but missing from sources.json")
    for n in by_n:
        if n not in cited:
            problems.append(f"source [{n}] never cited in body")

    ref_text = re.split(r"(?m)^##[ \t]+References[ \t]*$", report_text, maxsplit=1)
    ref_lines = ref_text[1].splitlines() if len(ref_text) > 1 else []
    seen = set()
    for line in ref_lines:
        m = re.match(r"\s*\[(\d+)\]\s*", line)
        if not m:
            continue
        n = int(m.group(1))
        if n in seen:
            problems.append(f"duplicate reference line for [{n}]")
            continue
        seen.add(n)
        if n not in by_n:
            problems.append(f"reference line for [{n}] that is not a source")
            continue
        found = re.findall(r"https?://\S+", line)
        if len(found) != 1:
            problems.append(f"reference line [{n}] must contain exactly one URL")
        elif found[0].rstrip(").,") != by_n[n]["url"]:
            problems.append(f"reference line [{n}] URL does not match sources.json")

    for n in sorted(set(by_n) - seen):
        problems.append(f"source [{n}] missing its reference line")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

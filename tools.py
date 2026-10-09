"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import httpx
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

_ATOM = "http://www.w3.org/2005/Atom"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_SUMMARY_CHARS = 600
_FETCH_CHARS = 12000
_ARXIV_INTERVAL = 3.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after_seconds(response):
    """Retry-After header as seconds (integer or HTTP date), else None."""
    value = (response.headers.get("Retry-After") if response is not None else None)
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        delta = parsedate_to_datetime(value).timestamp() - time.time()
        return max(0.0, delta)
    except (TypeError, ValueError, OverflowError):
        return None


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    Delay = the error's retry_after if the server told us, else exponential backoff
    base * 2**attempt (capped at cap) plus random jitter. The last failed attempt
    re-raises without sleeping again. Non-Retryable errors propagate immediately.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(max(0.0, float(exc.retry_after)), cap)
            else:
                delay = min(base * (2 ** attempt), cap) + random.uniform(0.0, 0.5)
            time.sleep(delay)
    raise RuntimeError("with_retry: unreachable")  # pragma: no cover


def _http(method, url, *, params=None, attempts=5, base=1.0, cap=30.0):
    """One HTTP request wrapped in with_retry; retryable failures raise RetryableError."""

    def call():
        try:
            response = httpx.request(method, url, params=params, follow_redirects=True, timeout=60.0)
        except httpx.TransportError as exc:
            raise RetryableError(f"network error: {exc}") from exc
        if response.status_code in _RETRYABLE_STATUS:
            raise RetryableError(f"HTTP {response.status_code}", retry_after=_retry_after_seconds(response))
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"HTTP {exc.response.status_code}") from exc
        return response

    return with_retry(call, attempts=attempts, base=base, cap=cap)


def _compact(text, limit):
    return re.sub(r"\s+", " ", (text or "")).strip()[:limit]


def _redact_key(text):
    key = (os.getenv("EXA_API_KEY") or "").strip()
    if key:
        text = text.replace(key, "<redacted>")
    return text


def _error(exc):
    return f"ERROR: {type(exc).__name__}: {_redact_key(str(exc))}"


_last_arxiv_call = 0.0


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    terms = re.findall(r"[A-Za-z0-9-]+", query or "")
    if not terms:
        return "NO RESULTS"
    search_query = " AND ".join(f"all:{t}" for t in terms)

    global _last_arxiv_call
    wait = _ARXIV_INTERVAL - (time.monotonic() - _last_arxiv_call)
    if wait > 0:
        time.sleep(wait)
    try:
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": min(max(int(max_results), 1), 30),
            "start": 0,
        }
        body = _http("GET", ARXIV_URL, params=params, attempts=6, base=1.0, cap=60.0).text
        root = ET.fromstring(body)
        records = []
        for entry in root.findall(f"{{{_ATOM}}}entry"):
            eid = (entry.findtext(f"{{{_ATOM}}}id") or "").split("/abs/")[-1]
            eid = re.sub(r"v\d+$", "", eid)
            title = _compact(entry.findtext(f"{{{_ATOM}}}title"), 300)
            if not eid or not title:
                continue
            records.append({
                "id": eid,
                "url": f"https://arxiv.org/abs/{eid}",
                "published": (entry.findtext(f"{{{_ATOM}}}published") or "")[:10],
                "title": title,
                "summary": _compact(entry.findtext(f"{{{_ATOM}}}summary"), _SUMMARY_CHARS),
            })
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)
    finally:
        _last_arxiv_call = time.monotonic()


def _hf_record(item):
    paper = item.get("paper") or {}
    pid = paper.get("id")
    if not pid:
        return None
    summary = paper.get("ai_summary") or item.get("summary") or paper.get("summary") or ""
    return {
        "id": pid,
        "url": f"https://huggingface.co/papers/{pid}",
        "published": (paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
        "title": _compact(paper.get("title") or item.get("title"), 300),
        "summary": _compact(summary, _SUMMARY_CHARS),
        "upvotes": int(paper.get("upvotes") or 0),
        "github": paper.get("githubRepo") or "",
        "stars": int(paper.get("githubStars") or 0),
    }


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    params = {"limit": min(max(int(limit), 1), 100)}
    if date:
        params["date"] = date
    try:
        items = json.loads(_http("GET", HF_DAILY_URL, params=params).text)
    except Exception as exc:
        return _error(exc)
    records = [r for r in (_hf_record(item) for item in items) if r is not None]
    if keyword:
        kw = keyword.lower()
        records = [r for r in records if kw in (r["title"] + " " + r["summary"]).lower()]
    records.sort(key=lambda r: r["upvotes"], reverse=True)
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    if not (query or "").strip():
        return "NO RESULTS"
    try:
        items = json.loads(_http("GET", HF_SEARCH_URL, params={
            "q": query.strip(), "limit": min(max(int(limit), 1), 50)}).text)
    except Exception as exc:
        return _error(exc)
    records = [r for r in (_hf_record(item) for item in items) if r is not None]
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


def _exa_url():
    key = (os.getenv("EXA_API_KEY") or "").strip()
    return EXA_URL + (f"?exaApiKey={key}" if key else "")


def _exa_call(name, arguments, *, attempts=10, base=2.0, cap=60.0):
    """One JSON-RPC MCP call to Exa, retrying on the free tier's hidden rate-limit flag."""
    if not (os.getenv('EXA_API_KEY') or '').strip():
        # shortcut: the free tier is rate-limited on shared IPs; failing fast beats hanging for minutes
        attempts, base, cap = 2, 1.0, 3.0
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": name, "arguments": arguments}}

    def call():
        try:
            response = httpx.post(
                _exa_url(), json=payload,
                headers={"Content-Type": "application/json",
                         "Accept": "application/json, text/event-stream"},
                timeout=90.0, follow_redirects=True)
        except httpx.TransportError as exc:
            raise RetryableError(f"network error: {exc}") from exc
        if response.status_code in _RETRYABLE_STATUS:
            raise RetryableError(f"HTTP {response.status_code}", retry_after=_retry_after_seconds(response))
        response.raise_for_status()
        message = None
        for line in response.text.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data:
                continue
            try:
                parsed = json.loads(data)
            except ValueError:
                continue
            if isinstance(parsed, dict) and ("result" in parsed or "error" in parsed):
                message = parsed
        if message is None:
            raise RetryableError("no JSON-RPC message in Exa response")
        if "error" in message:
            raise RuntimeError(f"Exa error: {message['error']}")
        result = message.get("result") or {}
        meta = result.get("_meta") or {}
        if meta.get("ai.exa/rateLimited") or meta.get("rateLimited"):
            raise RetryableError("Exa free tier rate limited")
        blocks = result.get("content") or []
        return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()

    return with_retry(call, attempts=attempts, base=base, cap=cap)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    if not (query or "").strip():
        return "NO RESULTS"
    objective = (objective or f"Find authoritative pages about: {query}").strip()
    arguments = {"query": query.strip(), "objective": objective,
                 "numResults": min(max(int(num_results), 1), 10)}
    try:
        text = _exa_call("web_search_exa", arguments)
    except Exception as exc:
        return _error(exc)
    return text or "NO RESULTS"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    if not (url or "").strip():
        return "NO RESULTS"
    try:
        text = _exa_call("web_fetch_exa", {"urls": [url.strip()]})
    except Exception as exc:
        return _error(exc)
    return text[:_FETCH_CHARS] if text else "NO RESULTS"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]

if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")

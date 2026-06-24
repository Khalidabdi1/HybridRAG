"""A small, fully-offline sample dataset for the evaluation harness.

The corpus is intentionally text-only so the harness runs with nothing but
numpy (no rendered screenshots, no GPU). It is enough to exercise every metric
and to demonstrate the comparison table end to end. To evaluate the *vision*
and *hybrid* columns meaningfully, supply your own dataset with ``image_path``
documents and real encoders (``--text-model`` / ``--vision-model``).

Documents are short and lexically distinctive so the dependency-free hashing
(bag-of-tokens) text encoder can retrieve them — i.e. this measures a lexical
baseline. Swap in a sentence-transformers model to measure semantic recall.
"""

from __future__ import annotations

from .dataset import EvalDataset, EvalDocument, EvalQuery

_DOCS = [
    ("py_zipdiv", "Python ZeroDivisionError traceback when dividing by zero",
     "A ZeroDivisionError is raised in Python when code divides an integer or "
     "float by zero. The traceback points at the offending line; guard the "
     "denominator or catch the exception with a try/except block."),
    ("json_parse", "Parsing malformed JSON and handling decode errors",
     "json.loads raises a JSONDecodeError when the input string is not valid "
     "JSON. Common causes are trailing commas, single quotes, and unescaped "
     "control characters. Validate the payload before decoding."),
    ("sql_index", "Speeding up slow SQL queries with indexes",
     "A slow SQL query that filters on an unindexed column forces a full table "
     "scan. Adding a B-tree index on the WHERE clause column lets the database "
     "seek directly to matching rows and cuts query latency."),
    ("regex_email", "Writing a regular expression to validate email addresses",
     "A pragmatic email regex matches a local part, an at sign, a domain, and a "
     "top-level domain. Overly strict patterns reject valid addresses; prefer a "
     "lenient regex plus a confirmation email."),
    ("http_429", "HTTP 429 Too Many Requests and rate limiting",
     "An HTTP 429 status code means the client has sent too many requests in a "
     "given window. Respect the Retry-After header and back off exponentially "
     "to stay under the API rate limit."),
    ("revenue_chart", "Quarterly revenue chart showing growth across regions",
     "The bar chart plots quarterly revenue by region. Each colored bar is a "
     "region and the vertical axis is revenue in millions. The legend maps "
     "colors to regions; growth is steepest in the rightmost quarter."),
    ("org_diagram", "Organizational chart diagram of reporting structure",
     "The org chart diagram shows the reporting hierarchy as boxes connected by "
     "lines. The CEO sits at the top, with VPs below and teams branching out. "
     "The layout makes the management structure visible at a glance."),
    ("pricing_table", "Pricing table comparing subscription tiers",
     "The pricing table lays out three subscription tiers in columns: Free, "
     "Pro, and Enterprise. Rows list features and a check mark or dash shows "
     "which tier includes each feature. Prices appear in the table header."),
    ("docker_build", "Caching Docker layers to speed up image builds",
     "Docker builds each instruction as a layer and caches it. Ordering the "
     "Dockerfile so that rarely changing steps come first maximizes cache hits "
     "and makes rebuilds fast after editing application code."),
    ("git_rebase", "Using git rebase to keep a linear commit history",
     "git rebase replays your commits on top of another branch, producing a "
     "linear history without merge commits. Interactive rebase lets you squash, "
     "reorder, and edit commits before pushing."),
    ("k8s_oom", "Kubernetes pod killed with OOMKilled status",
     "A pod showing OOMKilled was terminated because it exceeded its memory "
     "limit. Raise the container memory limit, fix the leak, or add a request "
     "so the scheduler places the pod on a node with enough memory."),
    ("ssl_handshake", "Debugging a TLS SSL handshake failure",
     "An SSL handshake failure usually stems from a certificate mismatch, an "
     "expired certificate, or an unsupported cipher suite. Inspect the chain "
     "with openssl s_client and confirm the hostname matches the certificate."),
]

_QUERIES = [
    ("q1", "how do I fix a divide by zero error in python", {"py_zipdiv": 1.0}),
    ("q2", "json decode error trailing comma", {"json_parse": 1.0}),
    ("q3", "my sql query is slow on a where clause", {"sql_index": 1.0}),
    ("q4", "regex to validate an email address", {"regex_email": 1.0}),
    ("q5", "api returns 429 too many requests rate limit", {"http_429": 1.0}),
    ("q6", "bar chart of quarterly revenue by region", {"revenue_chart": 1.0}),
    ("q7", "diagram of the org reporting structure", {"org_diagram": 1.0}),
    ("q8", "table comparing subscription pricing tiers", {"pricing_table": 1.0}),
    ("q9", "speed up docker image build with layer cache", {"docker_build": 1.0}),
    ("q10", "keep a linear git history with rebase", {"git_rebase": 1.0}),
    ("q11", "pod oomkilled exceeded memory limit kubernetes", {"k8s_oom": 1.0}),
    ("q12", "tls handshake failure certificate mismatch", {"ssl_handshake": 1.0}),
]


def sample_dataset() -> EvalDataset:
    """Return the built-in offline sample dataset."""
    docs = [EvalDocument(doc_id=did, title=title, text=text) for did, title, text in _DOCS]
    queries = [EvalQuery(id=qid, query=q, relevant=rel) for qid, q, rel in _QUERIES]
    return EvalDataset(name="hybridrag-sample", documents=docs, queries=queries)

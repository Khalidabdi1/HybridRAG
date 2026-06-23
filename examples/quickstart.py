"""HybridRAG quickstart — runs with only numpy (uses fallback encoders).

    python examples/quickstart.py
"""

from hybridrag import HybridConfig
from hybridrag.engine import HybridRAG


def main() -> None:
    rag = HybridRAG(HybridConfig())

    rag.add_text(
        "wiki_python",
        "Python is a high-level, general-purpose programming language. Its design "
        "philosophy emphasizes code readability with the use of significant "
        "indentation. Python is dynamically typed and garbage-collected.",
        title="Python (programming language)",
    )
    rag.add_text(
        "wiki_finance",
        "A balance sheet is a financial statement that reports a company's assets, "
        "liabilities, and shareholder equity. The chart and table on the report "
        "show quarterly revenue growth across business segments.",
        title="Balance sheet",
    )

    print("Index stats:", rag.stats())
    print()

    for query in [
        "how is python typed?",
        "quarterly revenue chart and table",
        "def function indentation syntax error",
    ]:
        print(f"Query: {query!r}")
        for r in rag.search(query, top_k=3):
            comps = ", ".join(f"{k}={v:.4f}" for k, v in r.components.items())
            print(f"  - doc={r.doc_id} score={r.score:.4f} [{comps}]")
            if r.text:
                print(f"      {r.text[:90]}...")
        print()


if __name__ == "__main__":
    main()

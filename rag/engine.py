import os
import glob
import chromadb
from chromadb.utils import embedding_functions
import requests

CORPUS_DIR   = "corpus"
DB_DIR       = "chroma_db"
COLLECTION   = "sports_science"
EMBED_MODEL  = "all-MiniLM-L6-v2"
TOP_K        = 4
MAX_DISTANCE = 0.75 # cosine distance; above this = "not relevant" (guardrail)

# 1. CHUNKING — split long documents into overlapping passages so retrieval
#    returns a focused, citable piece instead of a whole paper. Overlap keeps
#    sentences from being cut mid-thought across a boundary.
def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

# 2. INDEX (offline) — embed every chunk once and store it in a persistent
#    Chroma collection. upsert() makes re-running safe (no duplicates).
#    The .txt filename becomes the citation id.
def build_index() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=DB_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    coll = client.get_or_create_collection(
        name=COLLECTION, embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    for path in glob.glob(os.path.join(CORPUS_DIR, "*.txt")):
        source = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            chunks = chunk_text(f.read())
        coll.upsert(
            documents=chunks,
            metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
            ids=[f"{source}-{i}" for i in range(len(chunks))],
        )
    return coll

# 3. RETRIEVE (online) — embed the query, pull the nearest chunks.
def retrieve(coll, query: str, k: int = TOP_K) -> list[dict]:
    res = coll.query(query_texts=[query], n_results=k)
    return [
        {"text": d, "source": m["source"], "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]

# 4. GROUNDED GENERATION — answer only from context, cite sources, refuse
#    when unsupported. This trust behavior is the whole point for a health tool.
GUARDRAIL = (
    "You are a careful clinical/sports-science assistant. Answer ONLY using the "
    "provided context. After each claim, cite the source in brackets, e.g. [gabbett_acwr]. "
    "If the context does not support an answer, say you don't have enough grounded "
    "evidence. Never use outside knowledge or speculate."
)

def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
    return f"{GUARDRAIL}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

def generate(prompt: str) -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
        timeout=120,
    )
    return r.json()["response"]

# 5. PIPELINE — model's SHAP features -> query -> retrieve -> grounded answer.
#    explain_prediction() in pitchguard_model.py returns the feature list
#    that goes straight into `top_features` here. That's the whole system.
def explain(coll, top_features: list[str], k: int = TOP_K) -> dict:
    query = ("Why do these factors raise a pitcher's UCL / Tommy John injury risk: "
             + ", ".join(top_features) + "?")
    chunks = retrieve(coll, query, k)
    if not chunks or chunks[0]["distance"] > MAX_DISTANCE:    # GUARDRAIL
        return {"answer": "Not enough grounded evidence to explain this prediction.",
                "sources": [], "query": query}
    answer = generate(build_prompt(query, chunks))
    return {"answer": answer, "sources": sorted({c["source"] for c in chunks}), "query": query}

# 6. EVALUATION — does retrieval surface the RIGHT source? Catches a broken
#    or badly chunked corpus before it embarrasses you in a demo.
def eval_retrieval(coll, tests: list[dict]) -> None:
    hits = 0
    for t in tests:
        got = {c["source"] for c in retrieve(coll, t["query"])}
        ok = t["expect_source"] in got
        hits += ok
        print(f"  {'OK  ' if ok else 'MISS'} {t['query'][:50]!r} -> {got}")
    print(f"retrieval recall@{TOP_K}: {hits}/{len(tests)}\n")


if __name__ == "__main__":
    coll = build_index()
    print(f"indexed {coll.count()} chunks from {CORPUS_DIR}/\n")

    # prove retrieval works (edit expect_source to match YOUR filenames):
    eval_retrieval(coll, [
        {"query": "does a workload spike increase injury risk?", "expect_source": "gabbett_acwr"},
        {"query": "is higher fastball velocity a UCL risk factor?", "expect_source": "velo_ucl"},
    ])

    # the features below come from pitchguard_model.explain_prediction(...)
    result = explain(coll, [
        "sharp rise in acute-to-chronic workload ratio",
        "declining fastball velocity",
        "lowered release point",
    ])
    print("EXPLANATION:\n", result["answer"])
    print("\nSOURCES:", result["sources"])
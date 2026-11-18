"""Path A — dense retrieval over an embedded schema index (ChromaDB).

Built last on purpose (design_notes §6): it is the only path with an external
service dependency — a vector store plus an embedding model — and bringing it up
alongside the evaluation harness would have made every early failure ambiguous
between "the retriever is wrong" and "the harness is wrong". The three
zero/low-infra paths proved the PipelineResult contract and the eval loop first;
Path A was wired in against a harness already known to be sound.
"""
import hashlib

from core.generate import generate_sql
from db.connection import get_inspector
from db.schema import render_ddl


def build_index(engine, collection_name: str, persist_dir: str = ".chroma"):
    """One document per table: table name plus its column names. Table-level
    granularity is coarse but adequate below ~50 tables; per-column documents are
    the upgrade path if retrieval recall on a padded schema demands it.

    The index build is Path A's offline cost. The store is persistent, so the
    collection itself survives between runs, but every run re-upserts all
    documents and therefore re-pays the embedding cost — which is the part RQ3
    trades against per-query cost. build_runner times this call and the harness
    records it as `setup_ms`, so the cost is measured rather than assumed.
    """
    import chromadb

    inspector = get_inspector(engine)
    tables = inspector.get_table_names()

    # The collection name carries a fingerprint of the schema it was built from.
    # The store is persistent, so without this a run on a padded schema would
    # reuse — or upsert on top of — an index describing a *different* schema, and
    # the sweep would measure a retriever pointed at the wrong database.
    fingerprint = hashlib.sha1(",".join(sorted(tables)).encode()).hexdigest()[:8]
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(f"{collection_name}_{fingerprint}")

    docs, ids = [], []
    for t in tables:
        cols = inspector.get_columns(t)
        docs.append(f"{t}: columns " + ", ".join(c["name"] for c in cols))
        ids.append(t)
    # upsert, not add: re-running the same configuration must overwrite the
    # documents rather than raise on duplicate ids.
    collection.upsert(documents=docs, ids=ids)
    return collection


def run_vector(question: str, engine, llm, db_id: str, collection, k: int = 5,
               max_distance: float = 1.35):
    hits = collection.query(query_texts=[question], n_results=k, include=["documents", "distances"])
    # An empty collection (inspector saw no tables, or the engine points at the
    # wrong file) yields no result lists. Fall back to an empty context so the
    # query is recorded as an ordinary failed row instead of raising IndexError
    # and killing the sweep.
    docs = (hits.get("documents") or [[]])[0] or []
    ids = (hits.get("ids") or [[]])[0] or []
    distances = (hits.get("distances") or [[]])[0] or []

    # Corrective-RAG-style fallback (Yan et al. 2024; Asai et al. 2024 Self-RAG;
    # ch2 rw_rag names both as relevant but "posed only for text corpora" — this
    # is that mechanism for a schema index). An empty hit list or a weak best
    # match means the retriever has no confident answer, and handing the LLM an
    # irrelevant top-k is worse than the full schema Path B already falls back
    # to. Chroma's default distance here is squared L2 over normalised
    # embeddings (~2*(1-cosine similarity)); max_distance=1.35 is an
    # uncalibrated placeholder ceiling — tune against dev-set retrieval
    # failures once this path is run against a padded schema.
    if not docs or (distances and min(distances) > max_distance):
        inspector = get_inspector(engine)
        all_tables = inspector.get_table_names()
        context = render_ddl(inspector, all_tables)
        ids = all_tables + ["<fallback:full_stuffing>"]
    else:
        context = "\n".join(docs)

    result = generate_sql(
        question, context, llm, db_id, path="A_vector", dialect=engine.dialect.name
    )
    # Record which tables the retriever actually returned. Without it an A_vector
    # failure cannot be attributed: a missed gold table (retrieval failure) and a
    # wrong query over the right tables (generation failure) both surface as ex=0,
    # and the crossover claim is about the first of those specifically. Schema
    # linking recall is then a post-hoc join of this column against gold_sql.
    result.tool_trace = ",".join(ids)
    return result

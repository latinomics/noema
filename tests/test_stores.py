"""M0 DoD: round-trip a synthetic DomainOntology through all three stores."""

from __future__ import annotations

import duckdb
import networkx as nx
import numpy as np

from noema.kernel.stores import (
    ENTITIES_TABLE,
    GraphStore,
    HashEmbedder,
    TabularStore,
    VectorStore,
    get_embedder,
    load_ontology,
    ontology_dir,
    save_ontology,
)
from tests.conftest import SRC_ID


def test_hash_embedder_is_deterministic_and_normalized():
    emb = HashEmbedder(dim=64)
    a = emb.encode(["feedback loop", "feedback loop", "stock"])
    assert np.allclose(a[0], a[1])
    assert not np.allclose(a[0], a[2])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)


def test_get_embedder_honors_hash_provider(settings):
    emb = get_embedder(settings)
    assert emb.name == "hash-64"
    assert emb.dim == 64


def test_ontology_roundtrip_through_stores(settings, ontology):
    """The M0 definition-of-done round trip: canonical JSON + all projections."""
    vectors = VectorStore(settings.paths.lancedb_dir, get_embedder(settings))
    tabular = TabularStore(settings.paths.lineage_db)

    out = save_ontology(ontology, settings, vectors=vectors, tabular=tabular)
    assert out == ontology_dir(settings, SRC_ID)

    # 1. canonical artifact round-trips losslessly
    restored = load_ontology(settings, SRC_ID)
    assert restored == ontology

    # 2. projections exist and carry the right counts
    assert (out / "axioms.jsonl").read_text().count("\n") == len(ontology.axioms)
    assert (out / "models.jsonl").read_text().count("\n") == len(ontology.mental_models)
    n = duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{out / 'entities.parquet'}')").fetchone()[0]
    assert n == len(ontology.entities)

    # 3. DuckDB entity table is queryable
    rows = tabular.query(
        "SELECT ent_id, kind FROM entities WHERE src_id = ? ORDER BY ent_id", [SRC_ID]
    )
    assert [r["ent_id"] for r in rows] == ["ent_feedback", "ent_flow", "ent_stock"]

    # 4. LanceDB slice returns the right entity for its own indexed text
    feedback = ontology.entities[2]
    query = f"{feedback.canonical_name} — {feedback.definition}"
    hits = vectors.search(ENTITIES_TABLE, query, k=3)
    assert hits[0]["id"] == f"{SRC_ID}:ent_feedback"
    assert vectors.count(ENTITIES_TABLE) == len(ontology.entities)


def test_save_is_idempotent_no_vector_duplicates(settings, ontology):
    vectors = VectorStore(settings.paths.lancedb_dir, get_embedder(settings))
    save_ontology(ontology, settings, vectors=vectors)
    save_ontology(ontology, settings, vectors=vectors)  # upsert, not append
    assert vectors.count(ENTITIES_TABLE) == len(ontology.entities)


def test_graph_store_roundtrip_and_graphml(settings, ontology, tmp_path):
    store = GraphStore()
    store.add_thesaurus_edges(ontology.thesaurus)

    jsonl = tmp_path / "thesaurus.jsonl"
    store.save_jsonl(jsonl)
    restored = GraphStore.load_jsonl(jsonl)
    assert sorted(restored.g.nodes) == sorted(store.g.nodes)
    assert restored.g.number_of_edges() == len(ontology.thesaurus)
    assert restored.thesaurus_edges() == ontology.thesaurus  # full contract round-trip

    graphml = tmp_path / "thesaurus.graphml"
    store.export_graphml(graphml)
    reloaded = nx.read_graphml(graphml)
    assert reloaded.number_of_edges() == len(ontology.thesaurus)

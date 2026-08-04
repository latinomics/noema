"""Embedded stores (BLUEPRINT §2.5): DuckDB tabular, LanceDB vectors, NetworkX graphs.

Everything is local-first and swap-friendly. Embeddings default to BGE-M3 via
sentence-transformers (optional extra); without it we fall back to a
deterministic hash embedder (fine for tests/dev, useless semantically) and log
a warning. One config line swaps in a hosted embedding API.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Protocol

import networkx as nx
import numpy as np

from noema.kernel.config import Settings
from noema.kernel.schemas import DomainOntology, ThesaurusEdge

logger = logging.getLogger(__name__)

# ── embedders ────────────────────────────────────────────────────────────────


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic pseudo-embeddings seeded from text bytes.

    Zero semantics — identical text maps to identical vectors, nothing more.
    Exists so the full storage/retrieval plumbing runs in tests and on machines
    without torch. Never ship semantic features on top of this.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(self.dim)
            out[i] = (v / np.linalg.norm(v)).astype(np.float32)
        return out


class SentenceTransformerEmbedder:
    """BGE-M3-class local embeddings (requires the `local-embeddings` extra)."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        from sentence_transformers import SentenceTransformer  # lazy: pulls torch

        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


class LiteLLMEmbedder:
    """Hosted embedding API via litellm (provider == "api")."""

    def __init__(self, model: str) -> None:
        self.name = model
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self.encode(["probe"]).shape[1]
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        import litellm  # lazy

        resp = litellm.embedding(model=self.name, input=texts)
        vecs = np.asarray([d["embedding"] for d in resp["data"]], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)


def get_embedder(settings: Settings) -> Embedder:
    cfg = settings.embeddings
    if cfg.provider == "hash":
        return HashEmbedder(cfg.hash_dim)
    if cfg.provider == "api":
        if not cfg.api_model:
            raise ValueError("embeddings.provider = 'api' requires embeddings.api_model")
        return LiteLLMEmbedder(cfg.api_model)
    try:
        return SentenceTransformerEmbedder(cfg.model)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed (uv sync --extra local-embeddings); "
            "falling back to the non-semantic hash embedder"
        )
        return HashEmbedder(cfg.hash_dim)


# ── vector store (LanceDB) ───────────────────────────────────────────────────


class VectorStore:
    """Thin LanceDB wrapper. Rows are dicts with at least {"id", "text"};
    vectors are computed here so callers never touch embeddings directly."""

    def __init__(self, root: Path, embedder: Embedder) -> None:
        import lancedb  # lazy

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(root))
        self.embedder = embedder

    def table_names(self) -> list[str]:
        return list(self.db.table_names())

    def upsert(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        vectors = self.embedder.encode([r["text"] for r in rows])
        data = [{**r, "vector": vectors[i].tolist()} for i, r in enumerate(rows)]
        if table in self.db.table_names():
            tbl = self.db.open_table(table)
            ids = ", ".join("'" + str(r["id"]).replace("'", "''") + "'" for r in rows)
            tbl.delete(f"id IN ({ids})")
            tbl.add(data)
        else:
            self.db.create_table(table, data=data)

    def search(self, table: str, query: str, k: int = 8) -> list[dict[str, Any]]:
        tbl = self.db.open_table(table)
        vec = self.embedder.encode([query])[0].tolist()
        return tbl.search(vec).limit(k).to_list()

    def count(self, table: str) -> int:
        return self.db.open_table(table).count_rows()


# ── graph store (NetworkX + JSONL persistence + GraphML export) ──────────────


class GraphStore:
    def __init__(self, graph: nx.MultiDiGraph | None = None) -> None:
        self.g = graph or nx.MultiDiGraph()

    def add_thesaurus_edges(self, edges: list[ThesaurusEdge]) -> None:
        for e in edges:
            self.g.add_node(e.subject)
            self.g.add_node(e.object)
            self.g.add_edge(
                e.subject,
                e.object,
                predicate=e.predicate,
                provenance=json.dumps([p.model_dump(mode="json") for p in e.provenance]),
            )

    def thesaurus_edges(self) -> list[ThesaurusEdge]:
        from noema.kernel.schemas import Provenance

        out = []
        for u, v, attrs in self.g.edges(data=True):
            out.append(
                ThesaurusEdge(
                    subject=u,
                    predicate=attrs["predicate"],
                    object=v,
                    provenance=[
                        Provenance.model_validate(p)
                        for p in json.loads(attrs.get("provenance", "[]"))
                    ],
                )
            )
        return out

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for node, attrs in self.g.nodes(data=True):
                f.write(json.dumps({"type": "node", "id": node, **attrs}) + "\n")
            for u, v, attrs in self.g.edges(data=True):
                f.write(json.dumps({"type": "edge", "u": u, "v": v, **attrs}) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path) -> GraphStore:
        store = cls()
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                kind = rec.pop("type")
                if kind == "node":
                    store.g.add_node(rec.pop("id"), **rec)
                else:
                    store.g.add_edge(rec.pop("u"), rec.pop("v"), **rec)
        return store

    def export_graphml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        export = nx.MultiDiGraph()
        for node, attrs in self.g.nodes(data=True):
            export.add_node(node, **{k: _scalar(v) for k, v in attrs.items()})
        for u, v, attrs in self.g.edges(data=True):
            export.add_edge(u, v, **{k: _scalar(val) for k, val in attrs.items()})
        nx.write_graphml(export, path)


def _scalar(value: Any) -> str | int | float | bool:
    return value if isinstance(value, str | int | float | bool) else json.dumps(value, default=str)


# ── tabular store (DuckDB; shares lineage.duckdb with the manifest) ──────────


class TabularStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._con() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    src_id         TEXT NOT NULL,
                    ent_id         TEXT NOT NULL,
                    canonical_name TEXT,
                    kind           TEXT,
                    idiosyncratic  BOOLEAN,
                    definition     TEXT,
                    aliases        TEXT,
                    provenance     TEXT,
                    PRIMARY KEY (src_id, ent_id)
                )
                """
            )

    def _con(self):
        import duckdb

        return duckdb.connect(str(self.db_path))

    def write_entities(self, ontology: DomainOntology) -> int:
        src_id = ontology.source.src_id
        with self._con() as con:
            con.execute("DELETE FROM entities WHERE src_id = ?", [src_id])
            for e in ontology.entities:
                con.execute(
                    "INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        src_id,
                        e.ent_id,
                        e.canonical_name,
                        e.kind,
                        e.idiosyncratic,
                        e.definition,
                        json.dumps(e.aliases),
                        json.dumps([p.model_dump(mode="json") for p in e.provenance]),
                    ],
                )
        return len(ontology.entities)

    def query(self, sql: str, args: list[Any] | None = None) -> list[dict[str, Any]]:
        with self._con() as con:
            cur = con.execute(sql, args or [])
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


# ── DomainOntology persistence (data/ontology/<src_id>/ layout) ──────────────

ENTITIES_TABLE = "entities"  # shared LanceDB table across sources (P3 aligns over it)


def ontology_dir(settings: Settings, src_id: str) -> Path:
    return settings.paths.ontology_dir / src_id


def save_ontology(
    ontology: DomainOntology,
    settings: Settings,
    *,
    vectors: VectorStore | None = None,
    tabular: TabularStore | None = None,
) -> Path:
    """Write the canonical ontology.json plus queryable projections:
    entities.parquet, axioms.jsonl, models.jsonl, thesaurus.jsonl (+ .graphml),
    a DuckDB entities table, and LanceDB entity embeddings."""
    src_id = ontology.source.src_id
    out = ontology_dir(settings, src_id)
    out.mkdir(parents=True, exist_ok=True)

    # canonical artifact — the only file load_ontology reads back
    (out / "ontology.json").write_text(ontology.model_dump_json(indent=2), encoding="utf-8")

    with open(out / "axioms.jsonl", "w", encoding="utf-8") as f:
        for ax in ontology.axioms:
            f.write(ax.model_dump_json() + "\n")
    with open(out / "models.jsonl", "w", encoding="utf-8") as f:
        for mm in ontology.mental_models:
            f.write(mm.model_dump_json() + "\n")

    graph = GraphStore()
    graph.add_thesaurus_edges(ontology.thesaurus)
    graph.save_jsonl(out / "thesaurus.jsonl")
    graph.export_graphml(out / "thesaurus.graphml")

    _write_entities_parquet(ontology, out / "entities.parquet")

    if tabular is not None:
        tabular.write_entities(ontology)

    if vectors is not None:
        vectors.upsert(
            ENTITIES_TABLE,
            [
                {
                    "id": f"{src_id}:{e.ent_id}",
                    "text": f"{e.canonical_name} — {e.definition}",
                    "src_id": src_id,
                    "ent_id": e.ent_id,
                    "kind": e.kind,
                }
                for e in ontology.entities
            ],
        )
    return out


def load_ontology(settings: Settings, src_id: str) -> DomainOntology:
    path = ontology_dir(settings, src_id) / "ontology.json"
    return DomainOntology.model_validate_json(path.read_text(encoding="utf-8"))


def _write_entities_parquet(ontology: DomainOntology, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {
            "src_id": ontology.source.src_id,
            "ent_id": e.ent_id,
            "canonical_name": e.canonical_name,
            "kind": e.kind,
            "idiosyncratic": e.idiosyncratic,
            "definition": e.definition,
            "aliases": json.dumps(e.aliases),
            "n_provenance": len(e.provenance),
        }
        for e in ontology.entities
    ]
    schema = pa.schema(
        [
            ("src_id", pa.string()),
            ("ent_id", pa.string()),
            ("canonical_name", pa.string()),
            ("kind", pa.string()),
            ("idiosyncratic", pa.bool_()),
            ("definition", pa.string()),
            ("aliases", pa.string()),
            ("n_provenance", pa.int32()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)

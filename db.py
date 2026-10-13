"""SQLite storage layer: source of truth for samples, embeddings, and results.

Tables:
  samples      - one row per mammogram image + metadata + official split
  embeddings   - vector BLOB keyed by (sample, embedder)  [the "result vectors"]
  experiments  - one row per evaluated config
  predictions  - per-query audit trail (retrieved neighbours + correctness)
  run_meta     - key/value regime tags

FAISS indices are built on demand from `embeddings` and are NOT stored here.
Mirrors the training-free RAC-on-BreakHis storage design, adapted to the
mammography metadata (dataset / view / laterality / case).
"""
import sqlite3
import numpy as np

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path     TEXT NOT NULL UNIQUE,
    file_name     TEXT NOT NULL,
    dataset       TEXT NOT NULL,     -- 'cbis' | 'inbreast'
    patient_id    TEXT,
    laterality    TEXT,             -- 'LEFT' | 'RIGHT'
    view          TEXT,             -- 'CC' | 'MLO'
    abnormality   TEXT,             -- 'mass' | 'calc' | NULL
    pathology     TEXT,             -- raw label string (e.g. MALIGNANT / BI-RADS)
    binary_label  INTEGER,          -- 0 benign | 1 malignant
    case_id       TEXT,             -- one breast+abnormality: patient+laterality(+abn)
    split         TEXT              -- 'train' (gallery) | 'test' (query)
);

CREATE TABLE IF NOT EXISTS embeddings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id     INTEGER NOT NULL REFERENCES samples(id),
    embedder_name TEXT NOT NULL,
    dim           INTEGER NOT NULL,
    vector        BLOB NOT NULL,     -- float32 np.tobytes()
    created_at    TEXT DEFAULT (datetime('now')),
    UNIQUE(sample_id, embedder_name)
);

CREATE TABLE IF NOT EXISTS experiments (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    embedder_name  TEXT,
    k              INTEGER,
    distance       TEXT,
    head           TEXT,
    gallery_view   TEXT,          -- 'CC' | 'MLO' | 'all'
    query_view     TEXT,          -- 'CC' | 'MLO' | 'both'
    level          TEXT,          -- 'image' | 'case'
    dataset_scope  TEXT,          -- 'cbis_mass' | 'cbis_mass+inbreast' | 'inbreast_ext' ...
    variant        TEXT DEFAULT 'baseline',
    seed           INTEGER,
    n_test         INTEGER,
    accuracy       REAL,
    f1             REAL,
    auc            REAL,
    sensitivity    REAL,
    specificity    REAL,
    created_at     TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id),
    sample_id       INTEGER,       -- image-level: samples.id; case-level: NULL
    case_id         TEXT,          -- set for case-level rows
    predicted_label INTEGER,
    true_label      INTEGER,
    score_pos       REAL,          -- malignant probability/weight-mass
    correct         INTEGER,
    neighbor_ids    TEXT,          -- JSON list of gallery sample ids (image-level)
    neighbor_scores TEXT           -- JSON list of similarity/distance
);

CREATE TABLE IF NOT EXISTS run_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_emb_name       ON embeddings(embedder_name);
CREATE INDEX IF NOT EXISTS idx_pred_exp        ON predictions(experiment_id);
CREATE INDEX IF NOT EXISTS idx_samples_split  ON samples(split);
CREATE INDEX IF NOT EXISTS idx_samples_view   ON samples(view);
CREATE INDEX IF NOT EXISTS idx_samples_ds     ON samples(dataset);
CREATE INDEX IF NOT EXISTS idx_samples_case   ON samples(case_id);
"""

TABLES = ["samples", "embeddings", "experiments", "predictions", "run_meta"]


def connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path=DB_PATH):
    """Create all tables if absent. Idempotent."""
    conn = connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def set_meta(conn, **kv):
    for k, v in kv.items():
        conn.execute("INSERT OR REPLACE INTO run_meta (key,value) VALUES (?,?)",
                     (k, str(v)))
    conn.commit()


def get_meta(conn, key, default=None):
    row = conn.execute("SELECT value FROM run_meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


# --- vector (de)serialization ------------------------------------------------

def pack_vector(vec):
    return np.ascontiguousarray(vec, dtype=np.float32).tobytes()


def unpack_vector(blob, dim):
    return np.frombuffer(blob, dtype=np.float32, count=dim)


# --- embedding helpers -------------------------------------------------------

def embedded_sample_ids(conn, embedder_name):
    """Set of sample ids already embedded for this embedder (resumability)."""
    rows = conn.execute(
        "SELECT sample_id FROM embeddings WHERE embedder_name = ?", (embedder_name,)
    ).fetchall()
    return {r["sample_id"] for r in rows}


def insert_embedding(conn, sample_id, embedder_name, vec):
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (sample_id, embedder_name, dim, vector) "
        "VALUES (?, ?, ?, ?)",
        (sample_id, embedder_name, int(len(vec)), pack_vector(vec)),
    )


def load_embeddings(conn, embedder_name, split=None, view=None, dataset=None):
    """Return (sample_ids, vectors[N,dim], labels) for the given filters,
    ordered by sample id so multiple encoders stay aligned by construction."""
    q = (
        "SELECT s.id, s.binary_label, e.vector, e.dim "
        "FROM embeddings e JOIN samples s ON s.id = e.sample_id "
        "WHERE e.embedder_name = ?"
    )
    params = [embedder_name]
    if split is not None:
        q += " AND s.split = ?"
        params.append(split)
    if view is not None:
        q += " AND s.view = ?"
        params.append(view)
    if dataset is not None:
        if isinstance(dataset, (list, tuple, set)):
            ds = list(dataset)
            q += f" AND s.dataset IN ({','.join('?' * len(ds))})"
            params.extend(ds)
        else:
            q += " AND s.dataset = ?"
            params.append(dataset)
    q += " ORDER BY s.id"
    rows = conn.execute(q, params).fetchall()
    if not rows:
        return np.array([], dtype=np.int64), np.empty((0, 0), np.float32), np.array([])
    ids = np.array([r["id"] for r in rows], dtype=np.int64)
    labels = np.array([r["binary_label"] for r in rows], dtype=np.int64)
    vecs = np.stack([unpack_vector(r["vector"], r["dim"]) for r in rows])
    return ids, vecs, labels


def sample_meta(conn, ids):
    """Map sample id -> dict(patient_id, case_id, view, laterality, dataset, binary_label)."""
    ids = list(int(i) for i in ids)
    if not ids:
        return {}
    out = {}
    CHUNK = 900  # stay under SQLite's variable limit
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        ph = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT id, patient_id, case_id, view, laterality, dataset, binary_label "
            f"FROM samples WHERE id IN ({ph})", chunk).fetchall()
        for r in rows:
            out[r["id"]] = dict(patient_id=r["patient_id"], case_id=r["case_id"],
                                view=r["view"], laterality=r["laterality"],
                                dataset=r["dataset"], binary_label=r["binary_label"])
    return out


if __name__ == "__main__":
    conn = init_db()
    names = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    print("DB:", DB_PATH)
    print("Tables:", names)
    assert all(t in names for t in TABLES), f"missing tables; got {names}"
    print("OK: all tables present")

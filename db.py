# db.py
import sqlite3
import json
import zlib
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd

DB_PATH_DEFAULT = Path(".portfolio_cache") / "cache.db"

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def ensure_db(db_path: Path = DB_PATH_DEFAULT) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            meta_json TEXT NOT NULL,
            close_blob BLOB NOT NULL
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS published (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            dataset_id TEXT NOT NULL,
            published_at TEXT NOT NULL,
            FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS runs_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            action TEXT NOT NULL,
            dataset_id TEXT,
            ok INTEGER NOT NULL,
            message TEXT
        )
        """)
        con.commit()

def _pack_df(df: pd.DataFrame) -> bytes:
    payload = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(payload, level=9)

def _unpack_df(blob: bytes) -> pd.DataFrame:
    payload = zlib.decompress(blob)
    return pickle.loads(payload)

def save_dataset(
    dataset_id: str,
    meta: Dict[str, Any],
    close_df: pd.DataFrame,
    db_path: Path = DB_PATH_DEFAULT
) -> None:
    ensure_db(db_path)
    meta_json = json.dumps(meta, ensure_ascii=False, sort_keys=True)
    close_blob = _pack_df(close_df)

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO datasets (dataset_id, created_at, meta_json, close_blob)
            VALUES (?, ?, ?, ?)
        """, (dataset_id, _utcnow_iso(), meta_json, close_blob))
        cur.execute("""
            INSERT INTO runs_log (created_at, action, dataset_id, ok, message)
            VALUES (?, ?, ?, ?, ?)
        """, (_utcnow_iso(), "save_dataset", dataset_id, 1, "saved"))
        con.commit()

def list_datasets(db_path: Path = DB_PATH_DEFAULT) -> pd.DataFrame:
    ensure_db(db_path)
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query("""
            SELECT dataset_id, created_at, meta_json
            FROM datasets
            ORDER BY created_at DESC
            LIMIT 200
        """, con)
    if not df.empty:
        df["meta"] = df["meta_json"].apply(lambda s: json.loads(s))
        df = df.drop(columns=["meta_json"])
    return df

def publish_dataset(dataset_id: str, db_path: Path = DB_PATH_DEFAULT) -> None:
    ensure_db(db_path)
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()

        # verify dataset exists
        cur.execute("SELECT 1 FROM datasets WHERE dataset_id = ? LIMIT 1", (dataset_id,))
        if cur.fetchone() is None:
            cur.execute("""
                INSERT INTO runs_log (created_at, action, dataset_id, ok, message)
                VALUES (?, ?, ?, ?, ?)
            """, (_utcnow_iso(), "publish", dataset_id, 0, "dataset not found"))
            con.commit()
            raise ValueError(f"Dataset not found: {dataset_id}")

        cur.execute("""
            INSERT INTO published (id, dataset_id, published_at)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET dataset_id=excluded.dataset_id, published_at=excluded.published_at
        """, (dataset_id, _utcnow_iso()))

        cur.execute("""
            INSERT INTO runs_log (created_at, action, dataset_id, ok, message)
            VALUES (?, ?, ?, ?, ?)
        """, (_utcnow_iso(), "publish", dataset_id, 1, "published"))
        con.commit()

def get_published_meta(db_path: Path = DB_PATH_DEFAULT) -> Optional[Dict[str, Any]]:
    ensure_db(db_path)
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute("SELECT dataset_id, published_at FROM published WHERE id=1")
        row = cur.fetchone()
        if row is None:
            return None
        dataset_id, published_at = row

        cur.execute("SELECT meta_json, created_at FROM datasets WHERE dataset_id = ?", (dataset_id,))
        row2 = cur.fetchone()
        if row2 is None:
            return None
        meta_json, created_at = row2
        meta = json.loads(meta_json)
        meta["_dataset_id"] = dataset_id
        meta["_published_at"] = published_at
        meta["_created_at"] = created_at
        return meta

def load_published_close(db_path: Path = DB_PATH_DEFAULT) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      close_df: columns = tickers, index = DatetimeIndex
      meta: dict (includes _dataset_id, _published_at)
    """
    ensure_db(db_path)
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute("SELECT dataset_id, published_at FROM published WHERE id=1")
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("No published dataset. Run admin.py and publish one dataset first.")
        dataset_id, published_at = row

        cur.execute("SELECT meta_json, close_blob FROM datasets WHERE dataset_id = ?", (dataset_id,))
        row2 = cur.fetchone()
        if row2 is None:
            raise RuntimeError(f"Published dataset missing in datasets table: {dataset_id}")

        meta_json, close_blob = row2
        close_df = _unpack_df(close_blob)
        meta = json.loads(meta_json)
        meta["_dataset_id"] = dataset_id
        meta["_published_at"] = published_at
        return close_df, meta
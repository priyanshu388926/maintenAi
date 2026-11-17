"""Prediction history storage: MongoDB when MONGODB_URI is set and reachable, otherwise a local JSON-lines file."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app import config


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FileStore:
    kind = "file"

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _write(self, rows: list[dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
        tmp.replace(self.path)

    def add(self, doc: dict) -> dict:
        doc = {"id": uuid.uuid4().hex, "created_at": now(), **doc}
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(doc) + "\n")
        return doc

    def list(self, limit: int = 100, risk: str | None = None) -> list[dict]:
        rows = self._read()
        if risk:
            rows = [r for r in rows if r.get("risk") == risk]
        return list(reversed(rows))[:limit]

    def get(self, id_: str) -> dict | None:
        return next((r for r in self._read() if r["id"] == id_), None)

    def delete(self, id_: str) -> bool:
        with self._lock:
            rows = self._read()
            keep = [r for r in rows if r["id"] != id_]
            self._write(keep)
        return len(keep) != len(rows)

    def clear(self) -> int:
        with self._lock:
            n = len(self._read())
            self._write([])
        return n

    def count(self) -> int:
        return len(self._read())


class MongoStore:
    kind = "mongodb"

    def __init__(self, uri: str, db: str):
        from pymongo import DESCENDING, MongoClient
        self._desc = DESCENDING
        self.client = MongoClient(uri, serverSelectionTimeoutMS=1500)
        self.client.admin.command("ping")
        self.col = self.client[db]["predictions"]
        self.col.create_index([("created_at", DESCENDING)])

    @staticmethod
    def _clean(d):
        if d:
            d.pop("_id", None)
        return d

    def add(self, doc: dict) -> dict:
        doc = {"id": uuid.uuid4().hex, "created_at": now(), **doc}
        self.col.insert_one(dict(doc))
        return doc

    def list(self, limit: int = 100, risk: str | None = None) -> list[dict]:
        q = {"risk": risk} if risk else {}
        return [self._clean(d) for d in self.col.find(q).sort("created_at", self._desc).limit(limit)]

    def get(self, id_: str):
        return self._clean(self.col.find_one({"id": id_}))

    def delete(self, id_: str) -> bool:
        return self.col.delete_one({"id": id_}).deleted_count == 1

    def clear(self) -> int:
        return self.col.delete_many({}).deleted_count

    def count(self) -> int:
        return self.col.count_documents({})


def make_store():
    if config.MONGODB_URI:
        try:
            return MongoStore(config.MONGODB_URI, config.MONGODB_DB), None
        except Exception as e:  # unreachable server: fall back so the app still works
            return FileStore(config.DATA_DIR / "history.jsonl"), f"MongoDB unavailable ({type(e).__name__}); using file storage"
    return FileStore(config.DATA_DIR / "history.jsonl"), None

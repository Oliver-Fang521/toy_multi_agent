import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from uuid import uuid4

from gaia_agent.interactive.models import InteractiveRequest


class ServiceError(RuntimeError):
    def __init__(self, code: str, status_code: int = 404):
        super().__init__(code)
        self.code = code
        self.status_code = status_code


class ConversationStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY, owner TEXT, created REAL);
            CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, conversation TEXT, owner TEXT,
                question TEXT, attachment TEXT, status TEXT, created REAL, result TEXT,
                progress TEXT DEFAULT '[]', error TEXT, idem TEXT, fingerprint TEXT,
                UNIQUE(owner, idem));
            CREATE TABLE IF NOT EXISTS uploads(id TEXT PRIMARY KEY, conversation TEXT, owner TEXT,
                path TEXT, name TEXT, created REAL);
            CREATE INDEX IF NOT EXISTS runs_owner_created ON runs(owner, created);
        """)
        self.db.commit()

    def create(self, owner: str) -> dict:
        identifier = str(uuid4())
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO conversations VALUES(?,?,?)", (identifier, owner, time.time())
            )
        return self.get(owner, identifier)

    def get(self, owner: str, identifier: str) -> dict:
        row = self.db.execute(
            "SELECT * FROM conversations WHERE id=? AND owner=?", (identifier, owner)
        ).fetchone()
        if row is None:
            raise ServiceError("not_found")
        return dict(row)

    def list(self, owner: str) -> list[dict]:
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT * FROM conversations WHERE owner=? ORDER BY created DESC LIMIT 100",
                (owner,),
            )
        ]

    def reserve(self, owner: str, cid: str, request: InteractiveRequest, idem: str) -> dict:
        fingerprint = hashlib.sha256((cid + request.model_dump_json()).encode()).hexdigest()
        with self.lock, self.db:
            self.get(owner, cid)
            existing = self.db.execute(
                "SELECT * FROM runs WHERE owner=? AND idem=?", (owner, idem)
            ).fetchone()
            if existing:
                if existing["fingerprint"] != fingerprint:
                    raise ServiceError("idempotency_conflict", 409)
                return self.run(owner, existing["id"])
            if self.db.execute(
                "SELECT count(*) FROM runs WHERE owner=? AND status IN ('queued','running')",
                (owner,),
            ).fetchone()[0]:
                raise ServiceError("active_run_limit", 429)
            if (
                self.db.execute(
                    "SELECT count(*) FROM runs WHERE owner=? AND created>?",
                    (owner, time.time() - 3600),
                ).fetchone()[0]
                >= 3
            ):
                raise ServiceError("hourly_limit", 429)
            if (
                self.db.execute(
                    "SELECT count(*) FROM runs WHERE status IN ('queued','running')"
                ).fetchone()[0]
                >= 12
            ):
                raise ServiceError("service_busy", 503)
            identifier = str(uuid4())
            self.db.execute(
                "INSERT INTO runs(id,conversation,owner,question,attachment,status,created,idem,fingerprint) VALUES(?,?,?,?,?,'queued',?,?,?)",
                (
                    identifier,
                    cid,
                    owner,
                    request.question,
                    request.attachment_id,
                    time.time(),
                    idem,
                    fingerprint,
                ),
            )
            return self.run(owner, identifier)

    def run(self, owner: str, identifier: str) -> dict:
        row = self.db.execute(
            "SELECT * FROM runs WHERE id=? AND owner=?", (identifier, owner)
        ).fetchone()
        if row is None:
            raise ServiceError("not_found")
        result = dict(row)
        result["result"] = json.loads(result["result"]) if result["result"] else None
        result["progress"] = json.loads(result["progress"])
        return {
            key: value
            for key, value in result.items()
            if key not in {"fingerprint", "idem", "owner"}
        }

    def update(self, identifier: str, *, status: str, result=None, error=None):
        with self.lock, self.db:
            self.db.execute(
                "UPDATE runs SET status=?,result=?,error=? WHERE id=? AND status NOT IN ('cancelled','interrupted')",
                (status, json.dumps(result) if result is not None else None, error, identifier),
            )

    def progress(self, identifier: str, event: str, payload: dict):
        # Only semantic stage names and counts; never model/tool text or arguments.
        if event not in {
            "plan_started",
            "plan_completed",
            "act_started",
            "act_completed",
            "verify_started",
            "verify_completed",
            "submit_started",
            "submit_completed",
            "correction_started",
            "correction_completed",
        }:
            return
        entry = {"event": event, "time": time.time()}
        with self.lock, self.db:
            row = self.db.execute("SELECT progress FROM runs WHERE id=?", (identifier,)).fetchone()
            events = json.loads(row[0])[-99:] + [entry]
            self.db.execute(
                "UPDATE runs SET progress=? WHERE id=?", (json.dumps(events), identifier)
            )

    def messages(self, owner: str, cid: str, offset: int = 0) -> dict:
        self.get(owner, cid)
        rows = self.db.execute(
            "SELECT question,result FROM runs WHERE conversation=? AND owner=? AND status='completed' ORDER BY created,rowid LIMIT 101 OFFSET ?",
            (cid, owner, offset),
        ).fetchall()
        messages = []
        for row in rows[:100]:
            result = json.loads(row[1])
            messages.extend([
                {"role": "user", "content": row[0]},
                {"role": "assistant", "content": result["answer"], "evidence": result.get("evidence", [])},
            ])
        return {"messages": messages, "next_offset": offset + 100 if len(rows) > 100 else None}

    def history(self, owner: str, cid: str) -> list[dict[str, str]]:
        self.get(owner, cid)
        rows = self.db.execute(
            "SELECT question,result FROM runs WHERE conversation=? AND owner=? AND status='completed' ORDER BY created DESC LIMIT 4",
            (cid, owner),
        ).fetchall()
        result = []
        for row in reversed(rows):
            result.extend(
                [
                    {"role": "user", "content": row[0]},
                    {"role": "assistant", "content": json.loads(row[1])["answer"]},
                ]
            )
        return result

    def interrupt_unfinished(self):
        with self.lock, self.db:
            self.db.execute(
                "UPDATE runs SET status='interrupted',error='service_restarted' WHERE status IN ('queued','running')"
            )

    def upload(self, owner: str, cid: str, path: Path, name: str) -> str:
        self.get(owner, cid)
        identifier = str(uuid4())
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO uploads VALUES(?,?,?,?,?,?)",
                (identifier, cid, owner, str(path), name, time.time()),
            )
        return identifier

    def attachment(self, owner: str, cid: str, identifier: str) -> Path:
        row = self.db.execute(
            "SELECT path,created FROM uploads WHERE id=? AND owner=? AND conversation=?",
            (identifier, owner, cid),
        ).fetchone()
        if row is None:
            raise ServiceError("attachment_not_found")
        path = Path(row[0])
        if not path.is_file() or path.is_symlink() or row["created"] < time.time() - 7 * 86400:
            raise ServiceError("attachment_expired")
        return path

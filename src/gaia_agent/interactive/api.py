import asyncio
import hmac
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from gaia_agent.interactive.composition import execute_interactive
from gaia_agent.interactive.models import InteractiveConfig, InteractiveRequest
from gaia_agent.interactive.service import RunService
from gaia_agent.interactive.store import ConversationStore, ServiceError

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
ALLOWED_TYPES = {
    ".txt": {"text/plain"},
    ".csv": {"text/csv", "text/plain"},
    ".pdf": {"application/pdf"},
    ".xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
    ".png": {"image/png"},
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".webp": {"image/webp"},
    ".mp3": {"audio/mpeg"},
    ".wav": {"audio/wav", "audio/x-wav"},
    ".m4a": {"audio/mp4", "audio/x-m4a"},
}


class RequestBoundary:
    """Authenticate and bound actual bytes before multipart can spool to disk."""

    def __init__(self, app, service_token):
        self.app = app
        self.token = service_token
        self.active_uploads = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        auth = headers.get(b"authorization", b"")
        user = headers.get(b"x-agent-user", b"")
        if not hmac.compare_digest(auth, ("Bearer " + self.token).encode()) or not user or len(user) > 128 or any(c < 33 or c > 127 for c in user):
            return await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
        if scope["method"] != "POST":
            return await self.app(scope, receive, send)
        upload = scope["path"].endswith("/uploads")
        limit = MAX_UPLOAD_BYTES + 1024 * 1024 if upload else 64 * 1024
        if upload and self.active_uploads >= 2:
            return await JSONResponse({"error": "service_busy"}, status_code=503)(scope, receive, send)
        if upload:
            self.active_uploads += 1
        try:
            body = bytearray()
            while True:
                try:
                    message = await asyncio.wait_for(receive(), timeout=15)
                except TimeoutError:
                    return await JSONResponse({"error": "upload_timeout"}, status_code=408)(scope, receive, send)
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > limit:
                    return await JSONResponse({"error": "file_too_large"}, status_code=413)(scope, receive, send)
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
            delivered = False

            async def bounded_receive():
                nonlocal delivered
                if delivered:
                    return await receive()
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}

            await self.app(scope, bounded_receive, send)
        finally:
            if upload:
                self.active_uploads -= 1


def create_app(
    root: Path | None = None, service_token: str | None = None, *, execute=None
) -> FastAPI:
    root = root or Path(os.environ.get("AGENT_DATA_DIR", "interactive-data"))
    token = service_token or os.environ.get("AGENT_SERVICE_TOKEN", "")
    if len(token) < 16:
        raise RuntimeError("service_credential_missing")
    store = ConversationStore(root / "state.sqlite")
    if execute is None:
        config = InteractiveConfig(openai_api_key=os.environ["OPENAI_API_KEY"])

        async def execute(request, workspace, history, on_progress, attachment_path):
            return await execute_interactive(
                request, config, workspace, history, on_progress, attachment_path=attachment_path
            )

    service = RunService(store, root / "runs", execute)
    active_upload_users: set[str] = set()

    @asynccontextmanager
    async def lifespan(app):
        store.interrupt_unfinished()
        # Remove stale upload records/files without touching any outside path.
        for row in store.db.execute(
            "SELECT id,path FROM uploads WHERE created<?", (time.time() - 7 * 86400,)
        ).fetchall():
            path = Path(row["path"])
            if path.resolve().is_relative_to((root / "uploads").resolve()) and path.is_file():
                path.unlink()
            with store.db:
                store.db.execute("DELETE FROM uploads WHERE id=?", (row["id"],))
        yield
        await service.shutdown()
        store.db.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.service = service
    app.add_middleware(RequestBoundary, service_token=token)

    async def owner(request: Request):
        authorization = request.headers.get("authorization", "")
        if not hmac.compare_digest(authorization, "Bearer " + token):
            raise HTTPException(401, "unauthorized")
        user = request.headers.get("x-agent-user", "")
        if not user or len(user) > 128 or not user.isascii() or any(ord(c) < 33 for c in user):
            raise HTTPException(401, "identity_missing")
        return user

    owner_dependency = Depends(owner)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return JSONResponse({"error": "invalid_request"}, status_code=422)

    @app.exception_handler(ServiceError)
    async def service_error(request, error):
        return JSONResponse({"error": error.code}, status_code=error.status_code)

    @app.exception_handler(Exception)
    async def internal_error(request, error):
        return JSONResponse({"error": "internal_error"}, status_code=500)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/conversations")
    async def conversations(user=owner_dependency):
        return store.list(user)

    @app.post("/v1/conversations")
    async def create_conversation(user=owner_dependency):
        return store.create(user)

    @app.get("/v1/conversations/{cid}")
    async def conversation(cid: str, offset: int = 0, user=owner_dependency):
        return {**store.get(user, cid), **store.messages(user, cid, max(0, offset))}

    @app.post("/v1/conversations/{cid}/uploads")
    async def upload(cid: str, request: Request, user=owner_dependency):
        store.get(user, cid)
        form = await request.form(max_files=2, max_fields=2)
        files = [value for _, value in form.multi_items() if isinstance(value, UploadFile)]
        if len(files) != 1 or len(form.getlist("file")) != 1 or form.get("file") is not files[0]:
            await form.close()
            raise HTTPException(422, "one_file_required")
        file = files[0]
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_TYPES or file.content_type not in ALLOWED_TYPES[suffix]:
            await form.close()
            raise HTTPException(415, "unsupported_file")
        if user in active_upload_users:
            await form.close()
            raise HTTPException(429, "upload_in_progress")
        if (
            store.db.execute(
                "SELECT count(*) FROM uploads WHERE owner=? AND created>?",
                (user, time.time() - 3600),
            ).fetchone()[0]
            >= 3
        ):
            await form.close()
            raise HTTPException(429, "upload_limit")
        recent_bytes = sum(
            Path(row[0]).stat().st_size
            for row in store.db.execute(
                "SELECT path FROM uploads WHERE owner=? AND created>?", (user, time.time() - 3600)
            )
            if Path(row[0]).is_file()
        )
        if recent_bytes >= 3 * MAX_UPLOAD_BYTES:
            await form.close()
            raise HTTPException(429, "upload_limit")
        directory = root / "uploads" / cid
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / (str(uuid4()) + suffix)
        total = 0
        active_upload_users.add(user)
        try:
            with destination.open("xb") as stream:
                while chunk := await file.read(64 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "file_too_large")
                    stream.write(chunk)
            if total == 0:
                raise HTTPException(422, "empty_file")
            identifier = store.upload(user, cid, destination, (file.filename or "attachment")[:160])
            return {"id": identifier, "name": file.filename, "bytes": total}
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        finally:
            active_upload_users.discard(user)
            await form.close()

    @app.post("/v1/conversations/{cid}/runs")
    async def start(cid: str, body: InteractiveRequest, request: Request, user=owner_dependency):
        if not body.question.strip():
            raise HTTPException(422, "empty_question")
        idem = request.headers.get("idempotency-key", "")
        if not idem or len(idem) > 128:
            raise HTTPException(422, "idempotency_key_required")
        return service.start(user, cid, body, idem)

    @app.get("/v1/runs/{identifier}")
    async def get_run(identifier: str, user=owner_dependency):
        return service.get(user, identifier)

    @app.post("/v1/runs/{identifier}/cancel")
    async def cancel(identifier: str, user=owner_dependency):
        return await service.cancel(user, identifier)

    return app

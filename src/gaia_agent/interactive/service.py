import asyncio
import shutil
from pathlib import Path

from gaia_agent.interactive.composition import InteractiveFailure
from gaia_agent.interactive.credential_vault import CredentialError
from gaia_agent.interactive.models import InteractiveRequest
from gaia_agent.interactive.store import ConversationStore


class RunService:
    def __init__(self, store: ConversationStore, root: Path, execute, *, credential_sessions=None):
        self.store = store
        self.root = root
        self.execute = execute
        self.tasks: dict[str, asyncio.Task] = {}
        self.slots = asyncio.Semaphore(2)
        self.credential_sessions = credential_sessions

    def start(
        self,
        owner: str,
        cid: str,
        request: InteractiveRequest,
        idem: str,
        *,
        credential_ref=None,
        credential_handle=None,
    ) -> dict:
        existing = self.store.lookup(owner, cid, request, idem)
        if existing is not None:
            return existing
        if self.credential_sessions is not None:
            credential_ref = credential_ref or self.credential_sessions.reference(
                owner, credential_handle
            )
            if credential_ref.owner != owner:
                raise CredentialError("credential_required", 422)
            self.credential_sessions.resolve(credential_ref)
        attachment = (
            self.store.attachment(owner, cid, request.attachment_id)
            if request.attachment_id
            else None
        )
        run = self.store.reserve(owner, cid, request, idem)
        if run["status"] == "queued" and run["id"] not in self.tasks:
            self.tasks[run["id"]] = asyncio.create_task(
                self._execute(owner, cid, run["id"], request, attachment, credential_ref)
            )
            self.tasks[run["id"]].add_done_callback(lambda _: self.tasks.pop(run["id"], None))
        return run

    async def _execute(self, owner, cid, identifier, request, attachment, credential_ref):
        workspace = self.root / identifier
        try:
            async with self.slots:
                if self.credential_sessions is not None:
                    self.credential_sessions.resolve(credential_ref)
                self.store.update(identifier, status="running")
                workspace.mkdir(parents=True, exist_ok=True)
                registered = None
                if attachment:
                    registered = workspace / ("attachment" + attachment.suffix)
                    await asyncio.to_thread(shutil.copyfile, attachment, registered)
                async with asyncio.timeout(500):
                    args = (
                        request,
                        workspace,
                        self.store.history(owner, cid),
                        lambda event, payload: self.store.progress(identifier, event, payload),
                        registered,
                    )
                    result = (
                        await self.execute(credential_ref, *args)
                        if self.credential_sessions is not None
                        else await self.execute(*args)
                    )
                self.store.update(
                    identifier, status="completed", result=result.model_dump(mode="json")
                )
        except asyncio.CancelledError:
            self.store.update(identifier, status="cancelled", error="cancelled")
            raise
        except TimeoutError:
            self.store.update(identifier, status="failed", error="timeout")
        except InteractiveFailure as error:
            self.store.update(identifier, status="failed", error=str(error))
        except CredentialError as error:
            self.store.update(identifier, status="failed", error=error.code)
        except Exception:  # noqa: BLE001 - private execution errors are opaque at the API boundary
            self.store.update(identifier, status="failed", error="execution_failed")
        finally:
            if workspace.exists():
                await asyncio.to_thread(shutil.rmtree, workspace)

    def get(self, owner: str, identifier: str) -> dict:
        return self.store.run(owner, identifier)

    async def cancel(self, owner: str, identifier: str) -> dict:
        run = self.get(owner, identifier)
        if run["status"] in {"queued", "running"}:
            self.store.update(identifier, status="cancelled", error="cancelled")
            task = self.tasks.get(identifier)
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        return self.get(owner, identifier)

    async def shutdown(self):
        tasks = list(self.tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def cancel_owner(self, owner: str):
        identifiers = [
            row[0]
            for row in self.store.db.execute(
                "SELECT id FROM runs WHERE owner=? AND status IN ('queued','running')", (owner,)
            )
        ]
        for identifier in identifiers:
            await self.cancel(owner, identifier)

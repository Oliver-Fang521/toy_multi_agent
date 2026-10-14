import asyncio
import shutil
from pathlib import Path

from gaia_agent.interactive.composition import InteractiveFailure
from gaia_agent.interactive.models import InteractiveRequest
from gaia_agent.interactive.store import ConversationStore


class RunService:
    def __init__(self, store: ConversationStore, root: Path, execute):
        self.store = store
        self.root = root
        self.execute = execute
        self.tasks: dict[str, asyncio.Task] = {}
        self.slots = asyncio.Semaphore(2)

    def start(self, owner: str, cid: str, request: InteractiveRequest, idem: str) -> dict:
        attachment = (
            self.store.attachment(owner, cid, request.attachment_id)
            if request.attachment_id
            else None
        )
        run = self.store.reserve(owner, cid, request, idem)
        if run["status"] == "queued" and run["id"] not in self.tasks:
            self.tasks[run["id"]] = asyncio.create_task(
                self._execute(owner, cid, run["id"], request, attachment)
            )
            self.tasks[run["id"]].add_done_callback(lambda _: self.tasks.pop(run["id"], None))
        return run

    async def _execute(self, owner, cid, identifier, request, attachment):
        workspace = self.root / identifier
        try:
            async with self.slots:
                self.store.update(identifier, status="running")
                workspace.mkdir(parents=True, exist_ok=True)
                registered = None
                if attachment:
                    registered = workspace / ("attachment" + attachment.suffix)
                    await asyncio.to_thread(shutil.copyfile, attachment, registered)
                async with asyncio.timeout(500):
                    result = await self.execute(
                        request,
                        workspace,
                        self.store.history(owner, cid),
                        lambda event, payload: self.store.progress(identifier, event, payload),
                        registered,
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

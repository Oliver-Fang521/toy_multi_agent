"""Deploy with: modal deploy deploy/modal_agent.py (named secret required)."""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("toy-agent-web")
volume = modal.Volume.from_name("toy-agent-web-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject(str(ROOT / "pyproject.toml"))
    .add_local_dir(str(ROOT / "src"), remote_path="/root/src")
    .env({"PYTHONPATH": "/root/src", "AGENT_DATA_DIR": "/data"})
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("toy-agent-web-byok-runtime")],
    volumes={"/data": volume},
    max_containers=1,
    timeout=600,
    scaledown_window=600,
    cpu=1,
    memory=1024,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    import asyncio
    from contextlib import asynccontextmanager

    from gaia_agent.interactive.api import create_app

    server = create_app()
    original_lifespan = server.router.lifespan_context

    @asynccontextmanager
    async def lifespan(instance):
        async def checkpoint():
            while True:
                await asyncio.sleep(15)
                instance.state.service.store.db.execute("PRAGMA wal_checkpoint(FULL)")
                await volume.commit.aio()

        async with original_lifespan(instance):
            worker = asyncio.create_task(checkpoint())
            try:
                yield
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
        await volume.commit.aio()

    server.router.lifespan_context = lifespan
    return server

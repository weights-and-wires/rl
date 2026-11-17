"""The HTTP surface: public health/session status, and SSO-protected course data.

Everything under `/api` needs a vetted proxy identity except `/api/health` and
`/api/session`, and so does the PDF book under `/book`. The MCP server at `/mcp`
is the exception: it ignores the proxy identity and takes a bearer token.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import course, curriculum, db, mcp_server, mcp_tokens, projects
from .auth import current_user
from .config import Settings, get_settings
from .runner import runpod as runpod_backend
from .runs import start_run


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    db.init()
    db.carry_progress(curriculum.renames())
    async with mcp_server.lifespan():
        yield


app = FastAPI(
    title="Weights and Wires — RL", docs_url=None, redoc_url=None, lifespan=lifespan
)

# The hub reads the progress summary from the browser, across origins.
HUB_ORIGIN = "https://courses.ifkash.dev"
HUB_CORS_PATH = "/api/progress/summary"


@app.middleware("http")
async def hub_cors(request: Request, call_next: Any) -> Any:
    """Add CORS headers for the hub on the summary path only.

    A middleware, rather than `CORSMiddleware`, so that the 401 from
    `current_user` carries the headers too and the hub can read it.
    """
    response = await call_next(request)
    if request.url.path == HUB_CORS_PATH:
        response.headers.append("Vary", "Origin")
        if request.headers.get("origin") == HUB_ORIGIN:
            response.headers["Access-Control-Allow-Origin"] = HUB_ORIGIN
            response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


# Models -----------------------------------------------------------------




class DraftBody(BaseModel):
    code: str = Field(max_length=200_000)


class ProgressBody(BaseModel):
    status: str = Field(pattern="^(started|solved)$")


class TokenBody(BaseModel):
    name: str = Field(min_length=1, max_length=mcp_tokens.MAX_NAME_LENGTH)


class RunBody(BaseModel):
    exercise_id: str = Field(min_length=1, max_length=128)
    mode: str = Field(pattern="^(run|check)$")
    code: str = Field(max_length=200_000)
    prefer_gpu: bool = False


# Health and session -----------------------------------------------------


@app.get("/api/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    lessons = curriculum.lesson_list()
    return {
        "ok": True,
        "commit": settings.git_sha,
        "lessons": len(lessons),
        "exercises": len(curriculum.load_exercises()),
        "gpu": settings.gpu_available,
        "book_pages": (course.book_info(settings) or {}).get("pages", 0),
    }


@app.get("/api/session")
def session_state(request: Request, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    try:
        name = current_user(request, settings)
    except HTTPException:
        return {"signed_in": False, "username": None}
    return {"signed_in": True, "username": name}





# Content ----------------------------------------------------------------


@app.get("/api/lessons")
def lessons(user: str = Depends(current_user)) -> dict[str, Any]:
    return {"parts": curriculum.parts(), "progress": db.get_progress(user)}


@app.get("/api/lessons/{slug}")
def lesson(slug: str, user: str = Depends(current_user)) -> dict[str, Any]:
    found = curriculum.get_lesson(slug)
    if found is None:
        raise HTTPException(status_code=404, detail="No such lesson")

    payload: dict[str, Any] = dict(found)
    exercise_id = found.get("exercise")
    if exercise_id:
        exercise = curriculum.get_exercise(str(exercise_id))
        if exercise:
            payload["exercise_detail"] = curriculum.public_exercise(exercise)
            payload["draft"] = db.get_draft(user, str(exercise_id))
    db.set_progress(user, slug, "started")
    return payload


@app.get("/api/exercises/{exercise_id}/solution")
def solution(exercise_id: str, user: str = Depends(current_user)) -> dict[str, Any]:
    exercise = curriculum.get_exercise(exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail="No such exercise")
    return {"id": exercise_id, "solution": exercise["solution"]}


@app.get("/api/projects")
def project_index(user: str = Depends(current_user)) -> dict[str, Any]:
    return {"projects": projects.project_list()}


@app.get("/api/projects/{slug}")
def project(slug: str, user: str = Depends(current_user)) -> dict[str, Any]:
    found = projects.get_project(slug)
    if found is None:
        raise HTTPException(status_code=404, detail="No such project")
    return found


@app.get("/api/projects/{slug}/slides")
def project_slides(slug: str, user: str = Depends(current_user)) -> dict[str, Any]:
    deck = projects.get_slides(slug)
    if deck is None:
        raise HTTPException(status_code=404, detail="This project has no slides")
    return deck


@app.get("/api/book")
def book(
    user: str = Depends(current_user), settings: Settings = Depends(get_settings)
) -> dict[str, Any]:
    """The whole course as one PDF, or None when this server has no PDF."""
    return {"book": course.book_info(settings)}


@app.get("/book/{name}")
def book_file(
    name: str, user: str = Depends(current_user), settings: Settings = Depends(get_settings)
) -> FileResponse:
    """The PDF, sent as a download. Only the file the manifest names is served."""
    info = course.book_info(settings)
    if info is None or info["url"] != f"/book/{name}":
        raise HTTPException(status_code=404, detail="No such book")
    return FileResponse(settings.book_dir / name, media_type="application/pdf", filename=name)


# Drafts and progress ----------------------------------------------------


@app.put("/api/drafts/{exercise_id}")
def put_draft(
    exercise_id: str, body: DraftBody, user: str = Depends(current_user)
) -> dict[str, Any]:
    if curriculum.get_exercise(exercise_id) is None:
        raise HTTPException(status_code=404, detail="No such exercise")
    db.save_draft(user, exercise_id, body.code)
    return {"saved": True}


@app.delete("/api/drafts/{exercise_id}")
def drop_draft(exercise_id: str, user: str = Depends(current_user)) -> dict[str, Any]:
    db.delete_draft(user, exercise_id)
    return {"saved": False}


@app.get("/api/progress/summary")
def progress_summary(user: str = Depends(current_user)) -> dict[str, Any]:
    """Completed count, lesson total, and the most recently touched lesson."""
    return course.progress_summary(user)


@app.put("/api/progress/{slug}")
def put_progress(
    slug: str, body: ProgressBody, user: str = Depends(current_user)
) -> dict[str, Any]:
    db.set_progress(user, slug, body.status)
    return {"slug": slug, "status": body.status}


@app.get("/api/me")
def me(user: str = Depends(current_user), settings: Settings = Depends(get_settings)):
    return {
        "username": user,
        "stats": db.stats(user),
        "runs": db.recent_runs(user),
        "gpu": settings.gpu_available,
    }


@app.get("/api/me/export")
def export(user: str = Depends(current_user)) -> PlainTextResponse:
    return PlainTextResponse(
        db.export_all(user),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{user}-rl.json"'},
    )


@app.get("/api/compute")
async def compute(
    user: str = Depends(current_user), settings: Settings = Depends(get_settings)
) -> dict[str, Any]:
    return {
        "local": {
            "workers": settings.local_workers,
            "memory_mb": settings.local_memory_mb,
            "timeout_ceiling": settings.local_timeout_ceiling,
        },
        "runpod": await runpod_backend.probe(settings),
    }


# The run stream ---------------------------------------------------------


@app.post("/api/run")
async def run(
    body: RunBody,
    user: str = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    started = start_run(
        user, body.exercise_id, body.mode, body.code, settings, prefer_gpu=body.prefer_gpu
    )
    if started is None:
        raise HTTPException(status_code=404, detail="No such exercise")

    async def stream() -> AsyncIterator[bytes]:
        async for event in started.events:
            yield _sse(event)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _sse(event: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(event, default=str)}\n\n".encode()


# MCP --------------------------------------------------------------------


def _token_view(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in ("id", "name", "prefix", "created_at", "last_used_at")}


@app.get("/api/mcp/tokens")
def list_mcp_tokens(
    request: Request,
    user: str = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """The MCP endpoint's URL and the caller's active tokens, without secrets."""
    url = settings.mcp_public_url or str(request.base_url).rstrip("/") + "/mcp"
    return {"url": url, "tokens": [_token_view(t) for t in db.list_tokens(user)]}


@app.post("/api/mcp/tokens")
def create_mcp_token(body: TokenBody, user: str = Depends(current_user)) -> dict[str, Any]:
    """Create a token for the caller. The response is the only copy of the secret."""
    try:
        record = mcp_tokens.create_token(user, body.name)
    except mcp_tokens.TokenLimitError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {key: record[key] for key in ("id", "name", "prefix", "created_at", "token")}


@app.delete("/api/mcp/tokens/{token_id}")
def revoke_mcp_token(token_id: int, user: str = Depends(current_user)) -> dict[str, Any]:
    if not db.revoke_token(token_id, username=user):
        raise HTTPException(status_code=404, detail="No such token")
    return {"revoked": True}


# Both spellings are routes of their own, so neither redirects: MCP clients
# don't follow a redirect on POST. They are registered before the SPA catch-all.
app.router.add_route("/mcp", mcp_server.endpoint, include_in_schema=False)
app.router.add_route("/mcp/", mcp_server.endpoint, include_in_schema=False)


# Static assets ----------------------------------------------------------


def _mount_frontend() -> None:
    dist = get_settings().frontend_dist.resolve()
    if not dist.is_dir():
        return
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    index = dist / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> Any:
        if path.startswith(("api/", "mcp/")):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        candidate = dist / path
        if path and candidate.is_file() and dist in candidate.resolve().parents:
            return FileResponse(candidate)
        return FileResponse(index)


_mount_frontend()

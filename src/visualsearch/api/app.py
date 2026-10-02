"""HTTP API: text, image and "more like this" search over the catalog."""

import io
import logging
import mimetypes
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from visualsearch.api.schemas import (
    MAX_K,
    HealthResponse,
    ResultItem,
    SearchResponse,
    TextQuery,
)
from visualsearch.config import ROOT
from visualsearch.service import (
    ItemNotFoundError,
    SearchResult,
    SearchService,
    UnknownIndexError,
    load_service,
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_FRONTEND_DIR = ROOT / "frontend" / "dist"

logger = logging.getLogger("visualsearch.api")

router = APIRouter(prefix="/api")

K_QUERY = Query(10, ge=1, le=MAX_K, description="Number of results")
INDEX_QUERY = Query(None, description="Index to search: hnsw, lsh or brute-force")


def get_service(request: Request) -> SearchService:
    return request.app.state.service


def create_app(service: SearchService | None = None, frontend_dir: Path | None = None) -> FastAPI:
    """Build the app. Without a `service`, the real one is loaded at startup (needs the data)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not hasattr(app.state, "service"):
            logger.info("loading catalog, indexes and CLIP model ...")
            start = perf_counter()
            app.state.service = load_service()
            logger.info("ready in %.1f s", perf_counter() - start)
        yield

    app = FastAPI(title="Visual Search", version="0.1.0", lifespan=lifespan)
    if service is not None:
        app.state.service = service

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = perf_counter()
        response = await call_next(request)
        logger.info(
            "%s %s -> %d in %.1f ms",
            request.method,
            request.url.path,
            response.status_code,
            (perf_counter() - start) * 1000,
        )
        return response

    app.include_router(router)

    frontend_dir = DEFAULT_FRONTEND_DIR if frontend_dir is None else frontend_dir
    if frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return app


@router.get("/health", response_model=HealthResponse)
def health(service: SearchService = Depends(get_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        images=len(service),
        indexes=service.index_names,
        default_index=service.default_index,
    )


@router.post("/search/text", response_model=SearchResponse)
def search_text(body: TextQuery, service: SearchService = Depends(get_service)) -> SearchResponse:
    result = _run("text", lambda: service.search_text(body.query, body.k, body.index))
    return _respond(service, result)


@router.post("/search/image", response_model=SearchResponse)
def search_image(
    file: UploadFile = File(description="A JPEG, PNG or WebP image, up to 10 MB"),
    k: int = K_QUERY,
    index: str | None = INDEX_QUERY,
    service: SearchService = Depends(get_service),
) -> SearchResponse:
    image = _read_upload(file)
    result = _run("image", lambda: service.search_image(image, k, index))
    return _respond(service, result)


@router.get("/search/similar/{item_id}", response_model=SearchResponse)
def search_similar(
    item_id: int,
    k: int = K_QUERY,
    index: str | None = INDEX_QUERY,
    service: SearchService = Depends(get_service),
) -> SearchResponse:
    result = _run("similar", lambda: service.similar_to(item_id, k, index))
    return _respond(service, result)


@router.get("/images/{item_id}")
def get_image(item_id: int, service: SearchService = Depends(get_service)) -> FileResponse:
    try:
        path = service.image_path(item_id)
    except ItemNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(
        path, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"}
    )


def _read_upload(upload: UploadFile) -> Image.Image:
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(400, "could not read the upload as an image") from error
    return image


def _run(kind: str, search: Callable[[], SearchResult]) -> SearchResult:
    try:
        result = search()
    except UnknownIndexError as error:
        raise HTTPException(422, str(error)) from error
    except ItemNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    logger.info(
        "search kind=%s index=%s hits=%d embed_ms=%.1f search_ms=%.2f",
        kind,
        result.index,
        len(result.hits),
        result.embed_ms,
        result.search_ms,
    )
    return result


def _respond(service: SearchService, result: SearchResult) -> SearchResponse:
    items = []
    for hit in result.hits:
        width, height = service.image_size(hit.id)
        items.append(
            ResultItem(
                id=hit.id,
                score=round(hit.score, 4),
                category=hit.category,
                image_url=f"/api/images/{hit.id}",
                width=width,
                height=height,
            )
        )
    return SearchResponse(
        index=result.index,
        total_images=len(service),
        embed_ms=round(result.embed_ms, 2),
        search_ms=round(result.search_ms, 3),
        results=items,
    )

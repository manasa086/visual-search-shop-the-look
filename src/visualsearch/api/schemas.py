"""Request and response shapes for the HTTP API."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

MAX_K = 50
MAX_QUERY_CHARS = 200


class TextQuery(BaseModel):
    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS)
    ]
    k: int = Field(default=10, ge=1, le=MAX_K)
    index: str | None = None


class ResultItem(BaseModel):
    id: int
    score: float = Field(description="Cosine similarity to the query, from -1 to 1")
    category: str
    image_url: str
    width: int
    height: int


class SearchResponse(BaseModel):
    index: str
    total_images: int
    embed_ms: float
    search_ms: float
    results: list[ResultItem]


class HealthResponse(BaseModel):
    status: str
    images: int
    indexes: list[str]
    default_index: str

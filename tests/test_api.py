import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from visualsearch.api import app as app_module
from visualsearch.api.app import create_app


@pytest.fixture
def client(service, tmp_path) -> TestClient:
    return TestClient(create_app(service, frontend_dir=tmp_path / "no-frontend"))


def png_bytes(color=(210, 30, 30), size=(64, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_reports_catalog_and_indexes(client):
    body = client.get("/api/health").json()

    assert body == {
        "status": "ok",
        "images": 24,
        "indexes": ["brute-force", "hnsw", "lsh"],
        "default_index": "brute-force",
    }


def test_text_search(client):
    response = client.post("/api/search/text", json={"query": "  something red ", "k": 4})

    assert response.status_code == 200
    body = response.json()
    assert body["index"] == "brute-force"
    assert body["total_images"] == 24
    assert len(body["results"]) == 4
    assert {item["category"] for item in body["results"]} == {"red"}
    first = body["results"][0]
    assert set(first) == {"id", "score", "category", "image_url", "width", "height"}
    assert first["image_url"] == f"/api/images/{first['id']}"
    assert first["width"] > 0 and first["height"] > 0


def test_image_search_by_upload(client):
    files = {"file": ("query.png", png_bytes((30, 200, 40)), "image/png")}

    response = client.post("/api/search/image?k=3&index=hnsw", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["index"] == "hnsw"
    assert {item["category"] for item in body["results"]} == {"green"}


def test_similar_search_excludes_the_item(client):
    response = client.get("/api/search/similar/10?k=5")

    body = response.json()
    assert response.status_code == 200
    assert 10 not in [item["id"] for item in body["results"]]
    assert len(body["results"]) == 5
    assert body["embed_ms"] == 0


def test_result_images_can_be_fetched(client):
    item = client.post("/api/search/text", json={"query": "blue", "k": 1}).json()["results"][0]

    response = client.get(item["image_url"])

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert "max-age" in response.headers["cache-control"]
    assert Image.open(io.BytesIO(response.content)).size == (item["width"], item["height"])


@pytest.mark.parametrize("path", ["/api/images/9999", "/api/images/-1", "/api/search/similar/9999"])
def test_unknown_ids_return_404(client, path):
    assert client.get(path).status_code == 404


def test_unknown_index_returns_422_naming_the_choices(client):
    response = client.post("/api/search/text", json={"query": "red", "index": "faiss"})

    assert response.status_code == 422
    assert "hnsw" in response.json()["detail"]


@pytest.mark.parametrize("k", [0, 51, -3])
def test_k_out_of_range_is_rejected(client, k):
    assert client.post("/api/search/text", json={"query": "red", "k": k}).status_code == 422
    assert client.get(f"/api/search/similar/1?k={k}").status_code == 422
    files = {"file": ("q.png", png_bytes(), "image/png")}
    assert client.post(f"/api/search/image?k={k}", files=files).status_code == 422


@pytest.mark.parametrize("query", ["", "   ", "x" * 201])
def test_bad_text_queries_are_rejected(client, query):
    assert client.post("/api/search/text", json={"query": query}).status_code == 422


def test_upload_that_is_not_an_image_returns_400(client):
    files = {"file": ("notes.txt", b"this is not an image", "text/plain")}

    response = client.post("/api/search/image", files=files)

    assert response.status_code == 400


def test_empty_upload_returns_400(client):
    assert (
        client.post("/api/search/image", files={"file": ("e.png", b"", "image/png")}).status_code
        == 400
    )


def test_missing_file_field_returns_422(client):
    assert client.post("/api/search/image").status_code == 422


def test_oversized_upload_returns_413(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 100)
    files = {"file": ("big.png", png_bytes(size=(200, 200)), "image/png")}

    assert client.post("/api/search/image", files=files).status_code == 413


def test_built_frontend_is_served_at_the_root(service, tmp_path):
    (tmp_path / "index.html").write_text("<h1>Visual search</h1>")
    client = TestClient(create_app(service, frontend_dir=tmp_path))

    assert "Visual search" in client.get("/").text
    assert client.get("/api/health").status_code == 200

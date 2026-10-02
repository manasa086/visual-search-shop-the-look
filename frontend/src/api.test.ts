import { describe, expect, it, vi } from "vitest";
import { ApiError, getHealth, searchImage, searchSimilar, searchText } from "./api";

const RESPONSE = {
  index: "hnsw",
  total_images: 3,
  embed_ms: 1,
  search_ms: 0.1,
  results: [],
};

function stubFetch(body: unknown, status = 200) {
  const fetchMock = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("api client", () => {
  it("posts text queries as JSON", async () => {
    const fetchMock = stubFetch(RESPONSE);

    const result = await searchText("red shoe", "lsh", 12);

    expect(result).toEqual(RESPONSE);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/search/text");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ query: "red shoe", index: "lsh", k: 12 });
  });

  it("uploads images as multipart form data with the index and count in the URL", async () => {
    const fetchMock = stubFetch(RESPONSE);
    const file = new File(["bytes"], "photo.png", { type: "image/png" });

    await searchImage(file, "brute-force", 5);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/search/image?index=brute-force&k=5");
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get("file")).toBeInstanceOf(File);
  });

  it("requests similar items by id", async () => {
    const fetchMock = stubFetch(RESPONSE);

    await searchSimilar(42, "hnsw", 30);

    expect(fetchMock.mock.calls[0][0]).toBe("/api/search/similar/42?index=hnsw&k=30");
  });

  it("reads the health endpoint", async () => {
    const health = { status: "ok", images: 3, indexes: ["hnsw"], default_index: "hnsw" };
    stubFetch(health);

    expect(await getHealth()).toEqual(health);
  });

  it("raises the server's message for string errors", async () => {
    stubFetch({ detail: "unknown index 'x'" }, 422);

    await expect(searchText("a", "x", 5)).rejects.toThrow(new ApiError("unknown index 'x'"));
  });

  it("joins validation messages for list errors", async () => {
    stubFetch({ detail: [{ msg: "too short" }, { msg: "bad value" }] }, 422);

    await expect(searchText("", "hnsw", 5)).rejects.toThrow("too short; bad value");
  });

  it("falls back to a generic message when the error body is not JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("oops", { status: 502 })));

    await expect(getHealth()).rejects.toThrow("Request failed (502)");
  });
});

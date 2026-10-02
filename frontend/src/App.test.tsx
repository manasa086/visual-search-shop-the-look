import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { App } from "./App";
import type { ResultItem } from "./api";

const HEALTH = {
  status: "ok",
  images: 29780,
  indexes: ["hnsw", "lsh", "brute-force"],
  default_index: "hnsw",
};

function item(id: number, category: string, score = 0.5): ResultItem {
  return { id, score, category, image_url: `/api/images/${id}`, width: 200, height: 300 };
}

function searchResponse(results: ResultItem[], index = "hnsw", embedMs = 12) {
  return { index, total_images: 29780, embed_ms: embedMs, search_ms: 0.4, results };
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** Routes fetch calls to canned responses; unmatched URLs fail the test. */
function stubApi(routes: Record<string, (url: string, init?: RequestInit) => Response>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const path = url.split("?")[0];
    const handler = routes[path] ?? routes[url];
    if (!handler) throw new Error(`unexpected request: ${url}`);
    return handler(url, init);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("App", () => {
  it("shows the catalog size and example searches once connected", async () => {
    stubApi({ "/api/health": () => json(HEALTH) });

    render(<App />);

    expect(await screen.findByText(/Searching 29,780 images/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "a sunflower" })).toBeEnabled();
    expect(screen.getByRole("radio", { name: /HNSW/ })).toBeChecked();
  });

  it("searches by text and lists the results with timing", async () => {
    const user = userEvent.setup();
    const fetchMock = stubApi({
      "/api/health": () => json(HEALTH),
      "/api/search/text": () =>
        json(searchResponse([item(1, "butterfly", 0.32), item(2, "ketch-101")])),
    });
    render(<App />);
    await screen.findByText(/Searching 29,780/);

    await user.type(screen.getByRole("searchbox"), "a butterfly");
    await user.click(screen.getByRole("button", { name: "Search" }));

    expect(
      await screen.findByRole("button", { name: /butterfly, similarity 0.32/ }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /ketch,/ })).toBeInTheDocument();
    expect(
      screen.getByText(/2 results from 29,780 images in 0.40 ms using HNSW/),
    ).toBeInTheDocument();
    expect(screen.getByText(/embedding the query took 12 ms/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Results for .a butterfly./ })).toBeInTheDocument();
    const body = JSON.parse(fetchMock.mock.calls.at(-1)![1]!.body as string);
    expect(body).toEqual({ query: "a butterfly", index: "hnsw", k: 30 });
  });

  it("finds more like a clicked result", async () => {
    const user = userEvent.setup();
    const fetchMock = stubApi({
      "/api/health": () => json(HEALTH),
      "/api/search/text": () => json(searchResponse([item(7, "teapot")])),
      "/api/search/similar/7": () =>
        json(searchResponse([item(8, "teapot"), item(9, "coffee-mug")], "hnsw", 0)),
    });
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "a teapot" }));

    await user.click(await screen.findByRole("button", { name: /teapot, similarity/ }));

    expect(
      await screen.findByRole("heading", { name: "More like this teapot" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("searchbox")).toHaveValue(""); // the old text query no longer applies
    expect(await screen.findByRole("button", { name: /coffee mug/ })).toBeInTheDocument();
    expect(screen.queryByText(/embedding the query/)).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.at(-1)![0]).toBe("/api/search/similar/7?index=hnsw&k=30");
  });

  it("repeats the current search with the newly chosen method", async () => {
    const user = userEvent.setup();
    const fetchMock = stubApi({
      "/api/health": () => json(HEALTH),
      "/api/search/text": (_url, init) => {
        const { index } = JSON.parse(init!.body as string);
        return json(searchResponse([item(1, "guitar")], index));
      },
    });
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "an electric guitar" }));
    await screen.findByText(/using HNSW/);

    await user.click(screen.getByRole("radio", { name: /Brute force/ }));

    expect(await screen.findByText(/using Brute force/)).toBeInTheDocument();
    expect(fetchMock.mock.calls.filter(([url]) => url === "/api/search/text")).toHaveLength(2);
  });

  it("searches with an uploaded photo", async () => {
    const user = userEvent.setup();
    const fetchMock = stubApi({
      "/api/health": () => json(HEALTH),
      "/api/search/image": () => json(searchResponse([item(3, "spider")])),
    });
    render(<App />);
    await screen.findByText(/Searching 29,780/);

    await user.upload(
      screen.getByTestId("file-input"),
      new File(["x"], "spider.jpg", { type: "image/jpeg" }),
    );

    expect(
      await screen.findByRole("heading", { name: "Similar to your photo" }),
    ).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /spider/ })).toBeInTheDocument();
    expect(fetchMock.mock.calls.at(-1)![0]).toBe("/api/search/image?index=hnsw&k=30");
  });

  it("rejects non-image files without calling the API", async () => {
    const user = userEvent.setup({ applyAccept: false });
    const fetchMock = stubApi({ "/api/health": () => json(HEALTH) });
    render(<App />);
    await screen.findByText(/Searching 29,780/);

    await user.upload(
      screen.getByTestId("file-input"),
      new File(["x"], "notes.txt", { type: "text/plain" }),
    );

    expect(await screen.findByRole("alert")).toHaveTextContent("That file is not an image.");
    expect(fetchMock).toHaveBeenCalledTimes(1); // only the health check
  });

  it("shows the server's error message and can retry", async () => {
    const user = userEvent.setup();
    let calls = 0;
    stubApi({
      "/api/health": () => json(HEALTH),
      "/api/search/text": () =>
        ++calls === 1
          ? json({ detail: "model is warming up" }, 503)
          : json(searchResponse([item(1, "sunflower")])),
    });
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "a sunflower" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("model is warming up");
    await user.click(within(alert).getByRole("button", { name: "Try again" }));

    expect(
      await screen.findByRole("button", { name: /sunflower, similarity/ }),
    ).toBeInTheDocument();
  });

  it("explains when the API cannot be reached and reconnects on request", async () => {
    const user = userEvent.setup();
    let reachable = false;
    stubApi({
      "/api/health": () => {
        if (!reachable) throw new TypeError("network down");
        return json(HEALTH);
      },
    });
    render(<App />);

    expect(await screen.findByText(/Cannot reach the search API/)).toBeInTheDocument();
    expect(screen.getByRole("searchbox")).toBeDisabled();

    reachable = true;
    await user.click(screen.getByRole("button", { name: "Try again" }));

    await waitFor(() => expect(screen.getByRole("searchbox")).toBeEnabled());
    expect(screen.queryByText(/Cannot reach/)).not.toBeInTheDocument();
  });

  it("tells the user when a method returns nothing, and clears back to the start", async () => {
    const user = userEvent.setup();
    stubApi({
      "/api/health": () => json(HEALTH),
      "/api/search/text": () => json(searchResponse([], "lsh")),
    });
    render(<App />);
    await user.click(await screen.findByRole("button", { name: "a teapot" }));

    expect(await screen.findByText(/No matches/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Clear" }));

    expect(
      screen.getByRole("heading", { name: /Find images that look like an idea/ }),
    ).toBeInTheDocument();
    expect(screen.getByRole("searchbox")).toHaveValue("");
  });
});

import { useCallback, useEffect, useState } from "react";
import {
  getHealth,
  searchImage,
  searchSimilar,
  searchText,
  type Health,
  type ResultItem,
  type SearchResponse,
} from "./api";
import { IndexPicker, indexLabel } from "./components/IndexPicker";
import { QuerySummary, type Query } from "./components/QuerySummary";
import { ResultsGrid, ResultsSkeleton } from "./components/ResultsGrid";
import { SearchBar } from "./components/SearchBar";
import { formatMs } from "./format";
import { useImageInput } from "./useImageInput";

const RESULT_COUNT = 30;
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const EXAMPLES = [
  "an electric guitar",
  "a sunflower",
  "a teapot",
  "a mountain bike",
  "a grand piano",
];

type HealthState = { state: "loading" } | { state: "error" } | { state: "ready"; health: Health };

type Status =
  | { state: "loading" }
  | { state: "error"; message: string }
  | { state: "done"; data: SearchResponse };

export function App() {
  const [health, setHealth] = useState<HealthState>({ state: "loading" });
  const [index, setIndex] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [query, setQuery] = useState<Query | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const connect = useCallback(() => {
    setHealth({ state: "loading" });
    getHealth()
      .then((result) => {
        setHealth({ state: "ready", health: result });
        setIndex((current) => current ?? result.default_index);
      })
      .catch(() => setHealth({ state: "error" }));
  }, []);

  useEffect(() => {
    connect();
  }, [connect]);

  useEffect(() => {
    if (!query || !index) {
      setStatus(null);
      return;
    }
    const controller = new AbortController();
    const { signal } = controller;
    setStatus({ state: "loading" });

    const request =
      query.kind === "text"
        ? searchText(query.text, index, RESULT_COUNT, signal)
        : query.kind === "image"
          ? searchImage(query.file, index, RESULT_COUNT, signal)
          : searchSimilar(query.item.id, index, RESULT_COUNT, signal);

    request
      .then((data) => setStatus({ state: "done", data }))
      .catch((error: unknown) => {
        if (signal.aborted) return; // a newer search replaced this one
        const message = error instanceof Error ? error.message : "Something went wrong.";
        setStatus({ state: "error", message });
      });
    return () => controller.abort();
  }, [query, index]);

  const searchWithText = useCallback((text: string) => {
    setNotice(null);
    setDraft(text);
    setQuery({ kind: "text", text });
  }, []);

  const searchWithFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setNotice("That file is not an image.");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setNotice("Images must be 10 MB or smaller.");
      return;
    }
    setNotice(null);
    setDraft("");
    setQuery({ kind: "image", file });
  }, []);

  const searchSimilarTo = useCallback((item: ResultItem) => {
    setNotice(null);
    setDraft("");
    setQuery({ kind: "similar", item });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const clear = () => {
    setQuery(null);
    setDraft("");
    setNotice(null);
  };

  const dragging = useImageInput(searchWithFile);
  const ready = health.state === "ready" && index !== null;

  return (
    <div className="app">
      <header className="top">
        <div className="top-inner">
          <h1>Visual Search</h1>
          <SearchBar
            value={draft}
            onChange={setDraft}
            onSubmitText={searchWithText}
            onPickFile={searchWithFile}
            disabled={health.state !== "ready"}
          />
          {health.state === "ready" && index && (
            <IndexPicker indexes={health.health.indexes} value={index} onChange={setIndex} />
          )}
        </div>
      </header>

      <main>
        {notice && (
          <p role="alert" className="banner">
            {notice}
          </p>
        )}
        {health.state === "error" && (
          <div role="alert" className="banner">
            <span>Cannot reach the search API. Is the server running?</span>
            <button type="button" className="text-button" onClick={connect}>
              Try again
            </button>
          </div>
        )}

        {query ? (
          <QuerySummary query={query} onClear={clear} />
        ) : (
          <section className="welcome">
            <h2>Find images that look like an idea</h2>
            <p>
              Describe what you want, upload or paste a photo, or click any result to see more like
              it.
              {health.state === "ready" &&
                ` Searching ${health.health.images.toLocaleString()} images.`}
            </p>
            <ul className="examples" aria-label="Example searches">
              {EXAMPLES.map((example) => (
                <li key={example}>
                  <button type="button" disabled={!ready} onClick={() => searchWithText(example)}>
                    {example}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        {status?.state === "loading" && (
          <>
            <p role="status" className="visually-hidden">
              Searching…
            </p>
            <ResultsSkeleton />
          </>
        )}
        {status?.state === "error" && (
          <div role="alert" className="banner">
            <span>{status.message}</span>
            {query && (
              <button type="button" className="text-button" onClick={() => setQuery({ ...query })}>
                Try again
              </button>
            )}
          </div>
        )}
        {status?.state === "done" && (
          <>
            <p className="stats" aria-live="polite">
              {status.data.results.length} results from {status.data.total_images.toLocaleString()}{" "}
              images in {formatMs(status.data.search_ms)} using {indexLabel(status.data.index)}
              {status.data.embed_ms > 0 &&
                ` · embedding the query took ${formatMs(status.data.embed_ms)}`}
            </p>
            {status.data.results.length > 0 ? (
              <ResultsGrid results={status.data.results} onSelect={searchSimilarTo} />
            ) : (
              <p className="empty">
                No matches. This search method may have missed them; try another one above.
              </p>
            )}
          </>
        )}
      </main>

      {dragging && (
        <div className="drop-overlay" aria-hidden="true">
          Drop an image to search with it
        </div>
      )}
    </div>
  );
}

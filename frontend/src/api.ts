export type IndexName = "hnsw" | "lsh" | "brute-force";

export interface ResultItem {
  id: number;
  score: number;
  category: string;
  image_url: string;
  width: number;
  height: number;
}

export interface SearchResponse {
  index: string;
  total_images: number;
  embed_ms: number;
  search_ms: number;
  results: ResultItem[];
}

export interface Health {
  status: string;
  images: number;
  indexes: string[];
  default_index: string;
}

export class ApiError extends Error {}

async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new ApiError(await errorMessage(response));
  }
  return (await response.json()) as T;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((problem: { msg: string }) => problem.msg).join("; ");
    }
  } catch {
    // The body was not JSON; fall through to the generic message.
  }
  return `Request failed (${response.status})`;
}

function params(index: string, k: number): string {
  return new URLSearchParams({ index, k: String(k) }).toString();
}

export async function getHealth(signal?: AbortSignal): Promise<Health> {
  return parse(await fetch("/api/health", { signal }));
}

export async function searchText(
  query: string,
  index: string,
  k: number,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  const response = await fetch("/api/search/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, index, k }),
    signal,
  });
  return parse(response);
}

export async function searchImage(
  file: File,
  index: string,
  k: number,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(`/api/search/image?${params(index, k)}`, {
    method: "POST",
    body,
    signal,
  });
  return parse(response);
}

export async function searchSimilar(
  id: number,
  index: string,
  k: number,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  return parse(await fetch(`/api/search/similar/${id}?${params(index, k)}`, { signal }));
}

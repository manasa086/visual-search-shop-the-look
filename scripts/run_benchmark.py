"""Compare brute-force search, LSH and HNSW on held-out queries.

Writes results.csv, results.md and two charts to --output-dir.
"""

import argparse
import csv
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from visualsearch.bench.runner import evaluate, exact_neighbors, split_queries  # noqa: E402
from visualsearch.config import EMBEDDINGS_PATH, METADATA_PATH, ROOT  # noqa: E402
from visualsearch.data.caltech256 import read_metadata  # noqa: E402
from visualsearch.embedding.store import load_embeddings  # noqa: E402
from visualsearch.index import BruteForceIndex, HNSWIndex, LSHIndex  # noqa: E402

LSH_TABLES = (4, 8, 16, 32)
LSH_BITS = (10, 14, 18)
LSH_RADII = (0, 1)
HNSW_M = (8, 16, 32)
HNSW_EF_SEARCH = (10, 20, 40, 80, 160)
HNSW_EF_CONSTRUCTION = 200


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    k = args.k

    vectors, _ = load_embeddings(EMBEDDINGS_PATH)
    categories = np.array([record.category for record in read_metadata(METADATA_PATH)])
    index_ids, query_ids = split_queries(len(vectors), args.num_queries, args.seed)
    index_vectors, queries = vectors[index_ids], vectors[query_ids]
    labels, query_labels = categories[index_ids], categories[query_ids]
    print(f"{len(index_vectors)} indexed images, {len(queries)} held-out queries, k={k}")

    truth = exact_neighbors(index_vectors, queries, k)

    def run(name: str, settings: str, index, build: bool = True, **params) -> dict:
        result = {"index": name, "settings": settings, **params}
        result.update(
            evaluate(index, index_vectors, queries, truth, k, labels, query_labels, build)
        )
        print(
            f"{name:<12} {settings:<32} recall={result[f'recall@{k}']:.3f} "
            f"p50={result['p50_ms']:.2f}ms"
        )
        return result

    rows = [run("brute-force", "-", BruteForceIndex())]

    for tables, bits, radius in itertools.product(LSH_TABLES, LSH_BITS, LSH_RADII):
        index = LSHIndex(tables, bits, radius, seed=args.seed)
        settings = f"tables={tables} bits={bits} radius={radius}"
        rows.append(run("lsh", settings, index, tables=tables, bits=bits, radius=radius))

    for m in HNSW_M:
        index = HNSWIndex(M=m, ef_construction=HNSW_EF_CONSTRUCTION, seed=args.seed)
        build_seconds = None  # the graph is built once per M, then searched at several ef values
        for ef in HNSW_EF_SEARCH:
            index.set_ef_search(ef)
            settings = f"M={m} ef_search={ef}"
            row = run("hnsw", settings, index, build=build_seconds is None, M=m, ef_search=ef)
            build_seconds = row["build_seconds"] if build_seconds is None else build_seconds
            row["build_seconds"] = build_seconds
            rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "results.csv")
    write_markdown(rows, args.output_dir / "results.md", k)
    plot(rows, args.output_dir, k)
    print(f"Wrote results to {args.output_dir}")


def write_csv(rows: list[dict], path: Path) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path, k: int) -> None:
    columns = 9
    lines = [
        f"| index | settings | recall@{k} | category precision@{k} | p50 ms | p95 ms "
        "| candidates | memory MB | build s |",
        "|" + "---|" * columns,
    ]
    for row in rows:
        candidates = row["candidate_fraction"]
        lines.append(
            f"| {row['index']} | {row['settings']} | {row[f'recall@{k}']:.3f} "
            f"| {row[f'category_precision@{k}']:.3f} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} "
            f"| {'n/a' if np.isnan(candidates) else f'{candidates:.1%}'} "
            f"| {row['memory_mb']:.0f} | {row['build_seconds']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot(rows: list[dict], output_dir: Path, k: int) -> None:
    brute = rows[0]
    lsh = [row for row in rows if row["index"] == "lsh"]
    hnsw = [row for row in rows if row["index"] == "hnsw"]
    bit_colors = {10: "tab:blue", 14: "tab:orange", 18: "tab:green"}
    radius_markers = {0: "o", 1: "^"}
    recall_key = f"recall@{k}"

    for x_key, x_label, file_name in (
        ("p50_ms", "median query latency (ms)", "recall_vs_latency.png"),
        (
            "candidate_fraction",
            "fraction of the dataset ranked per query",
            "recall_vs_candidates.png",
        ),
    ):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for row in lsh:
            ax.scatter(
                row[x_key],
                row[recall_key],
                c=bit_colors[row["bits"]],
                marker=radius_markers[row["radius"]],
                s=20 + 3 * row["tables"],
                alpha=0.75,
            )
        for m in HNSW_M:
            curve = [row for row in hnsw if row["M"] == m]
            ax.plot(
                [row[x_key] for row in curve],
                [row[recall_key] for row in curve],
                marker="s",
                color="black",
                alpha=0.35 + 0.25 * HNSW_M.index(m),
                label=f"HNSW M={m}" if x_key == "p50_ms" else None,
            )
        ax.axvline(brute[x_key], color="gray", linestyle="--", label="brute force (exact)")
        for bits, color in bit_colors.items():
            ax.scatter([], [], c=color, label=f"LSH {bits} bits")
        for radius, marker in radius_markers.items():
            ax.scatter([], [], c="gray", marker=marker, label=f"LSH radius {radius}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"recall@{k}")
        ax.set_ylim(0, 1.02)
        ax.set_title("Recall vs. cost (LSH marker size grows with the number of tables)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / file_name, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()

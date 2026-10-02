"""Compare brute-force search with LSH settings on held-out queries.

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
from visualsearch.index import BruteForceIndex, LSHIndex  # noqa: E402

TABLES = (4, 8, 16, 32)
BITS = (10, 14, 18)
RADII = (0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    vectors, _ = load_embeddings(EMBEDDINGS_PATH)
    categories = np.array([record.category for record in read_metadata(METADATA_PATH)])
    index_ids, query_ids = split_queries(len(vectors), args.num_queries, args.seed)
    index_vectors, queries = vectors[index_ids], vectors[query_ids]
    labels, query_labels = categories[index_ids], categories[query_ids]
    print(f"{len(index_vectors)} indexed images, {len(queries)} held-out queries, k={args.k}")

    truth = exact_neighbors(index_vectors, queries, args.k)

    def run(name: str, index, **params) -> dict:
        result = {"index": name, **params}
        result.update(evaluate(index, index_vectors, queries, truth, args.k, labels, query_labels))
        print(
            f"{name:<11} {params!s:<48} recall={result[f'recall@{args.k}']:.3f} "
            f"p50={result['p50_ms']:.2f}ms candidates={result['candidate_fraction']:.1%}"
        )
        return result

    rows = [run("brute-force", BruteForceIndex())]
    for tables, bits, radius in itertools.product(TABLES, BITS, RADII):
        index = LSHIndex(tables, bits, radius, seed=args.seed)
        rows.append(run("lsh", index, tables=tables, bits=bits, radius=radius))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "results.csv")
    write_markdown(rows, args.output_dir / "results.md", args.k)
    plot(rows, args.output_dir, args.k)
    print(f"Wrote results to {args.output_dir}")


def write_csv(rows: list[dict], path: Path) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path, k: int) -> None:
    header = (
        f"| index | tables | bits | radius | recall@{k} | category precision@{k} | p50 ms | p95 ms "
        "| candidates | memory MB | build s |"
    )
    lines = [header, "|" + "---|" * 11]
    for row in rows:
        lines.append(
            f"| {row['index']} | {row.get('tables', '')} | {row.get('bits', '')} "
            f"| {row.get('radius', '')} | {row[f'recall@{k}']:.3f} "
            f"| {row[f'category_precision@{k}']:.3f} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} "
            f"| {row['candidate_fraction']:.1%} | {row['memory_mb']:.0f} "
            f"| {row['build_seconds']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot(rows: list[dict], output_dir: Path, k: int) -> None:
    brute = rows[0]
    lsh = rows[1:]
    colors = {10: "tab:blue", 14: "tab:orange", 18: "tab:green"}
    markers = {0: "o", 1: "^"}
    recall_key = f"recall@{k}"

    for x_key, x_label, file_name in (
        ("p50_ms", "median query latency (ms)", "recall_vs_latency.png"),
        (
            "candidate_fraction",
            "fraction of the dataset ranked per query",
            "recall_vs_candidates.png",
        ),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for row in lsh:
            ax.scatter(
                row[x_key],
                row[recall_key],
                c=colors[row["bits"]],
                marker=markers[row["radius"]],
                s=20 + 3 * row["tables"],
                alpha=0.8,
            )
        ax.axvline(brute[x_key], color="gray", linestyle="--", label="brute force (exact)")
        for bits, color in colors.items():
            ax.scatter([], [], c=color, label=f"{bits} bits")
        for radius, marker in markers.items():
            ax.scatter([], [], c="gray", marker=marker, label=f"radius {radius}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"recall@{k}")
        ax.set_ylim(0, 1.02)
        ax.set_title("LSH settings (marker size grows with number of tables)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / file_name, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()

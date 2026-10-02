import type { ResultItem } from "../api";
import { formatCategory } from "../format";

interface Props {
  results: ResultItem[];
  onSelect: (item: ResultItem) => void;
}

export function ResultsGrid({ results, onSelect }: Props) {
  return (
    <ul className="grid">
      {results.map((item) => (
        <li key={item.id}>
          <button
            type="button"
            className="card"
            onClick={() => onSelect(item)}
            aria-label={`${formatCategory(item.category)}, similarity ${item.score.toFixed(2)}. Find more like this.`}
          >
            <img
              src={item.image_url}
              alt=""
              loading="lazy"
              width={item.width}
              height={item.height}
              style={{ aspectRatio: `${item.width} / ${item.height}` }}
            />
            <span className="card-meta">
              <span className="card-category">{formatCategory(item.category)}</span>
              <span className="card-score">{item.score.toFixed(2)}</span>
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}

const SKELETON_HEIGHTS = [180, 240, 150, 280, 200, 170, 260, 190, 230, 160, 250, 210];

export function ResultsSkeleton() {
  return (
    <ul className="grid" aria-hidden="true">
      {SKELETON_HEIGHTS.map((height, position) => (
        <li key={position}>
          <div className="card skeleton" style={{ height }} />
        </li>
      ))}
    </ul>
  );
}

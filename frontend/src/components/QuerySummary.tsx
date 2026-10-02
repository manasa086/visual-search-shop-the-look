import type { ResultItem } from "../api";
import { formatCategory } from "../format";
import { useObjectUrl } from "../useObjectUrl";

export type Query =
  | { kind: "text"; text: string }
  | { kind: "image"; file: File }
  | { kind: "similar"; item: ResultItem };

interface Props {
  query: Query;
  onClear: () => void;
}

export function QuerySummary({ query, onClear }: Props) {
  const uploadUrl = useObjectUrl(query.kind === "image" ? query.file : null);

  let thumbnail: string | null = null;
  let label: string;
  if (query.kind === "text") {
    label = `Results for “${query.text}”`;
  } else if (query.kind === "image") {
    thumbnail = uploadUrl;
    label = "Similar to your photo";
  } else {
    thumbnail = query.item.image_url;
    label = `More like this ${formatCategory(query.item.category)}`;
  }

  return (
    <div className="query-summary">
      {thumbnail && <img src={thumbnail} alt="The image being searched with" />}
      <h2>{label}</h2>
      <button type="button" className="text-button" onClick={onClear}>
        Clear
      </button>
    </div>
  );
}

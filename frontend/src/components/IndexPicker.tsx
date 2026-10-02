interface Props {
  indexes: string[];
  value: string;
  onChange: (index: string) => void;
}

const LABELS: Record<string, { name: string; hint: string }> = {
  hnsw: { name: "HNSW", hint: "fast, approximate" },
  lsh: { name: "LSH", hint: "built from scratch" },
  "brute-force": { name: "Brute force", hint: "exact" },
};

export function indexLabel(index: string): string {
  return LABELS[index]?.name ?? index;
}

export function IndexPicker({ indexes, value, onChange }: Props) {
  return (
    <fieldset className="index-picker">
      <legend>Search method</legend>
      {indexes.map((index) => (
        <label key={index} className={index === value ? "chosen" : undefined}>
          <input
            type="radio"
            name="index"
            value={index}
            checked={index === value}
            onChange={() => onChange(index)}
          />
          <span>{indexLabel(index)}</span>
          {LABELS[index] && <small>{LABELS[index].hint}</small>}
        </label>
      ))}
    </fieldset>
  );
}

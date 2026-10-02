import { useRef, type FormEvent } from "react";

interface Props {
  value: string;
  onChange: (value: string) => void;
  onSubmitText: (text: string) => void;
  onPickFile: (file: File) => void;
  disabled: boolean;
}

export function SearchBar({ value, onChange, onSubmitText, onPickFile, disabled }: Props) {
  const fileInput = useRef<HTMLInputElement>(null);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const text = value.trim();
    if (text) onSubmitText(text);
  };

  return (
    <form className="search-bar" role="search" onSubmit={submit}>
      <label htmlFor="query" className="visually-hidden">
        Describe what you are looking for
      </label>
      <input
        id="query"
        type="search"
        value={value}
        maxLength={200}
        placeholder="Describe an image, or drop a photo anywhere"
        autoComplete="off"
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      />
      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        className="visually-hidden"
        tabIndex={-1}
        aria-label="Upload an image to search with"
        data-testid="file-input"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onPickFile(file);
          event.target.value = ""; // allow picking the same file again
        }}
      />
      <button
        type="button"
        className="icon-button"
        disabled={disabled}
        onClick={() => fileInput.current?.click()}
      >
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <path
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M4 8.5A2.5 2.5 0 0 1 6.5 6h1.2l1.1-1.6A1 1 0 0 1 9.6 4h4.8a1 1 0 0 1 .8.4L16.3 6h1.2A2.5 2.5 0 0 1 20 8.5v8a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 16.5zM12 16a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z"
          />
        </svg>
        <span>Search by photo</span>
      </button>
      <button type="submit" className="primary-button" disabled={disabled || !value.trim()}>
        Search
      </button>
    </form>
  );
}

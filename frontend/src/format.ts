/** 'electric-guitar-101' -> 'electric guitar' (Caltech-256 keeps the '-101' of its Caltech-101 classes). */
export function formatCategory(category: string): string {
  return category.replace(/-101$/, "").replace(/-/g, " ");
}

/** Milliseconds with two decimals for fast operations and none for slow ones. */
export function formatMs(ms: number): string {
  return `${ms < 10 ? ms.toFixed(2) : ms.toFixed(0)} ms`;
}

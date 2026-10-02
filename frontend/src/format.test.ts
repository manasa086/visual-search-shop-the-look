import { describe, expect, it } from "vitest";
import { formatCategory, formatMs } from "./format";

describe("formatCategory", () => {
  it.each([
    ["electric-guitar-101", "electric guitar"],
    ["sunflower-101", "sunflower"],
    ["american-flag", "american flag"],
    ["ak47", "ak47"],
    ["101-dalmatians", "101 dalmatians"],
  ])("%s -> %s", (input, expected) => {
    expect(formatCategory(input)).toBe(expected);
  });
});

describe("formatMs", () => {
  it("keeps two decimals under 10 ms and rounds above", () => {
    expect(formatMs(0.4)).toBe("0.40 ms");
    expect(formatMs(9.994)).toBe("9.99 ms");
    expect(formatMs(12.6)).toBe("13 ms");
  });
});

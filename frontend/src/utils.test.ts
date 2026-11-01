import { describe, expect, it } from "vitest";
import { formatBytes, validatePdfFile } from "./utils.ts";

describe("validatePdfFile", () => {
  it("accepts a normal PDF (case-insensitive extension)", () => {
    expect(validatePdfFile({ name: "Report.PDF", size: 1000 }, 20)).toBeNull();
  });
  it("rejects non-PDF files", () => {
    expect(validatePdfFile({ name: "notes.txt", size: 1000 }, 20)).toMatch(/Unsupported file type/);
    expect(validatePdfFile({ name: "pdf", size: 1000 }, 20)).toMatch(/Unsupported file type/);
  });
  it("rejects empty files", () => {
    expect(validatePdfFile({ name: "a.pdf", size: 0 }, 20)).toMatch(/empty/);
  });
  it("rejects oversized files", () => {
    expect(validatePdfFile({ name: "a.pdf", size: 21 * 1024 * 1024 }, 20)).toMatch(/exceeds the 20 MB limit/);
  });
});

describe("formatBytes", () => {
  it("formats sizes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(3 * 1024 * 1024)).toBe("3.00 MB");
  });
});

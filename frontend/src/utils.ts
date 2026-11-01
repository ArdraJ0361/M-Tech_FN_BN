/** Small pure helpers (unit-tested in utils.test.ts). */

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/** Returns a user-facing error message, or null when the file is acceptable. */
export function validatePdfFile(file: { name: string; size: number }, maxMb: number): string | null {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    return "Unsupported file type. Please select a PDF file (.pdf).";
  }
  if (file.size === 0) {
    return "This file is empty (0 bytes). Please select a different PDF.";
  }
  if (file.size > maxMb * 1024 * 1024) {
    return `This file is ${formatBytes(file.size)}, which exceeds the ${maxMb} MB limit.`;
  }
  return null;
}

export function formatNumber(n: number): string {
  return n.toLocaleString("en-US");
}

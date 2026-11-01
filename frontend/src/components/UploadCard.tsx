import { useRef, useState } from "react";
import { formatBytes, validatePdfFile } from "../utils.ts";

interface Props {
  maxMb: number;
  file: File | null;
  disabled: boolean;
  onFileChange: (file: File | null) => void;
}

/** Step 1: choose a PDF. Nothing is uploaded until the question is submitted in Step 2. */
export function UploadCard({ maxMb, file, disabled, onFileChange }: Props) {
  const [fileError, setFileError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const choose = (candidate: File | undefined) => {
    if (!candidate || disabled) return;
    const problem = validatePdfFile(candidate, maxMb);
    if (problem) {
      onFileChange(null);
      setFileError(problem);
      return;
    }
    onFileChange(candidate);
    setFileError(null);
  };

  return (
    <section className="card" aria-labelledby="step1">
      <h2 id="step1">Step 1: Choose Your Document</h2>

      <div className="notice notice-warn">
        <strong>Text-based PDFs only.</strong> This version reads the text already embedded in a PDF.
        Scanned or image-only PDFs have no extractable text and will be rejected; OCR is not implemented.
      </div>

      <div
        className={`dropzone${dragging ? " dropzone-active" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          choose(e.dataTransfer.files[0]);
        }}
      >
        <div className="dropzone-icon" aria-hidden="true">&#128196;</div>
        <p><strong>Drag and drop a PDF here</strong></p>
        <p className="muted small">or</p>
        <button type="button" className="btn btn-secondary" onClick={() => inputRef.current?.click()} disabled={disabled}>
          Browse Files
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          hidden
          onChange={(e) => {
            choose(e.target.files?.[0]);
            e.target.value = ""; // allow re-selecting the same file
          }}
        />
        <p className="muted small">PDF only, up to {maxMb} MB</p>
      </div>

      {file && (
        <p className="file-info">
          <span aria-hidden="true">&#128206;</span> <strong>{file.name}</strong>
          <span className="muted"> &middot; {formatBytes(file.size)}</span>
        </p>
      )}
      {fileError && <p className="error-text" role="alert">{fileError}</p>}
    </section>
  );
}

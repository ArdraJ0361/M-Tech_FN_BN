import { useState } from "react";
import type { AnswerResponse } from "../types.ts";
import { copyToClipboard } from "../clipboard.ts";

interface Props {
  result: AnswerResponse | null;
  asking: boolean;
}

/** The model's answer. */
export function AnswerCard({ result, asking }: Props) {
  const [copied, setCopied] = useState<string | null>(null);

  if (!result) {
    return (
      <section className="card" aria-labelledby="answer-heading">
        <h2 id="answer-heading">Generated Answer</h2>
        <p className="muted">Waiting for the model...</p>
      </section>
    );
  }

  const copy = async () => {
    if (!result.answer) return;
    const ok = await copyToClipboard(result.answer);
    setCopied(ok ? "Copied to clipboard." : "Copy failed.");
    window.setTimeout(() => setCopied(null), 3000);
  };

  return (
    <section className={`card${asking ? " card-stale" : ""}`} aria-labelledby="answer-heading">
      <h2 id="answer-heading">Generated Answer</h2>
      {asking && <p className="muted small">Showing the previous answer while a new one is generated...</p>}

      <div className="qa-block">
        <div className="label">Question</div>
        <p>{result.question}</p>
      </div>

      <div className="qa-block">
        <div className="label">Answer</div>
        <p className="answer">{result.answer || "(the model produced an empty answer)"}</p>
      </div>

      {result.unanswerable && (
        <div className="notice notice-warn" role="status">
          <strong>No answer found.</strong>{" "}
          {result.message ?? "The model did not find an answer in the supplied document text."}
        </div>
      )}
      {result.document_truncated && (
        <div className="notice notice-warn" role="status">
          <strong>The document was too long for the model.</strong> The model reads only about 1-2
          pages, so only the beginning was used and the rest was cut off. The answer may be wrong.
        </div>
      )}

      <div className="button-row">
        <button type="button" className="btn btn-secondary" onClick={copy} disabled={!result.answer}>
          Copy Answer
        </button>
        {copied && <span className="muted small" role="status">{copied}</span>}
      </div>

      {result.disclaimer && <p className="muted small">{result.disclaimer}</p>}
    </section>
  );
}

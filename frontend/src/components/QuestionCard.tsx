import { MAX_QUESTION_LENGTH } from "../config.ts";

// Suggestions only: clicking one fills the input. No answers are attached to them.
const EXAMPLE_QUESTIONS = [
  "What is the main topic of this document?",
  "What are the key conclusions or findings?",
  "Who is mentioned in the document?",
];

interface Props {
  question: string;
  onQuestionChange: (value: string) => void;
  /** Why submitting is currently impossible (no file, backend offline...), or null when allowed. */
  blockedReason: string | null;
  asking: boolean;
  error: string | null;
  onSubmit: () => void;
}

/** Step 2: type the question and submit. The PDF and the question go to the backend together. */
export function QuestionCard({ question, onQuestionChange, blockedReason, asking, error, onSubmit }: Props) {
  const canSubmit = blockedReason === null && !asking;

  return (
    <section className="card" aria-labelledby="step2">
      <h2 id="step2">Step 2: Ask a Question</h2>

      <label htmlFor="question" className="visually-hidden">Your question</label>
      <textarea
        id="question"
        rows={4}
        value={question}
        maxLength={MAX_QUESTION_LENGTH}
        placeholder="Ask a question about your chosen document..."
        disabled={asking}
        onChange={(e) => onQuestionChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && canSubmit) onSubmit();
        }}
      />

      <div className="chips" aria-label="Example questions">
        <span className="muted small">Try:</span>
        {EXAMPLE_QUESTIONS.map((q) => (
          <button key={q} type="button" className="chip" disabled={asking} onClick={() => onQuestionChange(q)}>
            {q}
          </button>
        ))}
      </div>

      <div className="button-row">
        <button type="button" className="btn btn-primary" disabled={!canSubmit} onClick={onSubmit}>
          {asking ? "Generating..." : "Get Answer"}
        </button>
        {blockedReason && <span className="muted small">{blockedReason}</span>}
        {!blockedReason && <span className="muted small">Tip: Cmd/Ctrl + Enter to submit</span>}
      </div>
      {asking && (
        <div className="status-row" role="status">
          <span className="spinner" aria-hidden="true" />
          Extracting the document structure and generating an answer. This can take a while on a small GPU or CPU...
        </div>
      )}
      {error && <div className="notice notice-error" role="alert"><strong>Could not get an answer.</strong> {error}</div>}
    </section>
  );
}

/** Response shapes returned by the backend (see the backend README, "API reference"). */

export interface HealthResponse {
  status: string;
  model_ready: boolean;
  model_name?: string | null;
  adapter_loaded?: boolean;
  device?: string | null;
  model_error?: string | null;
}

/** Facts measured while extracting the uploaded PDF. */
export interface DocumentInfo {
  filename: string;
  page_count: number;
  heading_count: number;
  table_count: number;
  character_count: number;
  word_count: number;
  empty_page_numbers: number[];
}

export interface TableCell {
  value: string;
  row_label?: string | null;
  column_label?: string | null;
}

/**
 * Where the answer text literally occurs in the extracted document. Found by plain string
 * matching on the backend. It is NOT produced by the model.
 */
export interface AnswerLocation {
  page_number: number;
  element_type: string;
  section_path: string[];
  snippet: string;
  table_cell?: TableCell | null;
}

export interface AnswerResponse {
  question: string;
  answer: string;
  /** The model produced its training label "unanswerable". */
  unanswerable: boolean;
  message?: string | null;
  model_name: string;
  inference_time_seconds: number;
  prompt_tokens: number;
  generated_tokens: number;
  document: DocumentInfo;
  /** The document exceeded the model's input limit and its END was cut. */
  document_truncated: boolean;
  document_chars_used: number;
  document_chars_total: number;
  /** The Markdown that was given to the model (bounded by the backend). */
  structured_text: string;
  structured_text_truncated: boolean;
  answer_locations: AnswerLocation[];
  disclaimer?: string;
}

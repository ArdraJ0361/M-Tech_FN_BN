/**
 * Backend API client. All network access goes through createApi(), which takes the base URL
 * as a parameter (the app passes API_BASE_URL from config.ts). Every failure is converted into
 * an ApiRequestError with a message that is safe to show to the user.
 */

import type { AnswerResponse, HealthResponse } from "./types.ts";

export type ErrorKind = "network" | "timeout" | "http" | "invalid_response";

export class ApiRequestError extends Error {
  kind: ErrorKind;
  status?: number;
  code?: string;

  constructor(message: string, kind: ErrorKind, status?: number, code?: string) {
    super(message);
    this.name = "ApiRequestError";
    this.kind = kind;
    this.status = status;
    this.code = code;
  }
}

type Guard<T> = (value: unknown) => value is T;

const isObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

const isHealth: Guard<HealthResponse> = (v): v is HealthResponse =>
  isObject(v) && typeof v.status === "string" && typeof v.model_ready === "boolean";

const isAnswer: Guard<AnswerResponse> = (v): v is AnswerResponse =>
  isObject(v) &&
  typeof v.question === "string" &&
  typeof v.answer === "string" &&
  typeof v.unanswerable === "boolean" &&
  typeof v.model_name === "string" &&
  typeof v.inference_time_seconds === "number" &&
  typeof v.document_truncated === "boolean" &&
  typeof v.structured_text === "string" &&
  isObject(v.document) &&
  typeof v.document.page_count === "number" &&
  Array.isArray(v.answer_locations);

function extractBackendError(body: unknown): { code: string; message: string } | null {
  if (isObject(body) && isObject(body.error)) {
    const { code, message } = body.error;
    if (typeof code === "string" && typeof message === "string") return { code, message };
  }
  return null;
}

// The backend waits up to 120 s for the model server; allow a little longer before giving up.
export const TIMEOUTS_MS = { health: 10_000, answer: 150_000 };

export function createApi(baseUrl: string, fetchImpl: typeof fetch = (...args) => fetch(...args)) {
  async function request<T>(
    path: string,
    init: RequestInit,
    timeoutMs: number,
    isValid: Guard<T>,
  ): Promise<T> {
    let response: Response;
    try {
      response = await fetchImpl(`${baseUrl}${path}`, {
        ...init,
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch (err) {
      if ((err as { name?: string }).name === "TimeoutError") {
        throw new ApiRequestError(
          "The request timed out. The backend or the model may be busy.",
          "timeout",
        );
      }
      throw new ApiRequestError(
        "Cannot reach the backend. Check that it is running and that VITE_API_BASE_URL is correct.",
        "network",
      );
    }

    let body: unknown;
    try {
      body = await response.json();
    } catch {
      body = undefined;
    }

    if (!response.ok) {
      const backendError = extractBackendError(body);
      if (backendError) {
        throw new ApiRequestError(backendError.message, "http", response.status, backendError.code);
      }
      throw new ApiRequestError(
        `The backend returned an unexpected response (HTTP ${response.status}).`,
        "invalid_response",
        response.status,
      );
    }
    if (!isValid(body)) {
      throw new ApiRequestError(
        "The backend returned a response in an unexpected format.",
        "invalid_response",
        response.status,
      );
    }
    return body;
  }

  return {
    health: () => request("/health", { method: "GET" }, TIMEOUTS_MS.health, isHealth),

    /** One request: the PDF and the question together. The server stores nothing. */
    askAboutDocument: (file: File, question: string) => {
      const form = new FormData();
      form.append("file", file); // field names must be "file" and "question"
      form.append("question", question);
      // Do not set Content-Type: the browser adds the multipart boundary itself.
      return request("/answer", { method: "POST", body: form }, TIMEOUTS_MS.answer, isAnswer);
    },
  };
}

export type Api = ReturnType<typeof createApi>;

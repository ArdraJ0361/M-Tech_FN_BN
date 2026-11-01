import { describe, expect, it } from "vitest";
import { createApi, ApiRequestError } from "./api.ts";

const BASE = "http://backend.test";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const answer = {
  question: "What was the profit?",
  answer: "30",
  unanswerable: false,
  message: null,
  model_name: "test-model",
  inference_time_seconds: 1.2,
  prompt_tokens: 100,
  generated_tokens: 2,
  document: {
    filename: "x.pdf",
    page_count: 2,
    heading_count: 3,
    table_count: 1,
    character_count: 300,
    word_count: 50,
    empty_page_numbers: [],
  },
  document_truncated: false,
  document_chars_used: 300,
  document_chars_total: 300,
  structured_text: "# Title",
  structured_text_truncated: false,
  answer_locations: [],
};

async function failureOf(p: Promise<unknown>): Promise<ApiRequestError> {
  try {
    await p;
  } catch (e) {
    if (e instanceof ApiRequestError) return e;
    throw e;
  }
  throw new Error("expected the request to fail");
}

describe("api client", () => {
  it("parses a valid health response and calls the right URL", async () => {
    let seen = "";
    const api = createApi(BASE, async (url) => {
      seen = String(url);
      return jsonResponse({ status: "ok", model_ready: true });
    });
    expect((await api.health()).model_ready).toBe(true);
    expect(seen).toBe(`${BASE}/health`);
  });

  it("sends the PDF and the question in ONE multipart request to /answer", async () => {
    let url = "";
    let init: RequestInit | undefined;
    const api = createApi(BASE, async (u, i) => {
      url = String(u);
      init = i;
      return jsonResponse(answer);
    });
    const file = new File([new Uint8Array([37, 80, 68, 70])], "x.pdf", { type: "application/pdf" });
    const result = await api.askAboutDocument(file, "What was the profit?");
    expect(result.answer).toBe("30");
    expect(url).toBe(`${BASE}/answer`);
    expect(init?.method).toBe("POST");
    const form = init?.body as FormData;
    expect(form.get("file")).toBeInstanceOf(File);
    expect(form.get("question")).toBe("What was the profit?");
    expect(init?.headers).toBeUndefined(); // the browser sets the multipart boundary
  });

  it("reports a network failure clearly", async () => {
    const api = createApi(BASE, async () => {
      throw new TypeError("Failed to fetch");
    });
    const err = await failureOf(api.health());
    expect(err.kind).toBe("network");
    expect(err.message).toMatch(/Cannot reach the backend/);
  });

  it("reports a timeout", async () => {
    const api = createApi(BASE, async () => {
      throw new DOMException("timed out", "TimeoutError");
    });
    expect((await failureOf(api.health())).kind).toBe("timeout");
  });

  it("surfaces the backend's structured error message and code", async () => {
    const api = createApi(BASE, async () =>
      jsonResponse({ error: { code: "malformed_pdf", message: "The PDF is corrupt." } }, 422),
    );
    const err = await failureOf(api.askAboutDocument(new File(["x"], "x.pdf"), "q?"));
    expect(err).toMatchObject({ kind: "http", status: 422, code: "malformed_pdf", message: "The PDF is corrupt." });
  });

  it("handles non-JSON error pages (e.g. a stopped tunnel)", async () => {
    const api = createApi(BASE, async () => new Response("<html>Bad gateway</html>", { status: 502 }));
    const err = await failureOf(api.health());
    expect(err.kind).toBe("invalid_response");
    expect(err.message).toMatch(/HTTP 502/);
  });

  it("rejects a 200 response with an unexpected shape", async () => {
    const api = createApi(BASE, async () => jsonResponse({ hello: "world" }));
    expect((await failureOf(api.health())).kind).toBe("invalid_response");
    const api2 = createApi(BASE, async () => jsonResponse({ answer: "x" }));
    expect((await failureOf(api2.askAboutDocument(new File(["x"], "x.pdf"), "q?"))).kind).toBe(
      "invalid_response",
    );
  });
});

import { describe, expect, it } from "vitest";
import { submitBlockedReason, type BackendState } from "./App.tsx";

const ready: BackendState = {
  kind: "online",
  health: { status: "ok", model_ready: true, model_name: "m" },
};

describe("submitBlockedReason", () => {
  it("allows submission when file, question and model are all ready", () => {
    expect(submitBlockedReason(true, "What is X?", ready)).toBeNull();
  });

  it("requires a file", () => {
    expect(submitBlockedReason(false, "What is X?", ready)).toMatch(/PDF/);
  });

  it("requires a non-empty question", () => {
    expect(submitBlockedReason(true, "", ready)).toMatch(/question/i);
  });

  it("is blocked while the backend is offline or checking", () => {
    expect(submitBlockedReason(true, "q", { kind: "offline", message: "x" })).toMatch(/unreachable/);
    expect(submitBlockedReason(true, "q", { kind: "checking" })).toMatch(/Checking/);
  });

  it("is blocked, with the reason, when the model is not ready", () => {
    const notReady: BackendState = {
      kind: "online",
      health: { status: "ok", model_ready: false, model_error: "adapter missing" },
    };
    expect(submitBlockedReason(true, "q", notReady)).toContain("adapter missing");
  });
});

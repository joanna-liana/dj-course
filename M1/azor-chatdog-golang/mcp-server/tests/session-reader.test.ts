import { describe, it } from "node:test";
import assert from "node:assert";
import { SessionReader } from "../src/session-reader.js";
import { CONFIG } from "../src/config.js";
import { join } from "path";

describe("SessionReader", () => {
  it("lists session IDs from fixtures", async () => {
    const originalDir = CONFIG.sessionDir;
    (CONFIG as any).sessionDir = join(process.cwd(), "tests", "fixtures");

    const reader = new SessionReader();
    const sessions = await reader.listSessionIds();

    assert.ok(sessions.includes("test-session-1"));
    assert.ok(sessions.includes("test-session-2"));
    (CONFIG as any).sessionDir = originalDir;
  });

  it("reads session log data", async () => {
    const originalDir = CONFIG.sessionDir;
    (CONFIG as any).sessionDir = join(process.cwd(), "tests", "fixtures");

    const reader = new SessionReader();
    const data = await reader.readSessionLog("test-session-1");

    assert.strictEqual(data.session_id, "test-session-1");
    assert.strictEqual(data.model, "gemini-2.5-flash");
    assert.strictEqual(data.history.length, 2);

    (CONFIG as any).sessionDir = originalDir;
  });

  it("gets session metadata", async () => {
    const originalDir = CONFIG.sessionDir;
    (CONFIG as any).sessionDir = join(process.cwd(), "tests", "fixtures");

    const reader = new SessionReader();
    const metadata = await reader.getSessionMetadata("test-session-1");

    assert.strictEqual(metadata.id, "test-session-1");
    assert.strictEqual(metadata.messagesCount, 2);
    assert.strictEqual(metadata.model, "gemini-2.5-flash");

    (CONFIG as any).sessionDir = originalDir;
  });

  it("gets session summary with token count", async () => {
    const originalDir = CONFIG.sessionDir;
    (CONFIG as any).sessionDir = join(process.cwd(), "tests", "fixtures");

    const reader = new SessionReader();
    const summary = await reader.getSessionSummary("test-session-2");

    assert.strictEqual(summary.id, "test-session-2");
    assert.strictEqual(summary.messagesCount, 4);
    assert.strictEqual(summary.totalTokens, 195);
    assert.ok(summary.firstMessage?.startsWith("What is AI?"));

    (CONFIG as any).sessionDir = originalDir;
  });
});

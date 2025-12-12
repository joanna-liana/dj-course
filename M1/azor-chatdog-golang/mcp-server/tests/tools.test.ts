import { describe, it } from "node:test";
import assert from "node:assert";
import { SessionReader } from "../src/session-reader.js";
import { ToolHandlers } from "../src/tools.js";
import { CONFIG } from "../src/config.js";
import { join } from "path";

describe("ToolHandlers", () => {
  it("lists available tools", async () => {
    const originalDir = CONFIG.sessionDir;
    (CONFIG as any).sessionDir = join(process.cwd(), "tests", "fixtures");

    const reader = new SessionReader();
    const handlers = new ToolHandlers(reader);
    const result = await handlers.listTools();

    assert.ok(result.tools.length >= 2);
    assert.ok(result.tools.some((t) => t.name === "list_sessions"));
    assert.ok(result.tools.some((t) => t.name === "get_session"));

    (CONFIG as any).sessionDir = originalDir;
  });

  it("calls list_sessions tool", async () => {
    const originalDir = CONFIG.sessionDir;
    (CONFIG as any).sessionDir = join(process.cwd(), "tests", "fixtures");

    const reader = new SessionReader();
    const handlers = new ToolHandlers(reader);
    const result = await handlers.callTool({
      params: {
        name: "list_sessions",
        arguments: {},
      },
      method: "tools/call",
    } as any);

    assert.strictEqual(result.content.length, 1);
    assert.strictEqual(result.content[0].type, "text");
    const text = result.content[0].text;
    assert.ok(text.includes("Session: test-session-1"));
    assert.ok(text.includes("Session: test-session-2"));
    assert.ok(text.includes("Last Update:"));
    assert.ok(text.includes("Messages:"));
    assert.ok(text.includes("Model:"));

    (CONFIG as any).sessionDir = originalDir;
  });

  it("calls get_session tool", async () => {
    const originalDir = CONFIG.sessionDir;
    (CONFIG as any).sessionDir = join(process.cwd(), "tests", "fixtures");

    const reader = new SessionReader();
    const handlers = new ToolHandlers(reader);
    const result = await handlers.callTool({
      params: {
        name: "get_session",
        arguments: { session_id: "test-session-1" },
      },
      method: "tools/call",
    } as any);

    assert.strictEqual(result.content.length, 1);
    assert.strictEqual(result.content[0].type, "text");
    const content = JSON.parse(result.content[0].text);
    assert.strictEqual(content.summary.id, "test-session-1");
    assert.strictEqual(content.full_conversation.session_id, "test-session-1");

    (CONFIG as any).sessionDir = originalDir;
  });
});

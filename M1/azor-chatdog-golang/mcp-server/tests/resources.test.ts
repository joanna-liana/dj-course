import { describe, it } from "node:test";
import assert from "node:assert";
import { SessionReader } from "../src/session-reader.js";
import { ResourceHandlers } from "../src/resources.js";
import { CONFIG } from "../src/config.js";
import { join } from "path";

describe("ResourceHandlers", () => {
  it("lists resources as MCP format", async () => {
    const originalDir = CONFIG.sessionDir;
    (CONFIG as any).sessionDir = join(process.cwd(), "tests", "fixtures");

    const reader = new SessionReader();
    const handlers = new ResourceHandlers(reader);
    const result = await handlers.listResources();

    assert.ok(result.resources.length >= 2);
    assert.ok(result.resources[0].uri.startsWith("azor://session/"));
    assert.strictEqual(result.resources[0].mimeType, "application/json");

    (CONFIG as any).sessionDir = originalDir;
  });

  it("reads resource content", async () => {
    const originalDir = CONFIG.sessionDir;
    (CONFIG as any).sessionDir = join(process.cwd(), "tests", "fixtures");

    const reader = new SessionReader();
    const handlers = new ResourceHandlers(reader);
    const result = await handlers.readResource({
      params: { uri: "azor://session/test-session-1" },
      method: "resources/read",
    } as any);

    assert.strictEqual(result.contents.length, 1);
    const content = JSON.parse(result.contents[0].text);
    assert.strictEqual(content.summary.id, "test-session-1");
    assert.strictEqual(content.full_conversation.session_id, "test-session-1");

    (CONFIG as any).sessionDir = originalDir;
  });

  it("reads list resource", async () => {
    const originalDir = CONFIG.sessionDir;
    (CONFIG as any).sessionDir = join(process.cwd(), "tests", "fixtures");

    const reader = new SessionReader();
    const handlers = new ResourceHandlers(reader);
    const result = await handlers.readResource({
      params: { uri: "azor://session/list" },
      method: "resources/read",
    } as any);

    assert.strictEqual(result.contents.length, 1);
    const content = JSON.parse(result.contents[0].text);
    assert.ok(Array.isArray(content));
    assert.ok(content.length >= 2);

    (CONFIG as any).sessionDir = originalDir;
  });
});

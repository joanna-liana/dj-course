import { readdir, readFile, unlink } from "fs/promises";
import { join } from "path";
import { CONFIG } from "./config.js";
import type {
  SessionLogData,
  SessionMetadata,
  WALEntry,
  SessionSummary,
} from "./types.js";

export class SessionReader {
  async listSessionIds(): Promise<string[]> {
    const files = await readdir(CONFIG.sessionDir);
    const sessionIds = files
      .filter((f) => f.endsWith("-log.json") && f !== CONFIG.walFilename)
      .map((f) => f.replace("-log.json", ""))
      .sort();
    return sessionIds;
  }

  async readSessionLog(sessionId: string): Promise<SessionLogData> {
    const filePath = join(CONFIG.sessionDir, `${sessionId}-log.json`);
    const content = await readFile(filePath, "utf-8");
    return JSON.parse(content) as SessionLogData;
  }

  async readWAL(): Promise<WALEntry[]> {
    const filePath = join(CONFIG.sessionDir, CONFIG.walFilename);
    try {
      const content = await readFile(filePath, "utf-8");
      return JSON.parse(content) as WALEntry[];
    } catch {
      return [];
    }
  }

  async getSessionMetadata(sessionId: string): Promise<SessionMetadata> {
    try {
      const sessionData = await this.readSessionLog(sessionId);
      const lastActivity =
        sessionData.history.length > 0
          ? new Date(
              sessionData.history[sessionData.history.length - 1].timestamp
            ).toISOString()
          : "No activity";

      return {
        id: sessionId,
        messagesCount: sessionData.history.length,
        lastActivity,
        model: sessionData.model,
      };
    } catch (error) {
      return {
        id: sessionId,
        messagesCount: 0,
        lastActivity: "Error",
        model: "unknown",
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  async getSessionSummary(sessionId: string): Promise<SessionSummary> {
    const metadata = await this.getSessionMetadata(sessionId);
    const sessionData = await this.readSessionLog(sessionId);
    const walEntries = await this.readWAL();

    const sessionWALEntries = walEntries.filter(
      (e) => e.session_id === sessionId
    );
    const totalTokens = sessionWALEntries.reduce(
      (sum, e) => sum + e.tokens_used,
      0
    );

    const userMessages = sessionData.history.filter((h) => h.role === "user");
    const firstMessage =
      userMessages.length > 0 ? userMessages[0].text.slice(0, 100) : undefined;
    const lastMessage =
      userMessages.length > 0
        ? userMessages[userMessages.length - 1].text.slice(0, 100)
        : undefined;

    return {
      ...metadata,
      totalTokens,
      firstMessage,
      lastMessage,
    };
  }

  async listAllSessions(): Promise<SessionMetadata[]> {
    const sessionIds = await this.listSessionIds();
    const metadataPromises = sessionIds.map((id) =>
      this.getSessionMetadata(id)
    );
    return Promise.all(metadataPromises);
  }

  async deleteSession(sessionId: string): Promise<void> {
    const filePath = join(CONFIG.sessionDir, `${sessionId}-log.json`);
    await unlink(filePath);
  }
}

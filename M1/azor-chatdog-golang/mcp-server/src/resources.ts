import { SessionReader } from "./session-reader.js";
import type { ReadResourceRequest } from "@modelcontextprotocol/sdk/types.js";

export class ResourceHandlers {
  constructor(private reader: SessionReader) {}

  async listResources() {
    const sessions = await this.reader.listAllSessions();

    return {
      resources: sessions.map((s) => ({
        uri: `azor://session/${s.id}`,
        name: `Session ${s.id.slice(0, 8)}`,
        description: `${s.messagesCount} messages, last: ${s.lastActivity}, model: ${s.model}`,
        mimeType: "application/json",
      })),
    };
  }

  async readResource(request: ReadResourceRequest) {
    const uri = request.params.uri;
    const match = uri.match(/^azor:\/\/session\/(.+)$/);

    if (!match) {
      throw new Error(`Invalid resource URI: ${uri}`);
    }

    const sessionId = match[1];

    if (sessionId === "list") {
      const sessions = await this.reader.listAllSessions();
      return {
        contents: [
          {
            uri,
            mimeType: "application/json",
            text: JSON.stringify(sessions, null, 2),
          },
        ],
      };
    }

    const summary = await this.reader.getSessionSummary(sessionId);
    const sessionData = await this.reader.readSessionLog(sessionId);

    return {
      contents: [
        {
          uri,
          mimeType: "application/json",
          text: JSON.stringify(
            {
              summary,
              full_conversation: sessionData,
            },
            null,
            2
          ),
        },
      ],
    };
  }
}

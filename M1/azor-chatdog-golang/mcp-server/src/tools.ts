import { SessionReader } from "./session-reader.js";
import type { CallToolRequest } from "@modelcontextprotocol/sdk/types.js";

export class ToolHandlers {
  constructor(private reader: SessionReader) {}

  async listTools() {
    return {
      tools: [
        {
          name: "list_sessions",
          description:
            "List all Azor Chatdog sessions with metadata (ID, message count, last activity, model)",
          inputSchema: {
            type: "object",
            properties: {},
            required: [],
          },
        },
        {
          name: "get_session",
          description:
            "Get full conversation history and details for a specific session",
          inputSchema: {
            type: "object",
            properties: {
              session_id: {
                type: "string",
                description: "The session ID to retrieve",
              },
            },
            required: ["session_id"],
          },
        },
        {
          name: "delete_session",
          description:
            "Delete a session from the chatdog storage (WARNING: This permanently removes the session file)",
          inputSchema: {
            type: "object",
            properties: {
              session_id: {
                type: "string",
                description: "The session ID to delete",
              },
            },
            required: ["session_id"],
          },
        },
      ],
    };
  }

  async callTool(request: CallToolRequest) {
    const { name, arguments: args } = request.params;

    switch (name) {
      case "list_sessions": {
        const sessions = await this.reader.listAllSessions();

        const formattedText = sessions
          .map((s) => {
            const lastUpdate = s.lastActivity !== "No activity" && s.lastActivity !== "Error"
              ? new Date(s.lastActivity).toLocaleString()
              : s.lastActivity;

            return `Session: ${s.id}
  Last Update: ${lastUpdate}
  Messages: ${s.messagesCount}
  Model: ${s.model}`;
          })
          .join("\n\n");

        return {
          content: [
            {
              type: "text" as const,
              text: formattedText || "No sessions found",
            },
          ],
        };
      }

      case "get_session": {
        const sessionId = (args as any).session_id;
        if (!sessionId) {
          throw new Error("session_id is required");
        }

        const summary = await this.reader.getSessionSummary(sessionId);
        const sessionData = await this.reader.readSessionLog(sessionId);

        return {
          content: [
            {
              type: "text" as const,
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

      case "delete_session": {
        const sessionId = (args as any).session_id;
        if (!sessionId) {
          throw new Error("session_id is required");
        }

        await this.reader.deleteSession(sessionId);

        return {
          content: [
            {
              type: "text" as const,
              text: `Session ${sessionId} has been deleted successfully`,
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }
}

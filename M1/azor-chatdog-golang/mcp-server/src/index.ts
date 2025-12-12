#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { SessionReader } from "./session-reader.js";
import { ResourceHandlers } from "./resources.js";

const server = new Server(
  {
    name: "azor-chatdog-mcp-server",
    version: "0.1.0",
  },
  {
    capabilities: {
      resources: {},
    },
  }
);

const sessionReader = new SessionReader();
const resourceHandlers = new ResourceHandlers(sessionReader);

server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return resourceHandlers.listResources();
});

server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  return resourceHandlers.readResource(request);
});

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Azor Chatdog MCP server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error in MCP server:", error);
  process.exit(1);
});

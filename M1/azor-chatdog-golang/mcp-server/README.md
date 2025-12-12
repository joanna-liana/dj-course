# Azor Chatdog MCP Server

Model Context Protocol server for Azor Chatdog session management.

## Features

- Lists all Azor Chatdog sessions as MCP resources
- Exposes session conversation history
- Read-only access to session metadata (messages count, timestamps, model)
- Token usage tracking from WAL

## Installation

```bash
cd mcp-server
npm install
npm run build
```

## Usage

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or equivalent:

```json
{
  "mcpServers": {
    "azor-chatdog": {
      "command": "node",
      "args": ["/absolute/path/to/azor-chatdog-golang/mcp-server/build/src/index.js"]
    }
  }
}
```

Restart Claude Desktop. Sessions will appear in the resource picker.

### Manual Testing

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"resources/list"}' | node build/src/index.js
```

## Development

```bash
npm run dev        # Watch mode
npm test           # Run tests
npm run lint       # Lint code
```

## Architecture

- **Standalone server**: No dependency on Go application
- **File-based**: Reads directly from `~/.azor/`
- **Stdio transport**: Standard MCP protocol
- **Read-only**: No mutations to session files

## Resource URIs

- `azor://session/list` - All sessions metadata
- `azor://session/{session-id}` - Full conversation history

## MCP Tools

**list_sessions**
- Lists all sessions with last update date, message count, and model
- No arguments required
- Returns formatted text output

**get_session**
- Retrieves full conversation history for a specific session
- Arguments: `session_id` (string)
- Returns JSON with summary and full conversation

## Session Directory

Default: `~/.azor/`

Override with environment variable:
```bash
AZOR_SESSION_DIR=/custom/path node build/src/index.js
```

## Testing

Tests use Node.js built-in test runner and fixture data in `tests/fixtures/`.

```bash
npm test
```

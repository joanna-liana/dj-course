# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AZOR the CHATDOG is a CLI chat application written in Go, ported 1:1 from Python. Supports Google Gemini API and local LLaMA models with persistent session management.

## Development Commands

### Task (Recommended)
```bash
task install          # Install dependencies
task env              # Create .env from template (one-time)
task build            # Compile binary
task run              # Build and run
task sessions         # List saved sessions
task clean            # Remove binary
```

### Without Task
```bash
go mod download       # Install dependencies
go build -o azor-chatdog .
./azor-chatdog        # Run
./azor-chatdog --session-id=<ID>  # Resume session
```

### Testing
```bash
go test ./...         # Run all tests
go test -v ./session  # Run specific package tests
go test -run TestName # Run specific test
```

## Configuration

### Environment Setup
Create `.env` file (from `.env.example`):

**Google Gemini (default):**
```bash
ENGINE=GEMINI
GEMINI_API_KEY=your_api_key_here
MODEL_NAME=gemini-2.5-flash
```

**Local LLaMA (requires llama.cpp):**
```bash
ENGINE=LLAMA_CPP
LLAMA_MODEL_NAME=llama-3.1-8b-instruct
LLAMA_MODEL_PATH=/path/to/model.gguf
LLAMA_GPU_LAYERS=1
LLAMA_CONTEXT_SIZE=2048
```

LLaMA support requires llama.cpp installation and C bindings (currently stub implementation).

**Cerebras Inference (cloud):**
```bash
ENGINE=CEREBRAS
CEREBRAS_API_KEY=your_cerebras_api_key_here
CEREBRAS_MODEL_NAME=llama-3.3-70b
```

Get free API key: https://cloud.cerebras.ai

## Architecture

### Core Components

**Module Structure:**
- `main.go` - Entry point with signal handling
- `chat.go` - Main chat loop (`InitChat`, `MainLoop`, `Cleanup`)
- `command_handler.go` - Slash command dispatcher

**Packages:**
- `assistant/` - Assistant configuration (Azor persona)
- `llm/` - LLM client abstraction (Gemini, LLaMA)
  - `types.go` - Universal interfaces (`LLMClient`, `ChatSession`, `Message`)
  - `gemini_client.go` - Google Gemini implementation
  - `llama_client.go` - LLaMA stub (needs llama.cpp bindings)
- `session/` - Session persistence and management
  - `chat_session.go` - Individual chat session logic
  - `session_manager.go` - Session lifecycle (create, switch, load, save)
- `cli/` - User interface
  - `console.go` - Colorized terminal output
  - `prompt.go` - Interactive input with tab completion
  - `args.go` - CLI argument parsing
- `files/` - File I/O
  - `config.go` - Environment variable loading
  - `session_files.go` - Session persistence to JSON
  - `wal.go` - Write-Ahead Log for all transactions
- `commands/` - Slash command handlers
  - `welcome.go`, `session_list.go`, `session_display.go`, etc.

### Data Flow

1. `main.go` → `InitChat()` → Load/create session
2. `MainLoop()` → Read user input
3. Slash command → `HandleCommand()` → `commands/` handlers
4. Regular message → `session.SendMessage()` → LLM client → Save
5. `Cleanup()` → Final session save

### Session Storage

Sessions saved to `~/.azor/`:
- `<session-id>-log.json` - Conversation history
- `azor-wal.json` - Write-Ahead Log (all transactions)

## Key Features

- **Tab Completion:** Type `/` + Tab for command suggestions and session IDs
- **Multi-engine support:** Gemini (full) and LLaMA (stub)
- **Session persistence:** Resume conversations with `--session-id`
- **Token tracking:** Display token usage per message
- **Colorized output:** Terminal colors for better UX

## Slash Commands

```
/exit             - Exit chat
/quit             - Exit chat
/help             - Show help
/switch <ID>      - Switch to another session
/session list     - List all sessions
/session display  - Show full history
/session pop      - Remove last exchange
/session clear    - Clear history
/session new      - Start new session
/session remove   - Delete current session
/pdf              - Export to PDF (TODO)
```

## Implementation Status

**Complete:**
- Google Gemini API integration
- Session management (create, load, save, switch, delete)
- Conversation history and WAL
- Tab completion for commands
- Token counting
- Colorized terminal output

**Stub/Incomplete:**
- LLaMA client (needs llama.cpp C bindings)
- PDF export

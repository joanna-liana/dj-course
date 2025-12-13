# Clarifying Questions Feature - Implementation Plan

## Goal
Enable AZOR to automatically detect unclear prompts, ask clarifying questions, and present multiple-choice options selectable via arrow keys.

## User Requirements
- **Automatic detection**: AZOR decides when clarification is needed
- **Free-form bypass**: Users can press Escape to skip selection and type custom response
- **Numbered list format**: Options formatted as `1.`, `2.`, `3.` (standard markdown)
- **Frequency limit**: Max 1-2 clarifying questions per user message

## Architecture Overview

**Flow:**
1. User sends message → LLM processes
2. LLM response contains clarifying question with numbered options
3. Parser detects options in response text
4. Display question + interactive selector with arrow keys
5. User selects option OR presses Escape to type custom response
6. Selected/typed response sent back to LLM as context

**Key Components:**
- **Response Parser**: Detect and extract numbered options from LLM text
- **Option Selector**: Reusable arrow key navigation component
- **System Prompt**: Guidelines for AZOR on when/how to ask clarifications
- **Chat Flow Integration**: Hook into `MainLoop()` to intercept clarifying responses

---

## Iteration 1: Response Parser

**Goal:** Detect clarifying questions with numbered options in LLM responses

**Files to create:**
- `clarify/parser.go` - Response parsing logic

**Implementation:**
```go
type ClarifyingQuestion struct {
    Question string   // Text before options
    Options  []string // Extracted option texts
}

func ParseResponse(text string) (*ClarifyingQuestion, bool)
```

**Logic:**
- Detect numbered list pattern: `1.`, `2.`, `3.` on separate lines
- Extract question text (everything before first numbered item)
- Extract option texts (strip numbers and whitespace)
- Return `nil, false` if no valid options found
- Minimum 2 options required

**Tests:**
- Parse valid numbered list
- Reject bullet lists without numbers
- Handle edge cases (empty lines, formatting variations)
- Return false for regular responses without options

**Validation:**
- Test with sample LLM responses containing options
- Ensure false positives don't occur on regular numbered lists in explanations

---

## Iteration 2: Generic Option Selector

**Goal:** Reusable arrow key navigation component

**Files to create:**
- `cli/selector.go` - Interactive option selector

**Implementation:**
```go
func SelectOption(prompt string, options []string) (selected string, ok bool)
```

**Features:**
- Arrow keys (↑/↓) to navigate options
- Enter to select
- Escape to cancel (returns "", false)
- Visual highlight with `→` indicator
- Clear screen + re-render on each key press

**Reuse from `commands/session_list.go`:**
- `readKey()` function for ANSI escape sequences
- Display pattern with colored highlight
- Terminal raw mode handling via `mattn/go-tty`

**Enhancements over session list:**
- Generic `[]string` input (not session-specific)
- Custom prompt text displayed above options
- Return empty string on Escape (free-form bypass)

**Tests:**
- Arrow key navigation (up wraps to bottom, down wraps to top)
- Enter returns selected option
- Escape returns empty string
- Display formatting

---

## Iteration 3: System Prompt Enhancement

**Goal:** Teach AZOR when and how to ask clarifying questions

**Files to modify:**
- `assistant/azor.go` - Update system instruction (line 6)

**Guidelines to add:**

```
Clarifying Questions:
- If user prompt is ambiguous or lacks context, ask ONE clarifying question
- Format options as numbered list (1., 2., 3.) on separate lines
- Include 2-4 options maximum
- Keep option text concise (one line each)
- Example format:

  I need clarification. What did you mean?

  1. Option one description
  2. Option two description
  3. Option three description

- Ask maximum ONE clarifying question per user message
- After user selects/provides answer, proceed with full response
- Don't ask clarification for clear, specific prompts
```

**Testing:**
- Test with deliberately vague prompts (e.g., "fix it", "what about that?")
- Verify AZOR formats options correctly
- Verify AZOR doesn't over-clarify on clear prompts

---

## Iteration 4: Chat Flow Integration

**Goal:** Intercept clarifying responses and trigger selector

**Files to modify:**
- `chat.go` - `MainLoop()` function (lines 44-102)

**Changes:**

1. **After receiving LLM response** (after line 84):
   ```go
   response, err := sess.SendMessage(userInput)

   // Check for clarifying question
   if clarifyQ, hasClarification := clarify.ParseResponse(response.Text); hasClarification {
       // Display question part only
       cli.PrintAssistant(clarifyQ.Question)

       // Show interactive selector
       selected, ok := cli.SelectOption("Select an option (Escape to skip):", clarifyQ.Options)

       if !ok {
           // User pressed Escape - get free-form input
           cli.PrintInfo("\nProvide your own answer:")
           selected = cli.GetUserInput("TY: ")
       }

       // Send selection back to LLM as follow-up
       response, err = sess.SendMessage(selected)
   }

   // Display final response (line 94)
   cli.PrintAssistant(response.Text)
   ```

2. **Clarification counter** (prevent loops):
   - Track clarifications in current exchange
   - Max 2 clarifying questions per user message
   - Display error if limit exceeded: "Too many clarifications. Please rephrase your question."

**Edge cases:**
- Handle EOF during selector (Ctrl+D)
- Handle errors during follow-up LLM call
- Ensure token counting includes both LLM calls
- Save session after clarification exchange

**Tests:**
- Successful option selection flow
- Escape to free-form flow
- Clarification limit enforcement
- Session persistence after clarification

---

## Iteration 5: Polish & Documentation

**Goal:** Refinements and user-facing docs

**Tasks:**

1. **Add slash command** (optional convenience):
   - `/clarify on|off` - Toggle auto-clarification
   - Store preference in session metadata

2. **Error handling:**
   - Graceful fallback if selector crashes
   - Show raw response if parsing fails unexpectedly

3. **Visual improvements:**
   - Color-code option numbers (cyan)
   - Show "Escape to type your own" hint in selector
   - Add subtle separator between question and options

4. **Documentation:**
   - Update `CLAUDE.md` with clarification feature
   - Add example to `/help` command output
   - Document markup format for contributors

5. **Testing:**
   - Integration test: unclear prompt → options → selection → final answer
   - Test all three LLM engines (Gemini, Cerebras, LLaMA)
   - Manual testing with real-world vague prompts

---

## Critical Files

**New files:**
- `clarify/parser.go` - Response parsing logic
- `clarify/parser_test.go` - Parser tests
- `cli/selector.go` - Generic option selector
- `cli/selector_test.go` - Selector tests

**Modified files:**
- `chat.go:44-102` - MainLoop integration
- `assistant/azor.go:6` - System instruction update
- `cli/console.go` - Update DisplayHelp with clarification info (iteration 5)
- `CLAUDE.md` - Feature documentation (iteration 5)

**Reused code:**
- `commands/session_list.go:29-63` - `readKey()` function (extract to `cli/`)
- `commands/session_list.go:69-83` - Display pattern

---

## Testing Strategy

**Unit tests:**
- `clarify/parser_test.go` - Option extraction, edge cases
- `cli/selector_test.go` - Key handling (mocked terminal)

**Integration tests:**
- `chat_test.go` - End-to-end clarification flow
- Test with real Gemini API responses

**Manual testing scenarios:**
1. Vague prompt: "help me with that thing"
2. Ambiguous request: "fix the bug"
3. Missing context: "what about the other option?"
4. Clear prompt: "list all sessions" (should NOT trigger clarification)
5. Escape flow: Select → Escape → Type custom answer
6. Multi-round: Ask clarification → Select → Ask follow-up clarification (test limit)

---

## Rollout Plan

**Phase 1 (MVP):**
- Iterations 1-4
- Basic clarification with option selection
- Free-form bypass via Escape

**Phase 2 (Polish):**
- Iteration 5
- Slash command toggle
- Improved visuals
- Comprehensive testing

---

## Success Criteria

✅ AZOR detects unclear prompts and asks clarifying questions
✅ Options presented in numbered list format
✅ Arrow keys navigate options smoothly
✅ Enter selects, Escape allows free-form input
✅ Selected answer sent back to LLM for final response
✅ Maximum 2 clarifications per user message enforced
✅ No false positives on clear prompts
✅ Session persistence works correctly with clarifications
✅ All tests pass

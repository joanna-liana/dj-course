package clarify

import (
	"testing"
)

func TestParseResponse_ValidNumberedList(t *testing.T) {
	text := `I need clarification. What did you mean?

1. Option one description
2. Option two description
3. Option three description`

	result, ok := ParseResponse(text)

	if !ok {
		t.Fatal("Expected parsing to succeed")
	}

	if result.Question != "I need clarification. What did you mean?" {
		t.Errorf("Expected question 'I need clarification. What did you mean?', got '%s'", result.Question)
	}

	expectedOptions := []string{
		"Option one description",
		"Option two description",
		"Option three description",
	}

	if len(result.Options) != len(expectedOptions) {
		t.Fatalf("Expected %d options, got %d", len(expectedOptions), len(result.Options))
	}

	for i, expected := range expectedOptions {
		if result.Options[i] != expected {
			t.Errorf("Option %d: expected '%s', got '%s'", i+1, expected, result.Options[i])
		}
	}
}

func TestParseResponse_MinimumTwoOptions(t *testing.T) {
	text := `Question here?

1. Only one option`

	_, ok := ParseResponse(text)

	if ok {
		t.Fatal("Expected parsing to fail with only one option")
	}
}

func TestParseResponse_BulletListRejected(t *testing.T) {
	text := `Question here?

- Bullet option one
- Bullet option two`

	_, ok := ParseResponse(text)

	if ok {
		t.Fatal("Expected parsing to fail with bullet list")
	}
}

func TestParseResponse_NoOptions(t *testing.T) {
	text := `This is just a regular response without any numbered options.
It might have multiple lines but no numbered list.`

	_, ok := ParseResponse(text)

	if ok {
		t.Fatal("Expected parsing to fail with no options")
	}
}

func TestParseResponse_NumberedListInExplanation(t *testing.T) {
	text := `Here are three steps to solve this:

1. First step
2. Second step
3. Third step

But this is not a clarifying question.`

	result, ok := ParseResponse(text)

	if !ok {
		t.Fatal("Expected parsing to succeed")
	}

	if result.Question != "Here are three steps to solve this:" {
		t.Errorf("Question parsing failed, got: '%s'", result.Question)
	}

	if len(result.Options) != 3 {
		t.Errorf("Expected 3 options, got %d", len(result.Options))
	}
}

func TestParseResponse_EmptyLinesBetween(t *testing.T) {
	text := `What did you mean?


1. First option
2. Second option`

	result, ok := ParseResponse(text)

	if !ok {
		t.Fatal("Expected parsing to succeed with empty lines")
	}

	if result.Question != "What did you mean?" {
		t.Errorf("Expected question 'What did you mean?', got '%s'", result.Question)
	}

	if len(result.Options) != 2 {
		t.Errorf("Expected 2 options, got %d", len(result.Options))
	}
}

func TestParseResponse_IndentedOptions(t *testing.T) {
	text := `Question?

  1. Indented option one
  2. Indented option two`

	result, ok := ParseResponse(text)

	if !ok {
		t.Fatal("Expected parsing to succeed with indented options")
	}

	if len(result.Options) != 2 {
		t.Errorf("Expected 2 options, got %d", len(result.Options))
	}

	if result.Options[0] != "Indented option one" {
		t.Errorf("Expected trimmed option text, got '%s'", result.Options[0])
	}
}

func TestParseResponse_NonSequentialNumbers(t *testing.T) {
	text := `Question?

1. First option
3. Third option (skipped 2)`

	_, ok := ParseResponse(text)

	if ok {
		t.Fatal("Expected parsing to fail when non-sequential numbers result in < 2 options")
	}
}

func TestParseResponse_TwoOptions(t *testing.T) {
	text := `Which do you prefer?

1. First option
2. Second option`

	result, ok := ParseResponse(text)

	if !ok {
		t.Fatal("Expected parsing to succeed with exactly 2 options")
	}

	if len(result.Options) != 2 {
		t.Errorf("Expected 2 options, got %d", len(result.Options))
	}
}

func TestParseResponse_FourOptions(t *testing.T) {
	text := `Choose one:

1. Option A
2. Option B
3. Option C
4. Option D`

	result, ok := ParseResponse(text)

	if !ok {
		t.Fatal("Expected parsing to succeed with 4 options")
	}

	if len(result.Options) != 4 {
		t.Errorf("Expected 4 options, got %d", len(result.Options))
	}
}

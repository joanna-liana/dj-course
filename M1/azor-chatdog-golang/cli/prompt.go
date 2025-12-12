package cli

import (
	"strings"

	"github.com/c-bata/go-prompt"
)

var inputResult string
var inputDone bool

// completer provides command suggestions based on current input
func completer(d prompt.Document) []prompt.Suggest {
	line := d.TextBeforeCursor()
	words := strings.Fields(line)

	// If no slash command yet, suggest all slash commands
	if len(line) == 0 || line[0] != '/' {
		return []prompt.Suggest{
			{Text: "/help", Description: "Wyświetla pomoc"},
			{Text: "/exit", Description: "Zakończenie czatu"},
			{Text: "/quit", Description: "Zakończenie czatu"},
			{Text: "/session", Description: "Zarządzanie sesjami"},
			{Text: "/pdf", Description: "Export do PDF"},
		}
	}

	// If we have /session command, suggest subcommands
	if len(words) >= 1 && words[0] == "/session" {
		if len(words) == 1 || (len(words) == 2 && !strings.HasSuffix(line, " ")) {
			return []prompt.Suggest{
				{Text: "list", Description: "Interaktywny wybór sesji"},
				{Text: "display", Description: "Wyświetla całą historię"},
				{Text: "pop", Description: "Usuwa ostatnią wymianę"},
				{Text: "clear", Description: "Czyści historię"},
				{Text: "new", Description: "Rozpoczyna nową sesję"},
				{Text: "remove", Description: "Usuwa bieżącą sesję"},
			}
		}
	}

	return []prompt.Suggest{}
}

// GetUserInput gets user input with autocompletion
func GetUserInput(promptText string) (string, error) {
	if promptText == "" {
		promptText = "TY: "
	}

	result := prompt.Input(
		promptText,
		completer,
		prompt.OptionTitle("AZOR the CHATDOG"),
		prompt.OptionPrefixTextColor(prompt.Blue),
		prompt.OptionPreviewSuggestionTextColor(prompt.DarkGray),
		prompt.OptionSelectedSuggestionBGColor(prompt.Cyan),
		prompt.OptionSuggestionBGColor(prompt.DarkGray),
		prompt.OptionMaxSuggestion(10),
		prompt.OptionShowCompletionAtStart(),
		prompt.OptionCompletionOnDown(),
	)

	return strings.TrimSpace(result), nil
}

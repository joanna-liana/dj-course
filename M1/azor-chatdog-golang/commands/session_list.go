package commands

import (
	"azor-chatdog/cli"
	"azor-chatdog/files"
	"fmt"

	"github.com/mattn/go-tty"
)

const (
	keyEscape    = 0x1b
	keyEnter     = '\r'
	keyNewline   = '\n'
	keyArrowUp   = 'A'
	keyArrowDown = 'B'
)

type keyPress int

const (
	keyUnknown keyPress = iota
	keyUp
	keyDown
	keySelect
	keyCancel
)

func readKey(t *tty.TTY) (keyPress, error) {
	r, err := t.ReadRune()
	if err != nil {
		return keyUnknown, err
	}

	if r == keyEnter || r == keyNewline {
		return keySelect, nil
	}

	if r == 'q' || r == 'Q' {
		return keyCancel, nil
	}

	if r == keyEscape {
		r2, err := t.ReadRune()
		if err != nil || r2 != '[' {
			return keyCancel, nil
		}

		arrowKey, err := t.ReadRune()
		if err != nil {
			return keyUnknown, err
		}

		switch arrowKey {
		case keyArrowUp:
			return keyUp, nil
		case keyArrowDown:
			return keyDown, nil
		}
	}

	return keyUnknown, nil
}

func clearScreen() {
	fmt.Print("\033[H\033[2J")
}

func displaySessionList(sessions []files.SessionInfo, selectedIndex int) {
	clearScreen()
	fmt.Println("\n=== Wybierz Sesję ===")
	fmt.Println("Strzałki: nawigacja | Enter: wybierz | ESC/q: anuluj")

	for i, session := range sessions {
		if i == selectedIndex {
			fmt.Printf("→ \033[36m%s\033[0m - %d wiadomości (ostatnia: %s)\n",
				session.ID, session.MessagesCount, session.LastActivity)
		} else {
			fmt.Printf("  %s - %d wiadomości (ostatnia: %s)\n",
				session.ID, session.MessagesCount, session.LastActivity)
		}
	}
}

// InteractiveListSessionsCommand shows an interactive session selector
func InteractiveListSessionsCommand() (string, bool) {
	sessions := files.ListSessions()

	if len(sessions) == 0 {
		cli.PrintInfo("Brak zapisanych sesji.")
		return "", false
	}

	validSessions := make([]files.SessionInfo, 0)
	for _, session := range sessions {
		if session.Error == "" {
			validSessions = append(validSessions, session)
		}
	}

	if len(validSessions) == 0 {
		cli.PrintError("Brak prawidłowych sesji do wyświetlenia.")
		return "", false
	}

	terminal, err := tty.Open()
	if err != nil {
		cli.PrintError(fmt.Sprintf("Nie można otworzyć terminala: %v", err))
		return "", false
	}
	defer terminal.Close()

	selectedIndex := 0

	for {
		displaySessionList(validSessions, selectedIndex)

		key, err := readKey(terminal)
		if err != nil {
			clearScreen()
			return "", false
		}

		switch key {
		case keyUp:
			if selectedIndex > 0 {
				selectedIndex--
			}
		case keyDown:
			if selectedIndex < len(validSessions)-1 {
				selectedIndex++
			}
		case keySelect:
			clearScreen()
			return validSessions[selectedIndex].ID, true
		case keyCancel:
			clearScreen()
			return "", false
		}
	}
}

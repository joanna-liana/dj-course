package cli

import (
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

func displayOptions(prompt string, options []string, selectedIndex int) {
	clearScreen()
	fmt.Printf("\n%s\n", prompt)
	fmt.Println("Strzałki: nawigacja | Enter: wybierz | ESC: anuluj")

	for i, option := range options {
		if i == selectedIndex {
			fmt.Printf("→ \033[36m%d. %s\033[0m\n", i+1, option)
		} else {
			fmt.Printf("  %d. %s\n", i+1, option)
		}
	}
}

func SelectOption(prompt string, options []string) (string, bool) {
	if len(options) == 0 {
		return "", false
	}

	terminal, err := tty.Open()
	if err != nil {
		PrintError(fmt.Sprintf("Nie można otworzyć terminala: %v", err))
		return "", false
	}
	defer terminal.Close()

	selectedIndex := 0

	for {
		displayOptions(prompt, options, selectedIndex)

		key, err := readKey(terminal)
		if err != nil {
			clearScreen()
			return "", false
		}

		switch key {
		case keyUp:
			if selectedIndex > 0 {
				selectedIndex--
			} else {
				selectedIndex = len(options) - 1
			}
		case keyDown:
			if selectedIndex < len(options)-1 {
				selectedIndex++
			} else {
				selectedIndex = 0
			}
		case keySelect:
			clearScreen()
			return options[selectedIndex], true
		case keyCancel:
			clearScreen()
			return "", false
		}
	}
}

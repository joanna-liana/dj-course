package main

import (
	"azor-chatdog/clarify"
	"azor-chatdog/cli"
	"azor-chatdog/commands"
	"azor-chatdog/session"
	"fmt"
	"io"
	"strings"
)

// InitChat initializes a new session or loads an existing one
func InitChat() error {
	commands.PrintWelcome()
	manager := session.GetSessionManager()

	// Initialize session based on CLI args
	cliSessionID := cli.GetSessionIDFromCLI()
	sess, err := manager.InitializeFromCLI(cliSessionID)

	if err != nil {
		if cliSessionID != "" {
			cli.PrintError(fmt.Sprintf("Błąd wczytywania sesji: %v", err))
			if sess != nil {
				cli.PrintInfo(fmt.Sprintf("Rozpoczęto nową sesję z ID: %s", sess.SessionID()))
			} else {
				return fmt.Errorf("nie udało się zainicjalizować sesji: %w", err)
			}
		} else {
			return fmt.Errorf("nie udało się utworzyć nowej sesji: %w", err)
		}
	}

	cli.DisplayHelp(sess.SessionID())

	if !sess.IsEmpty() {
		commands.DisplayHistorySummary(sess.GetHistory(), sess.AssistantName())
	}

	return nil
}

// MainLoop is the main loop of the interactive chat
func MainLoop() {
	manager := session.GetSessionManager()

	for {
		userInput, err := cli.GetUserInput("TY: ")

		// Handle EOF (Ctrl+D)
		if err == io.EOF {
			cli.PrintInfo("\nWyjście (Ctrl+D).")
			break
		}

		// Handle other errors
		if err != nil {
			cli.PrintError(fmt.Sprintf("\nWystąpił błąd podczas odczytu: %v", err))
			break
		}

		// Skip empty input
		if userInput == "" {
			continue
		}

		// Handle slash commands
		if strings.HasPrefix(userInput, "/") {
			shouldExit := HandleCommand(userInput)
			if shouldExit {
				break
			}
			continue
		}

		// Get current session
		sess, err := manager.GetCurrentSession()
		if err != nil {
			cli.PrintError(fmt.Sprintf("Błąd: %v", err))
			continue
		}

		// Send message
		response, err := sess.SendMessage(userInput)
		if err != nil {
			cli.PrintError(fmt.Sprintf("Błąd podczas wysyłania wiadomości: %v", err))
			continue
		}

		// Check for clarifying questions (max 2 iterations)
		clarificationCount := 0
		maxClarifications := 2

		for clarificationCount < maxClarifications {
			clarifyQ, hasClarification := clarify.ParseResponse(response.Text)
			if !hasClarification {
				break
			}

			clarificationCount++

			cli.PrintAssistant(fmt.Sprintf("\n%s: %s", sess.AssistantName(), clarifyQ.Question))

			selected, ok := cli.SelectOption("Wybierz opcję (ESC aby pominąć):", clarifyQ.Options)

			if !ok {
				cli.PrintInfo("\nPodaj swoją odpowiedź:")
				selected, err = cli.GetUserInput("TY: ")
				if err == io.EOF {
					cli.PrintInfo("\nWyjście (Ctrl+D).")
					return
				}
				if err != nil {
					cli.PrintError(fmt.Sprintf("\nWystąpił błąd podczas odczytu: %v", err))
					break
				}
			}

			response, err = sess.SendMessage(selected)
			if err != nil {
				cli.PrintError(fmt.Sprintf("Błąd podczas wysyłania odpowiedzi: %v", err))
				break
			}
		}

		if clarificationCount >= maxClarifications {
			cli.PrintError("Zbyt wiele pytań wyjaśniających. Proszę przeformułować pytanie.")
		}

		// Get token info
		totalTokens, remainingTokens, maxTokens := sess.GetTokenInfo()

		// Display response
		cli.PrintAssistant(fmt.Sprintf("\n%s: %s", sess.AssistantName(), response.Text))
		cli.PrintInfo(fmt.Sprintf("Tokens: %d (Pozostało: %d / %d)", totalTokens, remainingTokens, maxTokens))

		// Save session
		if err := sess.SaveToFile(); err != nil {
			cli.PrintError(fmt.Sprintf("Error saving session: %v", err))
		}
	}
}

// Cleanup performs cleanup on program exit
func Cleanup() {
	manager := session.GetSessionManager()
	sess, err := manager.GetCurrentSession()

	if err != nil {
		return
	}

	if sess.IsEmpty() {
		cli.PrintInfo("\nSesja jest pusta/niekompletna. Pominięto finalny zapis.")
	} else {
		cli.PrintInfo(fmt.Sprintf("\nFinalny zapis historii sesji: %s", sess.SessionID()))
		manager.CleanupAndSave()
		cli.DisplayFinalInstructions(sess.SessionID())
	}
}

package llm

import (
	"context"
	"errors"
	"fmt"

	openai "github.com/sashabaranov/go-openai"
)

// CerebrasChatSessionWrapper wraps Cerebras chat session
type CerebrasChatSessionWrapper struct {
	client  *openai.Client
	model   string
	history []Message
	sysInst string
}

// SendMessage sends a message to Cerebras
func (c *CerebrasChatSessionWrapper) SendMessage(text string) (*Response, error) {
	ctx := context.Background()

	// Add user message to history
	c.history = append(c.history, Message{
		Role:  "user",
		Parts: []Part{{Text: text}},
	})

	// Build messages for API (include system instruction + history)
	var messages []openai.ChatCompletionMessage

	// Add system instruction as system message
	if c.sysInst != "" {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: c.sysInst,
		})
	}

	// Add conversation history
	for _, msg := range c.history {
		if len(msg.Parts) > 0 {
			role := msg.Role
			content := msg.Parts[0].Text

			if role == "user" {
				messages = append(messages, openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleUser,
					Content: content,
				})
			} else if role == "model" || role == "assistant" {
				messages = append(messages, openai.ChatCompletionMessage{
					Role:    openai.ChatMessageRoleAssistant,
					Content: content,
				})
			}
		}
	}

	// Create chat completion request
	resp, err := c.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:    c.model,
		Messages: messages,
	})
	if err != nil {
		return nil, fmt.Errorf("cerebras API error: %w", err)
	}

	// Extract response text
	var responseText string
	if len(resp.Choices) > 0 {
		responseText = resp.Choices[0].Message.Content
	}

	// Update history
	c.history = append(c.history, Message{
		Role:  "model",
		Parts: []Part{{Text: responseText}},
	})

	return &Response{Text: responseText}, nil
}

// GetHistory returns conversation history
func (c *CerebrasChatSessionWrapper) GetHistory() []Message {
	return c.history
}

// CerebrasLLMClient encapsulates Cerebras Inference API interactions
type CerebrasLLMClient struct {
	modelName string
	apiKey    string
	client    *openai.Client
}

// NewCerebrasLLMClient creates a new Cerebras LLM client
func NewCerebrasLLMClient(modelName, apiKey string) (*CerebrasLLMClient, error) {
	if apiKey == "" {
		return nil, errors.New("API key cannot be empty")
	}

	// Create OpenAI client with Cerebras base URL
	config := openai.DefaultConfig(apiKey)
	config.BaseURL = "https://api.cerebras.ai/v1"

	client := openai.NewClientWithConfig(config)

	return &CerebrasLLMClient{
		modelName: modelName,
		apiKey:    apiKey,
		client:    client,
	}, nil
}

// FromEnvironmentCerebras creates a Cerebras client from environment
func FromEnvironmentCerebras() (*CerebrasLLMClient, error) {
	config, err := GetCerebrasConfigFromEnv()
	if err != nil {
		return nil, err
	}

	return NewCerebrasLLMClient(config.ModelName, config.CerebrasAPIKey)
}

// CreateChatSession creates a new chat session
func (c *CerebrasLLMClient) CreateChatSession(systemInstruction string, history []Message, thinkingBudget int) (ChatSession, error) {
	return &CerebrasChatSessionWrapper{
		client:  c.client,
		model:   c.modelName,
		history: history,
		sysInst: systemInstruction,
	}, nil
}

// CountHistoryTokens counts tokens in history
// Cerebras API doesn't expose token counting, so we approximate
func (c *CerebrasLLMClient) CountHistoryTokens(history []Message) (int, error) {
	if len(history) == 0 {
		return 0, nil
	}

	// Approximate token count (4 chars per token average)
	totalChars := 0
	for _, msg := range history {
		if len(msg.Parts) > 0 {
			totalChars += len(msg.Parts[0].Text)
		}
	}

	return totalChars / 4, nil
}

// GetModelName returns the model name
func (c *CerebrasLLMClient) GetModelName() string {
	return c.modelName
}

// IsAvailable checks if the client is available
func (c *CerebrasLLMClient) IsAvailable() bool {
	return c.client != nil && c.apiKey != ""
}

// ReadyForUseMessage returns a ready message
func (c *CerebrasLLMClient) ReadyForUseMessage() string {
	maskedKey := "****"
	if len(c.apiKey) > 8 {
		maskedKey = fmt.Sprintf("%s...%s", c.apiKey[:4], c.apiKey[len(c.apiKey)-4:])
	}
	return fmt.Sprintf("✅ Klient Cerebras gotowy (Model: %s, Key: %s)", c.modelName, maskedKey)
}

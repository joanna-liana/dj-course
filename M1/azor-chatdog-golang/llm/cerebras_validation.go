package llm

import (
	"errors"
	"os"
)

// CerebrasConfig holds validated Cerebras configuration
type CerebrasConfig struct {
	ModelName      string
	CerebrasAPIKey string
}

// ValidateCerebrasConfig validates Cerebras configuration
func ValidateCerebrasConfig(modelName, apiKey string) (*CerebrasConfig, error) {
	if apiKey == "" {
		return nil, errors.New("CEREBRAS_API_KEY is required but not set")
	}
	if modelName == "" {
		modelName = "llama-3.3-70b"
	}
	return &CerebrasConfig{
		ModelName:      modelName,
		CerebrasAPIKey: apiKey,
	}, nil
}

// GetCerebrasConfigFromEnv gets Cerebras config from environment
func GetCerebrasConfigFromEnv() (*CerebrasConfig, error) {
	modelName := os.Getenv("CEREBRAS_MODEL_NAME")
	if modelName == "" {
		modelName = os.Getenv("MODEL_NAME")
	}
	apiKey := os.Getenv("CEREBRAS_API_KEY")

	return ValidateCerebrasConfig(modelName, apiKey)
}

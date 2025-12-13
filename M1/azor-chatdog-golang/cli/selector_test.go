package cli

import (
	"testing"
)

func TestSelectOption_EmptyOptions(t *testing.T) {
	options := []string{}
	_, ok := SelectOption("Test", options)

	if ok {
		t.Fatal("Expected SelectOption to return false with empty options")
	}
}

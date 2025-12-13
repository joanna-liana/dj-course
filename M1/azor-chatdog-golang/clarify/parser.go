package clarify

import (
	"regexp"
	"strings"
)

type ClarifyingQuestion struct {
	Question string
	Options  []string
}

func ParseResponse(text string) (*ClarifyingQuestion, bool) {
	lines := strings.Split(text, "\n")

	numberedLinePattern := regexp.MustCompile(`^\s*(\d+)\.\s+(.+)$`)

	var firstOptionIndex int = -1
	var options []string

	for i, line := range lines {
		matches := numberedLinePattern.FindStringSubmatch(line)
		if matches != nil {
			if firstOptionIndex == -1 {
				firstOptionIndex = i
			}

			optionNumber := matches[1]
			optionText := strings.TrimSpace(matches[2])

			expectedNumber := len(options) + 1
			if optionNumber == string(rune('0'+expectedNumber)) {
				options = append(options, optionText)
			} else {
				break
			}
		} else if firstOptionIndex != -1 {
			break
		}
	}

	if len(options) < 2 {
		return nil, false
	}

	questionLines := lines[:firstOptionIndex]
	question := strings.TrimSpace(strings.Join(questionLines, "\n"))

	return &ClarifyingQuestion{
		Question: question,
		Options:  options,
	}, true
}

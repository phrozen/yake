package yake

import (
	"strconv"
	"strings"
	"unicode"
)

// Tag classifies a single word token into one of six categories for feature
// extraction. The classification drives which terms participate in the
// co-occurrence graph and how the casing feature is computed.
type Tag int

const (
	TagDigit       Tag = iota // numeric tokens ("123", "1,000")
	TagPunctuation            // tokens composed entirely of punctuation marks
	TagUnparsable             // mixed alphanumeric, special characters, or multiple punctuation
	TagAcronym                // all-uppercase words (NASA, CEO)
	TagUppercase              // capitalized words not at the start of a sentence (proper noun heuristic)
	TagParsable               // ordinary lowercased words
)

var tagNames = [...]string{
	TagDigit:       "Digit",
	TagPunctuation: "Punctuation",
	TagUnparsable:  "Unparsable",
	TagAcronym:     "Acronym",
	TagUppercase:   "Uppercase",
	TagParsable:    "Parsable",
}

func (t Tag) String() string {
	if int(t) < len(tagNames) {
		return tagNames[t]
	}
	return "Unknown"
}

// Classify determines the [Tag] for a word. Punctuation, numeric, and
// unparsable tokens are detected first because they are excluded from
// meaningful analysis. First words of sentences are never tagged as
// uppercase to avoid conflating sentence-start capitalization with
// proper nouns.
func Classify(word string, isFirstWordOfSentence bool, strictCapital bool, punctuation map[rune]struct{}) Tag {
	if isNumeric(word) {
		return TagDigit
	}
	if isPunctuationWord(word, punctuation) {
		return TagPunctuation
	}
	if isUnparsable(word, punctuation) {
		return TagUnparsable
	}
	if isAcronym(word) {
		return TagAcronym
	}
	if isUpper(word, isFirstWordOfSentence, strictCapital) {
		return TagUppercase
	}
	return TagParsable
}

func isNumeric(word string) bool {
	s := strings.ReplaceAll(word, ",", "")
	_, err := strconv.ParseFloat(s, 64)
	return err == nil
}

func isAcronym(word string) bool {
	for _, r := range word {
		if !unicode.IsUpper(r) {
			return false
		}
	}
	return true
}

// isUpper returns true for capitalized words that are not at sentence
// start. When strictCapital is true, only words with exactly one leading
// uppercase letter qualify (e.g. "Paypal" but not "PayPal").
func isUpper(word string, isFirstWordOfSentence, strictCapital bool) bool {
	if isFirstWordOfSentence {
		return false
	}
	if strictCapital {
		return isStrictCapitalized(word)
	}
	return isCapitalized(word)
}

// isCapitalized returns true if the first rune is uppercase.
func isCapitalized(word string) bool {
	runes := []rune(word)
	return len(runes) > 0 && unicode.IsUpper(runes[0])
}

// isStrictCapitalized returns true if only the first rune is uppercase
// and all subsequent runes are not. This distinguishes "Paypal" from
// "PayPal", which is classified as an acronym instead.
func isStrictCapitalized(word string) bool {
	runes := []rune(word)
	if len(runes) == 0 || !unicode.IsUpper(runes[0]) {
		return false
	}
	for _, r := range runes[1:] {
		if unicode.IsUpper(r) {
			return false
		}
	}
	return true
}

func isPunctuationWord(word string, punctuation map[rune]struct{}) bool {
	for _, r := range word {
		if _, ok := punctuation[r]; !ok {
			return false
		}
	}
	return true
}

// isUnparsable catches tokens that carry no semantic meaning: mixed
// letters and digits (e.g. "B2C"), strings with multiple distinct
// punctuation characters, or strings with neither letters nor digits.
func isUnparsable(word string, punctuation map[rune]struct{}) bool {
	if hasMultiplePunctuation(word, punctuation) {
		return true
	}
	hasDigit := false
	hasAlpha := false
	for _, r := range word {
		if unicode.IsDigit(r) {
			hasDigit = true
		}
		if unicode.IsLetter(r) {
			hasAlpha = true
		}
	}
	return hasAlpha == hasDigit
}

func hasMultiplePunctuation(word string, punctuation map[rune]struct{}) bool {
	count := 0
	seen := make(map[rune]struct{})
	for _, r := range word {
		if _, ok := punctuation[r]; ok {
			if _, already := seen[r]; !already {
				count++
				seen[r] = struct{}{}
			}
		}
	}
	return count > 1
}

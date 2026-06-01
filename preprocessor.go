package yake

import (
	"unicode"
	"unicode/utf8"
)

func splitIntoSentences(text string) []string {
	var sentences []string
	for _, s := range splitMulti(text) {
		s = trimSpace(s)
		if s != "" {
			sentences = append(sentences, s)
		}
	}
	return sentences
}

func splitIntoWords(text string) []string {
	var words []string
	for _, w := range splitContractions(webTokenizer(text)) {
		if w == "" {
			continue
		}
		if len(w) > 1 && w[0] == '\'' {
			continue
		}
		words = append(words, w)
	}
	return words
}

func trimSpace(s string) string {
	start := 0
	for i, r := range s {
		if !unicode.IsSpace(r) {
			start = i
			break
		}
	}
	s = s[start:]

	end := len(s)
	for end > 0 {
		r, size := utf8.DecodeLastRuneInString(s[:end])
		if !unicode.IsSpace(r) {
			break
		}
		end -= size
	}
	return s[:end]
}

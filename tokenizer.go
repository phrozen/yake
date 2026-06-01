package yake

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// sentenceEndRe matches a sentence terminal (.!?) followed by one or
// more whitespace characters and then an uppercase letter or digit.
// This is used as a heuristic for sentence boundary detection.
var sentenceEndRe = regexp.MustCompile(`[.!?]\s+[\p{Lu}\p{Lt}\d]`)

// splitMulti splits text into sentences. Paragraph breaks (blank lines)
// are treated as sentence boundaries. Lines not ending with a sentence
// terminal are joined into the previous paragraph.
func splitMulti(text string) []string {
	text = strings.ReplaceAll(text, "\t", " ")
	text = strings.ReplaceAll(text, "\r", "")

	lines := strings.Split(text, "\n")
	var paragraphs []string
	var current string

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			if current != "" {
				paragraphs = append(paragraphs, current)
				current = ""
			}
			continue
		}

		if current == "" {
			current = trimmed
		} else {
			lastRunes := []rune(current)
			lastCh := lastRunes[len(lastRunes)-1]
			if (lastCh == '.' || lastCh == '!' || lastCh == '?') && len(trimmed) > 0 {
				firstR, _ := utf8.DecodeRuneInString(trimmed)
				if unicode.IsUpper(firstR) || unicode.IsDigit(firstR) {
					paragraphs = append(paragraphs, current)
					current = trimmed
				} else {
					current = current + " " + trimmed
				}
			} else {
				current = current + " " + trimmed
			}
		}
	}

	if current != "" {
		paragraphs = append(paragraphs, current)
	}

	var result []string
	for _, p := range paragraphs {
		sents := splitParagraph(p)
		result = append(result, sents...)
	}
	return result
}

// splitParagraph splits a single paragraph into sentences by finding
// sentence-end markers followed by whitespace and an uppercase letter
// or digit. Text after the last sentence terminal is returned as-is.
func splitParagraph(text string) []string {
	var sentences []string

	for {
		loc := sentenceEndRe.FindStringIndex(text)
		if loc == nil {
			trimmed := strings.TrimSpace(text)
			if trimmed != "" {
				sentences = append(sentences, trimmed)
			}
			break
		}

		end := loc[0] + 1
		sentence := strings.TrimSpace(text[:end])
		sentences = append(sentences, sentence)

		start := loc[1] - 1
		for start < len(text) && text[start] == ' ' {
			start++
		}
		text = text[start:]
	}

	return sentences
}

// contractionsMap maps contraction suffixes to their token forms.
// These are the affixes that segtok's split_contractions recognizes.
var contractionsMap map[string][]string

func init() {
	contractionsMap = map[string][]string{
		"n't": {"n't"},
		"'s":  {"'s"},
		"'re": {"'re"},
		"'ve": {"'ve"},
		"'m":  {"'m"},
		"'ll": {"'ll"},
		"'d":  {"'d"},
	}
}

// splitContractions applies contraction splitting to each word.
// Words like "don't" become ["do", "n't"] so that the verb stem and
// the negation suffix can be classified independently.
func splitContractions(words []string) []string {
	var result []string
	for _, w := range words {
		result = append(result, splitOneContraction(w)...)
	}
	return result
}

var contractionPrefixes = map[string]string{
	"won't":  "will",
	"Won't":  "Will",
	"can't":  "ca",
	"Can't":  "Ca",
	"shan't": "shall",
	"Shan't": "Shall",
}

// splitOneContraction splits a single word on the first recognized
// contraction suffix. Special cases "won't", "can't", and "shan't"
// expand the stem to its full form ("will", "ca", "shall").
func splitOneContraction(word string) []string {
	if exp, ok := contractionPrefixes[word]; ok {
		return []string{exp, "n't"}
	}

	for i := 0; i < len(word); i++ {
		for suffix := range contractionsMap {
			if i+len(suffix) <= len(word) && word[i:i+len(suffix)] == suffix {
				left := word[:i]
				if left == "" {
					left = "wo"
				}
				return []string{left, suffix}
			}
		}
	}
	return []string{word}
}

// webTokenizer is a re-implementation of segtok's web_tokenizer. It
// splits text on whitespace and punctuation boundaries while preserving
// intra-word hyphens and decimal points embedded in numbers.
func webTokenizer(text string) []string {
	var tokens []string
	var current []rune

	for i := 0; i < len(text); {
		r, size := utf8.DecodeRuneInString(text[i:])
		if r == utf8.RuneError && size == 0 {
			break
		}

		if unicode.IsSpace(r) {
			if len(current) > 0 {
				tokens = append(tokens, string(current))
				current = nil
			}
			i += size
			continue
		}

		// Hyphens are attached to the current token when they
		// appear between letters (e.g. "high-tech" stays as one token).
		if r == '-' && len(current) > 0 && i+size < len(text) {
			nextR, _ := utf8.DecodeRuneInString(text[i+size:])
			if unicode.IsLetter(nextR) {
				current = append(current, r)
				i += size
				continue
			}
		}

		// Apostrophes in contractions like "don't", "it's" are
		// kept attached so splitContractions can handle them later.
		if r == '\'' && len(current) > 0 && i+size < len(text) {
			after := text[i+size:]
			if len(after) >= 2 && (strings.HasPrefix(after, "ll") || strings.HasPrefix(after, "LL")) {
				current = append(current, r)
				i += size
				continue
			}
			if len(after) >= 1 {
				nextR, _ := utf8.DecodeRuneInString(after)
				if nextR == 's' || nextR == 'S' || nextR == 't' || nextR == 'T' ||
					nextR == 'm' || nextR == 'M' || nextR == 'd' || nextR == 'D' ||
					nextR == 'v' || nextR == 'V' || nextR == 'r' || nextR == 'R' {
					if len(after) >= 2 && nextR == 't' {
						current = append(current, r)
						i += size
						continue
					}
					current = append(current, r)
					i += size
					continue
				}
			}
		}

		// Keep decimal numbers together (e.g. "3.14" stays as one token).
		if r == '.' && len(current) > 0 && i+size < len(text) {
			nextR, _ := utf8.DecodeRuneInString(text[i+size:])
			if unicode.IsDigit(nextR) {
				current = append(current, r)
				i += size
				continue
			}
		}

		if tokenBoundary(r) {
			if len(current) > 0 {
				tokens = append(tokens, string(current))
				current = nil
			}
			tokens = append(tokens, string(r))
			i += size
			continue
		}

		current = append(current, r)
		i += size
	}

	if len(current) > 0 {
		tokens = append(tokens, string(current))
	}

	return tokens
}

// tokenBoundary returns true for characters that always split tokens,
// including most ASCII punctuation. Apostrophe is excluded because it
// is handled separately by contraction splitting.
func tokenBoundary(r rune) bool {
	switch r {
	case '!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~':
		return true
	}
	return false
}

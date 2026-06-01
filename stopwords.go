package yake

import (
	"embed"
	"strings"
)

//go:embed stopwords/*.txt
var stopwordsFS embed.FS

// StopWords holds a set of lowercased words to exclude from keyword
// extraction. Words shorter than 3 non-punctuation characters are
// also treated as stopwords regardless of this set.
type StopWords struct {
	set map[string]struct{}
}

// NewStopWords returns an empty stopword set.
func NewStopWords() *StopWords {
	return &StopWords{set: make(map[string]struct{})}
}

// StopWordsFromList creates a [StopWords] from an explicit list.
// All entries are lowercased on insertion.
func StopWordsFromList(words []string) *StopWords {
	sw := NewStopWords()
	for _, w := range words {
		sw.set[strings.ToLower(w)] = struct{}{}
	}
	return sw
}

// PredefinedStopWords loads a built-in stopword list for the given
// ISO 639-2 language code. Returns nil if no list exists for the code.
//
// Supported languages: ar, bg, br, cz, da, de, el, en, es, et, fa,
// fi, fr, hi, hr, hu, hy, id, it, ja, lt, lv, nl, no, pl, pt, ro,
// ru, sk, sl, sv, tr, uk, zh.
func PredefinedStopWords(langISO6392 string) *StopWords {
	data, err := stopwordsFS.ReadFile("stopwords/" + langISO6392 + ".txt")
	if err != nil {
		return nil
	}
	sw := NewStopWords()
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		w := strings.TrimSpace(line)
		if w != "" {
			sw.set[w] = struct{}{}
		}
	}
	return sw
}

// Contains reports whether word is in the stopword set.
func (sw *StopWords) Contains(word string) bool {
	_, ok := sw.set[word]
	return ok
}

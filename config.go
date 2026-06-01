package yake

import (
	"errors"
	"fmt"
)

// Config controls keyword extraction behavior.
//
// Zero values are not valid; use [DefaultConfig] for sane defaults.
type Config struct {
	// Ngrams is the maximum number of words a keyphrase may contain.
	// For example, Ngrams=3 extracts 1-, 2-, and 3-word phrases.
	Ngrams int

	// Punctuation is the set of characters treated as token boundaries.
	// Words consisting solely of these characters are ignored.
	Punctuation map[rune]struct{}

	// WindowSize is the number of preceding tokens examined when building
	// the co-occurrence graph for term relatedness. Larger values capture
	// broader context but reduce performance.
	WindowSize int

	// StrictCapital, when true, counts a term as "uppercase" only when exactly
	// one leading letter is upper (e.g. "Paypal" but not "PayPal").
	// The original YAKE implementation uses true.
	StrictCapital bool

	// OnlyAlphanumericHyphen, when true, rejects candidates containing
	// any character that is not alphanumeric or a hyphen.
	OnlyAlphanumericHyphen bool

	// MinimumChars is the minimum total characters a keyphrase must have.
	// Prevents extremely short or degenerate candidates.
	MinimumChars int

	// RemoveDuplicates, when true, uses Levenshtein similarity to discard
	// near-duplicate keywords during extraction.
	RemoveDuplicates bool

	// DeduplicationThreshold is the similarity threshold (0..1) for dedup.
	// Keywords whose Levenshtein ratio exceeds this value are treated as
	// duplicates. Effective only when [Config.RemoveDuplicates] is true.
	DeduplicationThreshold float64

	// Language is the ISO 639-2 two-letter code used to load the built-in
	// stopword list. Defaults to "en".
	Language string

	// StopWords, if non-nil, overrides the built-in language stopword list.
	StopWords *StopWords
}

// DefaultConfig returns a [Config] with YAKE's original defaults:
// 3-grams, window size 1, strict capital detection, 0.9 dedup threshold,
// and English stopwords.
func DefaultConfig() Config {
	punct := `!"#$%&'()*+,-./:;<=>?@[\]^_` + "`{|}~"
	pm := make(map[rune]struct{})
	for _, r := range punct {
		pm[r] = struct{}{}
	}
	return Config{
		Ngrams:                 3,
		Punctuation:            pm,
		WindowSize:             1,
		StrictCapital:          true,
		OnlyAlphanumericHyphen: false,
		MinimumChars:           3,
		RemoveDuplicates:       true,
		DeduplicationThreshold: 0.9,
		Language:               "en",
	}
}

func (c *Config) Validate() error {
	if c.Ngrams <= 0 {
		return errors.New("Ngrams must be positive")
	}
	if c.Punctuation == nil {
		return errors.New("Punctuation must not be nil")
	}
	if c.WindowSize < 0 {
		return errors.New("WindowSize must be non-negative")
	}
	if c.MinimumChars <= 0 {
		return errors.New("MinimumChars must be positive")
	}
	if c.DeduplicationThreshold < 0 || c.DeduplicationThreshold > 1 {
		return fmt.Errorf("DeduplicationThreshold must be in [0, 1], got %f", c.DeduplicationThreshold)
	}
	if len(c.Language) != 2 && c.Language != "" {
		return fmt.Errorf("Language must be a 2-letter ISO code or empty, got %q", c.Language)
	}
	return nil
}

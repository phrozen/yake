package yake

import (
	"bufio"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type expectEntry struct {
	Raw     string  `json:"raw"`
	Keyword string  `json:"keyword"`
	Score   float64 `json:"score"`
}

type containsEntry struct {
	Raw     string `json:"raw"`
	Keyword string `json:"keyword"`
}

type testCase struct {
	Name             string          `json:"name"`
	Text             string          `json:"text,omitempty"`
	File             string          `json:"file,omitempty"`
	Language         string          `json:"language"`
	Ngrams           int             `json:"ngrams,omitempty"`
	WindowSize       int             `json:"window_size,omitempty"`
	TopK             int             `json:"top_k"`
	RemoveDuplicates *bool           `json:"remove_duplicates,omitempty"`
	Expected         []expectEntry   `json:"expected,omitempty"`
	Contains         []containsEntry `json:"contains,omitempty"`
}

func loadTestCases(t *testing.T) []testCase {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", "testdata.jsonl"))
	if err != nil {
		t.Fatalf("open testdata.jsonl: %v", err)
	}
	defer f.Close()

	var cases []testCase
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var tc testCase
		if err := json.Unmarshal(scanner.Bytes(), &tc); err != nil {
			t.Fatalf("unmarshal test case: %v", err)
		}
		cases = append(cases, tc)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan testdata.jsonl: %v", err)
	}
	return cases
}

func (tc *testCase) buildConfig(t *testing.T) Config {
	t.Helper()
	cfg := DefaultConfig()
	if tc.Language != "" {
		cfg.Language = tc.Language
	}
	if tc.Ngrams != 0 {
		cfg.Ngrams = tc.Ngrams
	}
	if tc.WindowSize != 0 {
		cfg.WindowSize = tc.WindowSize
	}
	if tc.RemoveDuplicates != nil {
		cfg.RemoveDuplicates = *tc.RemoveDuplicates
	}
	return cfg
}

func (tc *testCase) text(t *testing.T) string {
	t.Helper()
	if tc.File != "" {
		data, err := os.ReadFile(filepath.Join("testdata", tc.File))
		if err != nil {
			t.Fatalf("read sample %s: %v", tc.File, err)
		}
		return string(data)
	}
	return tc.Text
}

const scoreTolerance = 0.0005

func TestExtraction(t *testing.T) {
	cases := loadTestCases(t)
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			text := tc.text(t)
			cfg := tc.buildConfig(t)

			y, err := New(cfg)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			actual := y.Extract(text, tc.TopK)

			for i := range actual {
				actual[i].Score = math.Round(actual[i].Score*10000) / 10000
			}

			if tc.Expected != nil {
				if len(actual) != len(tc.Expected) {
					t.Errorf("got %d results, want %d\nactual: %v", len(actual), len(tc.Expected), actual)
					return
				}
				for i := range actual {
					if actual[i].Raw != tc.Expected[i].Raw {
						t.Errorf("[%d].Raw: got %q, want %q", i, actual[i].Raw, tc.Expected[i].Raw)
					}
					if actual[i].Keyword != tc.Expected[i].Keyword {
						t.Errorf("[%d].Keyword: got %q, want %q", i, actual[i].Keyword, tc.Expected[i].Keyword)
					}
					if math.Abs(actual[i].Score-tc.Expected[i].Score) > scoreTolerance {
						t.Errorf("[%d].Score: got %.4f, want %.4f (diff: %.4f)", i, actual[i].Score, tc.Expected[i].Score, math.Abs(actual[i].Score-tc.Expected[i].Score))
					}
				}
			}

			if len(tc.Expected) == 0 && tc.Expected != nil {
				if len(actual) != 0 {
					t.Errorf("want 0 results, got %d: %v", len(actual), actual)
				}
			}

			if len(actual) > 0 {
				assertKeywordOrder(t, actual)

				for _, r := range actual {
					if r.Score < 0 {
						t.Errorf("negative score for %q: %f", r.Keyword, r.Score)
					}
				}
			}

			for _, c := range tc.Contains {
				assertKeywordPresent(t, actual, c.Raw, c.Keyword)
			}
		})
	}
}

func TestLevenshteinDistance(t *testing.T) {
	tests := []struct{ a, b string; want int }{
		{"hello", "hello", 0},
		{"abc", "xyz", 3},
		{"hello", "helo", 1},
		{"", "abc", 3},
		{"abc", "", 3},
		{"", "", 0},
	}
	for _, tt := range tests {
		if got := levenshteinDistance(tt.a, tt.b); got != tt.want {
			t.Errorf("levenshteinDistance(%q, %q) = %d, want %d", tt.a, tt.b, got, tt.want)
		}
	}
}

func TestLevenshteinRatio(t *testing.T) {
	if r := levenshteinRatio("hello", "hello"); r != 1.0 {
		t.Errorf("want 1.0, got %f", r)
	}
	r := levenshteinRatio("hello", "helo")
	if r <= 0 || r >= 1 {
		t.Errorf("want 0 < r < 1, got %f", r)
	}
	r = levenshteinRatio("abc", "xyz")
	if r < 0 || r >= 1 {
		t.Errorf("want 0 <= r < 1, got %f", r)
	}
	if r := levenshteinRatio("", ""); r != 1.0 {
		t.Errorf("empty strings should have ratio 1.0, got %f", r)
	}
}

func TestRemoveDuplicates(t *testing.T) {
	results := []ResultItem{
		{Raw: "machine learning", Keyword: "machine learning", Score: 0.1},
		{Raw: "machine-learning", Keyword: "machine-learning", Score: 0.15},
		{Raw: "deep learning", Keyword: "deep learning", Score: 0.2},
		{Raw: "machin learnin", Keyword: "machin learnin", Score: 0.3},
	}
	deduped := removeDuplicates(0.85, results, 10)
	found := false
	for _, r := range deduped {
		if r.Keyword == "machine-learning" || r.Keyword == "machin learnin" {
			found = true
		}
	}
	if found {
		t.Errorf("near-duplicate should have been removed, got %v", deduped)
	}
	if len(deduped) < 2 {
		t.Errorf("want at least 2 unique results, got %d", len(deduped))
	}
}

func TestStopwordsContains(t *testing.T) {
	sw := PredefinedStopWords("en")
	if sw == nil {
		t.Fatal("expected English stopwords")
	}
	if !sw.Contains("the") {
		t.Error("'the' should be a stopword")
	}
	if sw.Contains("computer") {
		t.Error("'computer' should not be a stopword")
	}
}

func TestStopWordsFromList(t *testing.T) {
	sw := StopWordsFromList([]string{"Foo", "BAR", "Baz"})
	if !sw.Contains("foo") {
		t.Error("should contain lowercase 'foo'")
	}
	if !sw.Contains("bar") {
		t.Error("should contain lowercase 'bar'")
	}
	if sw.Contains("other") {
		t.Error("should not contain 'other'")
	}
}

func TestPredefinedStopWordsMissing(t *testing.T) {
	if sw := PredefinedStopWords("xx"); sw != nil {
		t.Error("expected nil for unknown language")
	}
}

func TestTokenizerSplitIntoWords(t *testing.T) {
	words := splitIntoWords("Truly high-tech!")
	if len(words) != 3 || words[0] != "Truly" || words[1] != "high-tech" || words[2] != "!" {
		t.Errorf("unexpected words: %v", words)
	}
}

func TestTokenizerDecimalNumber(t *testing.T) {
	words := splitIntoWords("Pi is 3.14 approximately")
	found := false
	for _, w := range words {
		if w == "3.14" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected '3.14' as a single token, got %v", words)
	}
}

func TestTokenizerSplitIntoSentences(t *testing.T) {
	sents := splitIntoSentences("One smartwatch. One phone. Many phones.")
	if len(sents) != 3 {
		t.Errorf("want 3 sentences, got %d: %v", len(sents), sents)
	}
}

func TestTagClassification(t *testing.T) {
	punct := map[rune]struct{}{'!': {}, '"': {}, '#': {}, '$': {}}
	tests := []struct {
		word    string
		isFirst bool
		strict  bool
		want    Tag
	}{
		{"HELLO", false, true, TagAcronym},
		{"Hello", true, true, TagParsable},
		{"Hello", false, true, TagUppercase},
		{"123", false, true, TagDigit},
		{"!!!", false, true, TagPunctuation},
		{"hello", false, true, TagParsable},
	}
	for _, tt := range tests {
		if tag := Classify(tt.word, tt.isFirst, tt.strict, punct); tag != tt.want {
			t.Errorf("Classify(%q, %v, %v) = %v, want %v", tt.word, tt.isFirst, tt.strict, tag, tt.want)
		}
	}
}

func TestTagString(t *testing.T) {
	tests := []struct {
		tag  Tag
		want string
	}{
		{TagDigit, "Digit"},
		{TagPunctuation, "Punctuation"},
		{TagUnparsable, "Unparsable"},
		{TagAcronym, "Acronym"},
		{TagUppercase, "Uppercase"},
		{TagParsable, "Parsable"},
	}
	for _, tt := range tests {
		if got := tt.tag.String(); got != tt.want {
			t.Errorf("Tag(%d).String() = %q, want %q", tt.tag, got, tt.want)
		}
	}
}

func TestUniqueTerm(t *testing.T) {
	y, err := New(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if got := y.uniqueTerm("Phones"); got != "phone" {
		t.Errorf("want 'phone', got %q", got)
	}
	if got := y.uniqueTerm("phone"); got != "phone" {
		t.Errorf("want 'phone', got %q", got)
	}
}

func TestIsStopword(t *testing.T) {
	y, err := New(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if !y.isStopword("the") {
		t.Error("'the' should be stopword")
	}
	if !y.isStopword("a") {
		t.Error("'a' should be stopword (too short)")
	}
	if y.isStopword("computer") {
		t.Error("'computer' should not be stopword")
	}
}

func TestToSingle(t *testing.T) {
	if got := toSingle("phones"); got != "phone" {
		t.Errorf("want 'phone', got %q", got)
	}
	if got := toSingle("abc"); got != "abc" {
		t.Errorf("want 'abc', got %q", got)
	}
	if got := toSingle("PHONES"); got != "PHONE" {
		t.Errorf("want 'PHONE', got %q", got)
	}
}

func TestToSingleUnicode(t *testing.T) {
	if got := toSingle("cafés"); got != "café" {
		t.Errorf("want 'café', got %q", got)
	}
}

func TestDeterministicOutput(t *testing.T) {
	y, err := New(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	text := "Google is acquiring data science community Kaggle. Machine learning competitions are hosted on this platform."
	r1 := y.Extract(text, 10)
	r2 := y.Extract(text, 10)
	r3 := y.Extract(text, 10)
	if len(r1) != len(r2) || len(r2) != len(r3) {
		t.Fatal("different result lengths")
	}
	for i := range r1 {
		if r1[i].Raw != r2[i].Raw || r1[i].Keyword != r2[i].Keyword || r1[i].Score != r2[i].Score {
			t.Errorf("results differ at index %d: %v vs %v", i, r1[i], r2[i])
		}
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.Ngrams != 3 {
		t.Error("default ngrams should be 3")
	}
	if cfg.WindowSize != 1 {
		t.Error("default window size should be 1")
	}
	if !cfg.RemoveDuplicates {
		t.Error("remove_duplicates should default to true")
	}
	if cfg.DeduplicationThreshold != 0.9 {
		t.Error("dedup threshold should be 0.9")
	}
}

func TestNewWithNilStopWords(t *testing.T) {
	cfg := DefaultConfig()
	cfg.StopWords = nil
	y, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if y.stopWords == nil {
		t.Error("stopwords should be initialized")
	}
}

func TestNewRejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{"zero ngrams", Config{Ngrams: 0, Punctuation: map[rune]struct{}{}}},
		{"nil punctuation", Config{Ngrams: 1, Punctuation: nil}},
		{"negative window", Config{Ngrams: 1, Punctuation: map[rune]struct{}{}, WindowSize: -1}},
		{"zero min chars", Config{Ngrams: 1, Punctuation: map[rune]struct{}{}, MinimumChars: 0}},
		{"threshold > 1", Config{Ngrams: 1, Punctuation: map[rune]struct{}{}, DeduplicationThreshold: 2.0}},
		{"threshold < 0", Config{Ngrams: 1, Punctuation: map[rune]struct{}{}, DeduplicationThreshold: -0.1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := New(tt.cfg)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestNewAutoLoadsStopWords(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Language = "de"
	cfg.StopWords = nil
	y, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !y.stopWords.Contains("und") {
		t.Error("German stopwords should contain 'und'")
	}
}

func TestExtractReturnsSorted(t *testing.T) {
	y, err := New(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	r := y.Extract("data science machine learning algorithms", 10)
	for i := 1; i < len(r); i++ {
		if r[i].Score < r[i-1].Score {
			t.Fatalf("unsorted at index %d: score %.6f > %.6f", i, r[i-1].Score, r[i].Score)
		}
	}
}

func assertKeywordOrder(t *testing.T, results []ResultItem) {
	t.Helper()
	for i := 1; i < len(results); i++ {
		if results[i].Score < results[i-1].Score {
			t.Errorf("results not sorted at index %d: %f < %f", i, results[i].Score, results[i-1].Score)
		}
	}
}

func assertKeywordPresent(t *testing.T, results []ResultItem, raw, keyword string) {
	t.Helper()
	for _, r := range results {
		if r.Raw == raw && r.Keyword == keyword {
			return
		}
	}
	t.Errorf("expected keyword %q / %q in results", raw, keyword)
}

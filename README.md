# YAKE — Yet Another Keyword Extractor for Go

A zero-dependency Go implementation of [YAKE](https://github.com/LIAAD/yake), an unsupervised, lightweight keyword extraction algorithm. YAKE selects the most relevant keywords from a single document using only statistical features — no external corpora, no training data, and no dictionary lookups beyond a stopword list.

## Installation

```sh
go get github.com/phrozen/yake
```

## Quick Start

```go
package main

import (
	"fmt"
	"log"

	"github.com/phrozen/yake"
)

func main() {
	text := "Google is acquiring data science community Kaggle. " +
		"Sources tell us that Google is acquiring Kaggle, " +
		"a platform that hosts data science and machine learning competitions."

	y, err := yake.New(yake.DefaultConfig())
	if err != nil {
		log.Fatal(err)
	}
	keywords := y.Extract(text, 10)

	for _, kw := range keywords {
		fmt.Printf("%-30s  %.4f\n", kw.Keyword, kw.Score)
	}
	// Output (sorted by score, lower = better):
	// google                         0.0251
	// kaggle                         0.0273
	// ceo anthony goldbloom          0.0483
	// data science                   0.0550
	// acquiring data science         0.0603
	// ...
}
```

Lower scores indicate more important keywords.

## How It Works

YAKE extracts keywords by computing five per-term features and combining them into an **H score**:

| Feature | What it captures | Rationale |
|---|---|---|
| **Casing** | Proportion of acronym/uppercase occurrences | Uppercase terms (NASA, CEO) tend to be more relevant |
| **Position** | Median sentence position of the term | Important keywords appear earlier in a document |
| **Frequency** | Term frequency normalized by corpus statistics | Rare terms are penalized; very common terms are balanced |
| **Relatedness** | Dispersion of neighboring words | Terms with varied neighbors are more meaningful than fixed collocations |
| **Spread** | Proportion of sentences containing the term | Well-distributed terms are more important than clustered ones |

These are combined in a co-occurrence-graph–based formula to produce the final score. The algorithm supports n-grams up to a configurable length and deduplicates near-duplicate phrases using Levenshtein similarity.

## Configuration

```go
cfg := yake.DefaultConfig()
cfg.Language = "pt"               // ISO 639-2 code for built-in stopwords
cfg.Ngrams = 2                    // max words per keyphrase (default: 3)
cfg.WindowSize = 2                // co-occurrence window (default: 1)
cfg.RemoveDuplicates = true       // filter near-duplicates (default: true)
cfg.DeduplicationThreshold = 0.8  // similarity threshold (default: 0.9)
cfg.MinimumChars = 4              // min characters per candidate (default: 3)

y, err := yake.New(cfg)           // validated on construction
```

### Custom Stopwords

```go
sw := yake.StopWordsFromList([]string{"fig", "table", "figure"})
cfg := yake.DefaultConfig()
cfg.StopWords = sw
```

### Supported Languages

Built-in stopword lists are embedded for 34 languages:

`ar bg br cz da de el en es et fa fi fr hi hr hu hy id it ja lt lv nl no pl pt ro ru sk sl sv tr uk zh`

Use `yake.PredefinedStopWords("xx")` to load a list by its ISO 639-2 code. Returns `nil` for unsupported codes.

## API

```go
func DefaultConfig() Config
func New(config Config) (*Yake, error)
func (y *Yake) Extract(text string, n int) []ResultItem
```

`New` validates the configuration and returns an error for invalid values (zero n-grams, nil punctuation, out-of-range thresholds, etc.).

`ResultItem` carries the raw surface form, the normalized keyword, and the score:

```go
type ResultItem struct {
    Raw     string  // original casing, e.g. "Machine Learning"
    Keyword string  // lowercased, normalized, e.g. "machine learning"
    Score   float64 // lower is better
}
```

## Validation

This implementation is cross-validated against both the [original Python](https://github.com/LIAAD/yake) and the [Rust port](https://github.com/quesurifn/yake-rust). The test suite includes 35 cross-validation tests covering:

- Inline English tests (singular, plural, hyphenated, multi-ngram, stopword weighting, deduplication)
- File-based English tests matching the LIAAD reference samples (Google/Kaggle, Gitter, Genius, Fukushima, Global Crossing)
- Multilingual tests in 14 languages (Arabic, German, Dutch, Finnish, French, Italian, Polish, Portuguese, Spanish, Turkish)

All tests use byte-identical input files and stopword lists. Scores match the Python and Rust reference implementations. Tokenizer differences account for minor score variance in non-English edge cases; keyword identity and ranking are consistent.

## Benchmarks

Run `go test -bench=. -benchmem` to measure throughput on your hardware (Apple M2 Max reference):

| Benchmark | Time | Memory | Allocs |
|---|---|---|---|
| ExtractShort (~20 words) | ~61 µs | ~44 KB | 457 |
| ExtractMedium (~120 words) | ~199 µs | ~148 KB | 1,638 |
| Tokenizer | ~8.2 µs | ~4.2 KB | 81 |
| Sentence Splitter | ~2.4 µs | ~418 B | 11 |
| Levenshtein | ~430 ns | ~256 B | 2 |

## License

MIT — see the [original paper](https://arxiv.org/abs/2111.07068) for algorithm attribution.

## References

- Campos, R., Mangaravite, V., Pasquali, A., Jorge, A., Nunes, C., Jatowt, A. (2020). [YAKE! Keyword extraction from single documents using multiple local features](https://doi.org/10.1016/j.ins.2019.09.013). *Information Sciences*, 509, 257–289.
- [LIAAD/yake](https://github.com/LIAAD/yake) — original Python implementation

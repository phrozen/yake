// Package yake implements YAKE (Yet Another Keyword Extractor), an unsupervised,
// light-weight keyword extraction algorithm that rests on text statistical features
// extracted from single documents to select the most important keywords.
//
// YAKE scores candidate keywords using five features that capture casing, position,
// frequency, spread, and relatedness to context. A lower score indicates a more
// relevant keyword. The algorithm requires no external corpora, no training data,
// and only a stopword list for the target language.
//
// Basic usage:
//
//	cfg := yake.DefaultConfig()
//	y, err := yake.New(cfg)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	keywords := y.Extract(text, 10)
//	for _, kw := range keywords {
//	    fmt.Printf("%s (%s): %.4f\n", kw.Raw, kw.Keyword, kw.Score)
//	}
package yake

import (
	"math"
	"slices"
	"strings"
	"unicode/utf8"
)

// Yake is a keyword extractor configured for a specific language and
// set of extraction parameters. It is safe for concurrent use across
// multiple goroutines as long as Extract is called independently.
type Yake struct {
	config    Config
	stopWords *StopWords
}

// ResultItem represents a single extracted keyword.
type ResultItem struct {
	// Raw is the keyword as it first appeared in the source text,
	// with original casing and punctuation preserved.
	Raw string

	// Keyword is the lowercased, normalized form used for scoring.
	Keyword string

	// Score is the YAKE importance score. Lower values indicate
	// more important keywords. Scores are not bounded but typically
	// fall in (0.0, 1.0].
	Score float64
}

// New creates a [Yake] extractor from the given configuration.
// If [Config.StopWords] is nil, the built-in stopword list for
// [Config.Language] is loaded automatically.
func New(config Config) (*Yake, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	sw := config.StopWords
	if sw == nil {
		sw = PredefinedStopWords(config.Language)
		if sw == nil {
			sw = NewStopWords()
		}
	}
	return &Yake{config: config, stopWords: sw}, nil
}

// Extract extracts up to n keywords from text, sorted by score (best first).
//
// The pipeline:
//  1. Split text into sentences and words, classify each word by tag.
//  2. Build a co-occurrence graph within a sliding window.
//  3. Compute five per-term features (casing, position, frequency,
//     relatedness, spread) and combine them into an H score.
//  4. Generate n-gram candidates up to Ngrams size, weighting them
//     from constituent term scores and stopword edge probabilities.
//  5. Sort candidates by H and optionally remove near-duplicates.
func (y *Yake) Extract(text string, n int) []ResultItem {
	sentences := y.preprocessText(text)

	ctx, vocabulary := y.buildContextAndVocabulary(sentences)
	features := y.extractFeatures(ctx, vocabulary, sentences)
	candidateList := y.ngramSelection(sentences)
	y.candidateWeighting(features, ctx, candidateList)

	results := make([]ResultItem, 0, len(candidateList))
	for _, c := range candidateList {
		results = append(results, ResultItem{
			Raw:     strings.Join(c.raw, " "),
			Keyword: strings.Join(c.lcTerms, " "),
			Score:   c.score,
		})
	}

	sortResults(results)

	if y.config.RemoveDuplicates {
		results = removeDuplicates(y.config.DeduplicationThreshold, results, n)
	} else {
		if len(results) > n {
			results = results[:n]
		}
	}

	return results
}

func sortResults(results []ResultItem) {
	slices.SortStableFunc(results, func(a, b ResultItem) int {
		if a.Score < b.Score {
			return -1
		}
		if a.Score > b.Score {
			return 1
		}
		return 0
	})
}

type sentence struct {
	words         []string
	lcTerms       []string
	uqTerms       []string
	tags          []Tag
	stopwordFlags []bool
}

type occurrence struct {
	sentenceIdx int
	word        string
	tag         Tag
}

type termScore struct {
	tf    float64
	score float64
}

type termStats struct {
	tf          float64
	tfA         float64
	tfN         float64
	casing      float64
	position    float64
	frequency   float64
	relatedness float64
	sentences   float64
	score       float64
}

type candidate struct {
	occurrences int
	raw         []string
	lcTerms     []string
	uqTerms     []string
	score       float64
}

func (y *Yake) uniqueTerm(word string) string {
	return strings.ToLower(toSingle(word))
}

func (y *Yake) isStopword(lcTerm string) bool {
	if y.stopWords.Contains(lcTerm) {
		return true
	}
	singular := toSingle(lcTerm)
	if singular != lcTerm && y.stopWords.Contains(singular) {
		return true
	}
	count := 0
	for _, ch := range singular {
		if _, ok := y.config.Punctuation[ch]; !ok {
			count++
		}
	}
	return count < 3
}

func toSingle(s string) string {
	if s == "" {
		return s
	}
	last := s[len(s)-1]
	if last != 's' && last != 'S' {
		return s
	}
	r := []rune(s)
	if len(r) <= 3 {
		return s
	}
	return string(r[:len(r)-1])
}

func (y *Yake) preprocessText(text string) []sentence {
	strSentences := splitIntoSentences(text)
	sentences := make([]sentence, 0, len(strSentences))

	for _, s := range strSentences {
		words := splitIntoWords(s)
		n := len(words)
		lcTerms := make([]string, n)
		uqTerms := make([]string, n)
		tags := make([]Tag, n)
		stopwordFlags := make([]bool, n)

		for i, w := range words {
			lcTerms[i] = strings.ToLower(w)
			uqTerms[i] = y.uniqueTerm(w)
			tags[i] = Classify(w, i == 0, y.config.StrictCapital, y.config.Punctuation)
			stopwordFlags[i] = y.isStopword(lcTerms[i])
		}

		sentences = append(sentences, sentence{
			words:         words,
			lcTerms:       lcTerms,
			uqTerms:       uqTerms,
			tags:          tags,
			stopwordFlags: stopwordFlags,
		})
	}

	return sentences
}

func (y *Yake) buildContextAndVocabulary(sentences []sentence) (*Contexts, map[string][]occurrence) {
	ctx := NewContexts()
	vocab := make(map[string][]occurrence)

	for idx, s := range sentences {
		window := make([]struct {
			term string
			tag  Tag
		}, 0, y.config.WindowSize+1)

		for i, word := range s.words {
			tag := s.tags[i]
			term := s.uqTerms[i]

			if tag == TagPunctuation {
				window = window[:0]
				continue
			}

			vocab[term] = append(vocab[term], occurrence{
				sentenceIdx: idx,
				word:        word,
				tag:         tag,
			})

			if tag != TagDigit && tag != TagUnparsable {
				for _, w := range window {
					if w.tag == TagDigit || w.tag == TagUnparsable {
						continue
					}
					ctx.Track(w.term, term)
				}
			}

			if len(window) == y.config.WindowSize && y.config.WindowSize > 0 {
				window = window[1:]
			}
			window = append(window, struct {
				term string
				tag  Tag
			}{term, tag})
		}
	}

	return ctx, vocab
}

func (y *Yake) extractFeatures(ctx *Contexts, vocab map[string][]occurrence, sentences []sentence) map[string]termScore {
	candidateWords := make(map[string]string)
	for _, s := range sentences {
		for i, lc := range s.lcTerms {
			if s.tags[i] != TagPunctuation {
				candidateWords[lc] = s.uqTerms[i]
			}
		}
	}

	nonStopWords := make(map[string]float64)
	for lc, uq := range candidateWords {
		if !y.isStopword(lc) {
			if _, exists := nonStopWords[uq]; !exists {
				nonStopWords[uq] = float64(len(vocab[uq]))
			}
		}
	}

	nonStopFreqs := make([]float64, 0, len(nonStopWords))
	for _, freq := range nonStopWords {
		nonStopFreqs = append(nonStopFreqs, freq)
	}

	nswMean := meanFloat(nonStopFreqs)
	nswStd := stddevFloat(nonStopFreqs, nswMean)

	maxTF := 0.0
	for _, occs := range vocab {
		if float64(len(occs)) > maxTF {
			maxTF = float64(len(occs))
		}
	}

	features := make(map[string]termScore)

	for _, uq := range candidateWords {
		occs := vocab[uq]
		stats := termStats{tf: float64(len(occs))}

		for _, occ := range occs {
			if occ.tag == TagAcronym {
				stats.tfA++
			}
			if occ.tag == TagUppercase {
				stats.tfN++
			}
		}

		stats.casing = maxFloat(stats.tfA, stats.tfN)
		stats.casing /= 1.0 + math.Log(stats.tf)

		sentenceIDs := make([]int, 0)
		for _, occ := range occs {
			sentenceIDs = append(sentenceIDs, occ.sentenceIdx)
		}
		sentenceIDs = dedupInts(sentenceIDs)
		stats.position = math.Log(math.Log(3.0 + medianInt(sentenceIDs)))

		stats.frequency = stats.tf / (nswMean + nswStd)

		dl, dr := ctx.dispersionOf(uq)
		stats.relatedness = 1.0 + (dr+dl)*(stats.tf/maxTF)

		stats.sentences = float64(len(sentenceIDs)) / float64(len(sentences))

		stats.score = (stats.relatedness * stats.position) /
			(stats.casing + (stats.frequency / stats.relatedness) + (stats.sentences / stats.relatedness))

		features[uq] = termScore{tf: stats.tf, score: stats.score}
	}

	return features
}

func (y *Yake) ngramSelection(sentences []sentence) []*candidate {
	seen := make(map[string]*candidate)
	var ordered []*candidate
	ignored := make(map[string]struct{})

	for _, s := range sentences {
		length := len(s.words)

		for j := 0; j < length; j++ {
			if s.stopwordFlags[j] {
				continue
			}

			for k := j + 1; k <= length && k <= j+y.config.Ngrams; k++ {
				lcTerms := s.lcTerms[j:k]

				key := strings.Join(lcTerms, " ")
				if _, ok := ignored[key]; ok {
					continue
				}

				if !y.isCandidate(lcTerms, s.tags[j:k], s.stopwordFlags[k-1]) {
					ignored[key] = struct{}{}
				} else {
					if c, ok := seen[key]; ok {
						c.occurrences++
					} else {
						c := &candidate{
							lcTerms:     lcTerms,
							uqTerms:     s.uqTerms[j:k],
							raw:         s.words[j:k],
							occurrences: 1,
						}
						seen[key] = c
						ordered = append(ordered, c)
					}
				}
			}
		}
	}

	return ordered
}

func (y *Yake) isCandidate(lcTerms []string, tags []Tag, lastIsStopword bool) bool {
	for _, tag := range tags {
		if tag == TagDigit || tag == TagPunctuation || tag == TagUnparsable {
			return false
		}
	}

	if lastIsStopword {
		return false
	}

	totalChars := 0
	for _, term := range lcTerms {
		totalChars += utf8.RuneCountInString(term)
	}
	if totalChars < y.config.MinimumChars {
		return false
	}

	if y.config.OnlyAlphanumericHyphen {
		for _, term := range lcTerms {
			for _, ch := range term {
				if !((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-') {
					return false
				}
			}
		}
	}

	return true
}

func (y *Yake) candidateWeighting(features map[string]termScore, ctx *Contexts, candidates []*candidate) {
	for _, c := range candidates {
		uqTerms := c.uqTerms
		prod := 1.0
		sum := 0.0

		for j, lc := range c.lcTerms {
			uq := uqTerms[j]

			if y.isStopword(lc) {
				probPrev := 0.0
				if j > 0 {
					prevUq := uqTerms[j-1]
					if ft, ok := features[prevUq]; ok && ft.tf > 0 {
						probPrev = float64(ctx.casesTermIsFollowed(prevUq, uq)) / ft.tf
					}
				}

				probSucc := 0.0
				if j < len(uqTerms)-1 {
					nextUq := uqTerms[j+1]
					if ft, ok := features[nextUq]; ok && ft.tf > 0 {
						probSucc = float64(ctx.casesTermIsFollowed(uq, nextUq)) / ft.tf
					}
				}

				prob := probPrev * probSucc
				prod *= 1.0 + (1.0 - prob)
				sum -= 1.0 - prob
			} else if ft, ok := features[uq]; ok {
				prod *= ft.score
				sum += ft.score
			}
		}

		if sum <= -0.999999999 {
			sum = 0.999999999
		}

		c.score = prod / (float64(c.occurrences) * (1.0 + sum))
	}
}

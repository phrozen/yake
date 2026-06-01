package yake

import (
	"math"
	"slices"
)

// contextFreq tracks, for a given term, which terms appear before
// (follows) and after (followedBy) it in the sliding window.
type contextFreq struct {
	follows    map[string]int
	followedBy map[string]int
}

// Contexts tracks pairwise co-occurrence frequencies between terms
// within a configurable sliding window. It powers the relatedness
// feature (term dispersion) and the bidirectional edge probability
// used in n-gram stopword weighting.
type Contexts struct {
	m map[string]*contextFreq
}

// NewContexts returns an empty co-occurrence tracker.
func NewContexts() *Contexts {
	return &Contexts{m: make(map[string]*contextFreq)}
}

// Track records that left appears immediately before right in the text.
// Both directions are stored: right.follows[left] and left.followedBy[right].
func (c *Contexts) Track(left, right string) {
	if c.m[right] == nil {
		c.m[right] = &contextFreq{follows: make(map[string]int), followedBy: make(map[string]int)}
	}
	c.m[right].follows[left]++

	if c.m[left] == nil {
		c.m[left] = &contextFreq{follows: make(map[string]int), followedBy: make(map[string]int)}
	}
	c.m[left].followedBy[right]++
}

// casesTermIsFollowed returns how many times term appears immediately
// before by in the text. Used to compute stopword transition probabilities.
func (c *Contexts) casesTermIsFollowed(term, by string) int {
	if entry, ok := c.m[term]; ok {
		return entry.followedBy[by]
	}
	return 0
}

// dispersionOf computes how varied a term's left and right neighbors are.
// A value near 1 means the term appears with many distinct neighbors
// (highly dispersive); near 0 means it always co-occurs with the same
// words (a fixed expression). distinct(neighbors) / total(neighbors).
func (c *Contexts) dispersionOf(term string) (left, right float64) {
	entry, ok := c.m[term]
	if !ok {
		return 0, 0
	}

	if len(entry.follows) == 0 {
		left = 0
	} else {
		total := 0
		for _, v := range entry.follows {
			total += v
		}
		left = float64(len(entry.follows)) / float64(total)
	}

	if len(entry.followedBy) == 0 {
		right = 0
	} else {
		total := 0
		for _, v := range entry.followedBy {
			total += v
		}
		right = float64(len(entry.followedBy)) / float64(total)
	}

	return left, right
}

// medianInt returns the median of a slice of integers. A copy is sorted
// internally; the original is not mutated.
func medianInt(slice []int) float64 {
	if len(slice) == 0 {
		return 0
	}
	sorted := make([]int, len(slice))
	copy(sorted, slice)
	slices.Sort(sorted)

	n := len(sorted)
	if n%2 == 0 {
		return float64(sorted[n/2-1]+sorted[n/2]) / 2.0
	}
	return float64(sorted[n/2])
}

// dedupInts returns a slice with duplicate values removed, preserving
// first-occurrence order.
func dedupInts(ints []int) []int {
	seen := make(map[int]struct{})
	result := make([]int, 0, len(ints))
	for _, i := range ints {
		if _, ok := seen[i]; !ok {
			seen[i] = struct{}{}
			result = append(result, i)
		}
	}
	return result
}

// maxFloat returns the larger of two float64s.
func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

// meanFloat returns the arithmetic mean of a slice.
func meanFloat(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

// stddevFloat returns the population standard deviation of a slice
// given its pre-computed mean. Returns 0 for fewer than 2 values.
func stddevFloat(vals []float64, mean float64) float64 {
	if len(vals) < 2 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		diff := v - mean
		sum += diff * diff
	}
	return math.Sqrt(sum / float64(len(vals)))
}

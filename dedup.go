package yake

// levenshteinDistance returns the minimum number of single-character
// edits (insertions, deletions, substitutions) required to transform
// a into b. Uses an O(n*m) dynamic programming approach with only two
// rows to keep memory usage linear.
func levenshteinDistance(a, b string) int {
	la, lb := len([]rune(a)), len([]rune(b))
	if la == 0 {
		return lb
	}
	if lb == 0 {
		return la
	}

	ra, rb := []rune(a), []rune(b)

	prev := make([]int, lb+1)
	curr := make([]int, lb+1)

	for j := 0; j <= lb; j++ {
		prev[j] = j
	}

	for i := 1; i <= la; i++ {
		curr[0] = i
		for j := 1; j <= lb; j++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			curr[j] = min3(prev[j]+1, curr[j-1]+1, prev[j-1]+cost)
		}
		prev, curr = curr, prev
	}

	return prev[lb]
}

func min3(a, b, c int) int {
	if a < b {
		if a < c {
			return a
		}
		return c
	}
	if b < c {
		return b
	}
	return c
}

// levenshteinRatio returns a normalized similarity between seq1 and
// seq2 in [0.0, 1.0], where 1.0 means identical strings. The distance
// is always computed with the shorter string as the source to bias
// toward subsequence matching, which is more useful for keyword
// deduplication than a symmetric metric.
func levenshteinRatio(seq1, seq2 string) float64 {
	var distance int
	if len([]rune(seq1)) <= len([]rune(seq2)) {
		distance = levenshteinDistance(seq1, seq2)
	} else {
		distance = levenshteinDistance(seq2, seq1)
	}
	length := len([]rune(seq1))
	if len([]rune(seq2)) > length {
		length = len([]rune(seq2))
	}
	if length == 0 {
		return 1.0
	}
	return 1.0 - float64(distance)/float64(length)
}

// removeDuplicates filters results by Levenshtein similarity, keeping
// only the first occurrence of each distinct keyword. Candidates are
// processed in score order (best first), so a lower-scored duplicate
// is discarded in favor of the higher-scored original. Stops early
// when n results have been collected.
func removeDuplicates(threshold float64, results []ResultItem, n int) []ResultItem {
	unique := make([]ResultItem, 0, n)

	for _, res := range results {
		if len(unique) >= n {
			break
		}

		isDuplicate := false
		for _, it := range unique {
			if levenshteinRatio(it.Keyword, res.Keyword) > threshold {
				isDuplicate = true
				break
			}
		}

		if !isDuplicate {
			unique = append(unique, res)
		}
	}

	return unique
}

package yake

import "testing"

func BenchmarkExtractShort(b *testing.B) {
	y, err := New(DefaultConfig())
	if err != nil {
		b.Fatal(err)
	}
	text := "Google is acquiring data science community Kaggle. Machine learning competitions are hosted on this platform."
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		y.Extract(text, 10)
	}
}

func BenchmarkExtractMedium(b *testing.B) {
	y, err := New(DefaultConfig())
	if err != nil {
		b.Fatal(err)
	}
	text := `This is your weekly newsletter! Hundreds of great deals - everything from men's fashion
to high-tech drones! We have exciting offers for everyone interested in machine learning and data science.
Artificial intelligence is transforming how we approach complex problems in the modern world.
Companies like Google, Amazon, and Microsoft are investing heavily in cutting-edge research.
The field of natural language processing has made tremendous progress in recent years.
Deep learning models are now capable of understanding complex patterns in unstructured data.`
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		y.Extract(text, 10)
	}
}

func BenchmarkTokenizer(b *testing.B) {
	text := "Google is acquiring data-science community Kaggle. It's truly high-tech! Don't you think so? 3.14 is pi."
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitIntoWords(text)
	}
}

func BenchmarkSentenceSplitter(b *testing.B) {
	text := "Google is acquiring data science community Kaggle.\n\nMachine learning competitions are hosted on this platform.\n\nIt's an exciting time for AI enthusiasts everywhere."
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		splitIntoSentences(text)
	}
}

func BenchmarkLevenshtein(b *testing.B) {
	a := "machine learning"
	b2 := "machin learnin"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		levenshteinDistance(a, b2)
	}
}

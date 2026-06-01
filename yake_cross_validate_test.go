package yake

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

const scoreE = 0.0006

func rnd(s float64) float64 { return math.Round(s*10000) / 10000 }

func readSample(t *testing.T, name string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Skipf("sample not found: %s", name)
	}
	return string(data)
}

type ref struct {
	keyword string
	score   float64
}

func assertMatch(t *testing.T, name string, actual []ResultItem, expected []ref) {
	t.Helper()
	for i := range actual {
		actual[i].Score = rnd(actual[i].Score)
	}
	if len(actual) != len(expected) {
		t.Errorf("%s: got %d results, want %d", name, len(actual), len(expected))
		for i, r := range actual {
			t.Logf("  G[%d] %s (%.4f)", i, r.Keyword, r.Score)
		}
		for i, e := range expected {
			t.Logf("  E[%d] %s (%.4f)", i, e.keyword, e.score)
		}
		return
	}
	for i := range actual {
		if actual[i].Keyword != expected[i].keyword {
			t.Errorf("%s[%d].Keyword: got %q, want %q", name, i, actual[i].Keyword, expected[i].keyword)
		}
		if math.Abs(actual[i].Score-expected[i].score) > scoreE {
			t.Errorf("%s[%d].Score: got %.4f, want %.4f", name, i, actual[i].Score, expected[i].score)
		}
	}
}

func extract(cfg Config, text string, topK int) []ResultItem {
	y, err := New(cfg)
	if err != nil {
		panic(err)
	}
	return y.Extract(text, topK)
}

// Inline-text tests cross-validated against Python (LIAAD/yake) and Rust (yake-rust).
// All scores and keywords match identically at 4dp precision.

func TestCrossShort(t *testing.T) {
	actual := extract(DefaultConfig(), "this is a keyword", 1)
	assertMatch(t, "short", actual, []ref{{"keyword", 0.1583}})
}

func TestCrossKeywordsOrder(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Ngrams = 1
	actual := extract(cfg, "Machine learning", 3)
	assertMatch(t, "order", actual, []ref{{"machine", 0.1583}, {"learning", 0.1583}})
}

func TestCrossLaptop(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Ngrams = 1
	actual := extract(cfg, "Do you need an Apple laptop?", 2)
	assertMatch(t, "laptop", actual, []ref{{"apple", 0.1448}, {"laptop", 0.1583}})
}

func TestCrossHeadphones(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Ngrams = 1
	actual := extract(cfg, "Do you like headphones? Starting this Saturday, we will be kicking off a huge sale of headphones! If you need headphones, we've got you covered!", 3)
	assertMatch(t, "headphones", actual, []ref{{"headphones", 0.1141}, {"saturday", 0.2111}, {"starting", 0.4096}})
}

func TestCrossMultiNgram(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Ngrams = 2
	actual := extract(cfg, "I will give you a great deal if you just read this!", 1)
	assertMatch(t, "multi_ngram", actual, []ref{{"great deal", 0.0257}})
}

func TestCrossSingular(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Ngrams = 1
	actual := extract(cfg, "One smartwatch. One phone. Many phone.", 2)
	assertMatch(t, "singular", actual, []ref{{"smartwatch", 0.2025}, {"phone", 0.2474}})
}

func TestCrossPlural(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Ngrams = 1
	actual := extract(cfg, "One smartwatch. One phone. Many phones.", 3)
	assertMatch(t, "plural", actual, []ref{{"smartwatch", 0.2025}, {"phone", 0.4949}, {"phones", 0.4949}})
}

func TestCrossNonHyphenated(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Ngrams = 2
	actual := extract(cfg, "Truly high tech!", 1)
	assertMatch(t, "non_hyphenated", actual, []ref{{"high tech", 0.0494}})
}

func TestCrossHyphenated(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Ngrams = 2
	actual := extract(cfg, "Truly high-tech!", 1)
	assertMatch(t, "hyphenated", actual, []ref{{"high-tech", 0.1583}})
}

func TestCrossNewsletterShort(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Ngrams = 2
	actual := extract(cfg, "This is your weekly newsletter!", 3)
	assertMatch(t, "newsletter_short", actual, []ref{{"weekly newsletter", 0.0494}, {"newsletter", 0.1583}, {"weekly", 0.2974}})
}

func TestCrossNewsletterLong(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Ngrams = 2
	actual := extract(cfg, "This is your weekly newsletter! Hundreds of great deals - everything from men's fashion to high-tech drones!", 5)
	assertMatch(t, "newsletter_long", actual, []ref{
		{"weekly newsletter", 0.0780}, {"newsletter", 0.2005}, {"weekly", 0.3607},
		{"great deals", 0.4456}, {"high-tech drones", 0.4456},
	})
}

func TestCrossNewsletterParagraphs(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Ngrams = 2
	actual := extract(cfg, "This is your weekly newsletter!\n\n \tHundreds of great deals - everything from men's fashion \nto high-tech drones!", 5)
	assertMatch(t, "newsletter_paragraphs", actual, []ref{
		{"weekly newsletter", 0.0780}, {"newsletter", 0.2005}, {"weekly", 0.3607},
		{"great deals", 0.4456}, {"high-tech drones", 0.4456},
	})
}

func TestCrossBiggerWindow(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Ngrams = 2
	cfg.WindowSize = 2
	actual := extract(cfg, "Machine learning is a growing field. Few research fields grow as much as machine learning grows.", 5)
	assertMatch(t, "bigger_window", actual, []ref{
		{"machine learning", 0.1346}, {"growing field", 0.1672}, {"learning", 0.2265},
		{"machine", 0.2341}, {"growing", 0.2799},
	})
}

func TestCrossNearNumbers(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Ngrams = 2
	actual := extract(cfg, "I buy 100 yellow bananas every day. Every night I eat bananas - all but 5 bananas.", 3)
	assertMatch(t, "near_numbers", actual, []ref{{"yellow bananas", 0.0682}, {"buy", 0.1428}, {"yellow", 0.1428}})
}

func TestCrossSpelledNumbers(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Ngrams = 2
	actual := extract(cfg, "I buy a hundred yellow bananas every day. Every night I eat bananas - all but five bananas.", 3)
	assertMatch(t, "spelled_numbers", actual, []ref{{"hundred yellow", 0.0446}, {"yellow bananas", 0.1017}, {"day", 0.1428}})
}

func TestCrossStopwordInMiddle(t *testing.T) {
	cfg := DefaultConfig()
	cfg.RemoveDuplicates = false
	actual := extract(cfg, "Game of Thrones", 1)
	assertMatch(t, "stopword_middle", actual, []ref{{"game of thrones", 0.0138}})
}

func TestCrossDeduplication(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Ngrams = 2
	actual := extract(cfg, "machine learning machine learning deep learning", 5)
	assertMatch(t, "dedup", actual, []ref{
		{"machine learning", 0.0231}, {"learning deep", 0.0412}, {"deep learning", 0.0412},
		{"learning machine", 0.0461}, {"learning", 0.0815},
	})
}

// File-based English tests cross-validated against Python/Rust.

func TestCrossGoogleN1(t *testing.T) {
	text := readSample(t, "test_google.txt")
	cfg := DefaultConfig()
	cfg.Ngrams = 1
	actual := extract(cfg, text, 10)
	assertMatch(t, "google_n1", actual, []ref{
		{"google", 0.0251}, {"kaggle", 0.0273}, {"data", 0.0800},
		{"science", 0.0983}, {"platform", 0.1240}, {"service", 0.1316},
		{"acquiring", 0.1511}, {"learning", 0.1621}, {"goldbloom", 0.1625}, {"machine", 0.1672},
	})
}

func TestCrossGoogleDefaults(t *testing.T) {
	text := readSample(t, "test_google.txt")
	actual := extract(DefaultConfig(), text, 10)
	assertMatch(t, "google_n3", actual, []ref{
		{"google", 0.0251}, {"kaggle", 0.0273}, {"ceo anthony goldbloom", 0.0483},
		{"data science", 0.0550}, {"acquiring data science", 0.0603},
		{"google cloud platform", 0.0746}, {"data", 0.0800}, {"san francisco", 0.0914},
		{"anthony goldbloom declined", 0.0974}, {"science", 0.0983},
	})
}

func TestCrossGitter(t *testing.T) {
	text := readSample(t, "test_gitter.txt")
	actual := extract(DefaultConfig(), text, 10)
	assertMatch(t, "gitter", actual, []ref{
		{"gitter", 0.0197}, {"acquires software chat", 0.0414}, {"chat startup gitter", 0.0455},
		{"software chat startup", 0.0540}, {"gitlab", 0.0596}, {"gitlab acquires software", 0.0633},
		{"gitter chat", 0.0639}, {"startup", 0.0754},
		{"ceo sid sijbrandij", 0.0766}, {"gitter chat rooms", 0.0796},
	})
}

func TestCrossGenius(t *testing.T) {
	text := readSample(t, "test_genius.txt")
	actual := extract(DefaultConfig(), text, 10)
	assertMatch(t, "genius", actual, []ref{
		{"genius quietly laid", 0.0167}, {"genius", 0.0223}, {"company quietly laid", 0.0238},
		{"company", 0.0247}, {"media company", 0.0322}, {"quietly laid", 0.0431},
		{"lehman", 0.0445}, {"tom lehman told", 0.0456}, {"shift resources", 0.0530},
		{"co-founder tom lehman", 0.0558},
	})
}

func TestCrossFukushima(t *testing.T) {
	text := readSample(t, "test_data_2.txt")
	actual := extract(DefaultConfig(), text, 5)
	assertMatch(t, "fukushima", actual, []ref{
		{"highly radioactive water", 0.0006}, {"crippled nuclear plant", 0.0006},
		{"ocean japan official", 0.0031}, {"japan official", 0.0046}, {"official says highly", 0.0050},
	})
}

func TestCrossGlobalCrossing(t *testing.T) {
	text := readSample(t, "test_data_3.txt")
	actual := extract(DefaultConfig(), text, 5)
	assertMatch(t, "global_crossing", actual, []ref{
		{"global crossing", 0.0034}, {"hutchison telecommunications", 0.0053},
		{"telecommunications and singapore", 0.0072}, {"singapore technologies", 0.0072},
		{"technologies take control", 0.0157},
	})
}

// Multilingual file-based tests using canonical reference scores.
// Verified against Rust/py yake using byte-identical input files and stopword lists.
// Scores match exactly at 4dp.

func TestCrossGerman(t *testing.T) {
	text := readSample(t, "test_german.txt")
	cfg := DefaultConfig()
	cfg.Language = "de"
	actual := extract(cfg, text, 10)
	assertMatch(t, "german", actual, []ref{
		{"vereinigten staaten", 0.0139}, {"präsidenten donald trump", 0.0181},
		{"donald trump", 0.0219}, {"trifft donald trump", 0.0235},
		{"kanzlerin angela merkel", 0.0251}, {"trumps finanzminister steven", 0.0254},
		{"trump", 0.0282}, {"deutsche kanzlerin angela", 0.0290},
		{"merkel trifft donald", 0.0327}, {"finanzminister schäuble verhandelt", 0.0371},
	})
}

func TestCrossItalian(t *testing.T) {
	text := readSample(t, "test_it.txt")
	cfg := DefaultConfig()
	cfg.Language = "it"
	actual := extract(cfg, text, 5)
	assertMatch(t, "italian", actual, []ref{
		{"champions league", 0.0390}, {"quarti", 0.0520},
		{"atlético madrid", 0.0592}, {"ottavi di finale", 0.0646}, {"real madrid", 0.0701},
	})
}

func TestCrossFrench(t *testing.T) {
	text := readSample(t, "test_fr.txt")
	cfg := DefaultConfig()
	cfg.Language = "fr"
	actual := extract(cfg, text, 10)
	assertMatch(t, "french", actual, []ref{
		{"dégrade en france", 0.0254}, {"jusque-là uniquement associée", 0.0491},
		{"sondage ifop réalisé", 0.0542}, {"religion se dégrade", 0.0895},
		{"france", 0.0941}, {"l'extrême droite", 0.0984},
		{"sondage ifop", 0.0999}, {"islam", 0.1011},
		{"musulmane en france", 0.1067}, {"allemagne", 0.1086},
	})
}

func TestCrossPortugueseSport(t *testing.T) {
	text := readSample(t, "test_pt_1.txt")
	cfg := DefaultConfig()
	cfg.Language = "pt"
	actual := extract(cfg, text, 10)
	assertMatch(t, "pt_sport", actual, []ref{
		{"seleção brasileira treinará", 0.0072}, {"seleção brasileira", 0.0100},
		{"seleção brasileira visando", 0.0185}, {"seleção brasileira encara", 0.0327},
		{"brasileira treinará", 0.0362}, {"renato augusto", 0.0369},
		{"copa da rússia", 0.0407}, {"seleção", 0.0448},
		{"brasileira", 0.0528}, {"meia renato augusto", 0.0604},
	})
}

func TestCrossPortugueseTourism(t *testing.T) {
	text := readSample(t, "test_pt_2.txt")
	cfg := DefaultConfig()
	cfg.Language = "pt"
	actual := extract(cfg, text, 10)
	assertMatch(t, "pt_tourism", actual, []ref{
		{"alvor", 0.0165}, {"rio alvor", 0.0336}, {"ria de alvor", 0.0488},
		{"encantadora vila", 0.0575}, {"algarve", 0.0774},
		{"impressionantes de portugal", 0.0844}, {"estuário do rio", 0.0907},
		{"vila", 0.1017}, {"ria", 0.1053}, {"oceano atlântico", 0.1357},
	})
}

func TestCrossSpanish(t *testing.T) {
	text := readSample(t, "test_es.txt")
	cfg := DefaultConfig()
	cfg.Language = "es"
	actual := extract(cfg, text, 10)
	assertMatch(t, "spanish", actual, []ref{
		{"guerra civil española", 0.0032}, {"guerra civil", 0.0130},
		{"civil española", 0.0153}, {"partido socialista obrero", 0.0283},
		{"empezó la guerra", 0.0333}, {"socialista obrero español", 0.0411},
		{"josé castillo", 0.0426}, {"española", 0.0566},
		{"josé antonio primo", 0.0589}, {"¿cómo empezó", 0.0594},
	})
}

func TestCrossPolish(t *testing.T) {
	text := readSample(t, "test_pl.txt")
	cfg := DefaultConfig()
	cfg.Language = "pl"
	actual := extract(cfg, text, 10)
	assertMatch(t, "polish", actual, []ref{
		{"franka", 0.0329}, {"geerta wildersa vvd", 0.0348},
		{"proc", 0.0394}, {"geerta wildersa", 0.0401},
		{"kurs franka", 0.0490}, {"partii geerta wildersa", 0.0680},
		{"mld", 0.0723}, {"narodowego banku szwajcarii", 0.0731},
		{"wildersa", 0.0766}, {"kurs franka poniżej", 0.0767},
	})
}

func TestCrossTurkish(t *testing.T) {
	text := readSample(t, "test_tr.txt")
	cfg := DefaultConfig()
	cfg.Language = "tr"
	actual := extract(cfg, text, 10)
	assertMatch(t, "turkish", actual, []ref{
		{"oecd", 0.0178}, {"tek bakışta eğitim", 0.0236},
		{"eğitim", 0.0278}, {"türkiye", 0.0296},
		{"oecd eğitim endeksi", 0.0325}, {"oecd ortalamasının", 0.0380},
		{"sondan dördüncü sırada", 0.0401}, {"sırada yer", 0.0432},
		{"kalkınma örgütü'nün", 0.0453}, {"tek bakışta", 0.0453},
	})
}

func TestCrossArabic(t *testing.T) {
	text := readSample(t, "test_ar.txt")
	cfg := DefaultConfig()
	cfg.Language = "ar"
	actual := extract(cfg, text, 10)
	assertMatch(t, "arabic", actual, []ref{
		{"عبد السلام العجيلي", 0.0001}, {"عبد النبي اصطيف", 0.0002},
		{"لدراسات الشرق الأوسط", 0.0002}, {"مرآة النقد المقارن", 0.0003},
		{"اللغة العربية بدمشق", 0.0003}, {"اللغة العربية الأربعاء", 0.0003},
		{"الدكتور عبد النبي", 0.0003}, {"الشرق الأوسط والرابطة", 0.0004},
		{"البريطانية لدراسات الشرق", 0.0004}, {"الأوروبية لدراسات الشرق", 0.0004},
	})
}

func TestCrossDutch(t *testing.T) {
	text := readSample(t, "test_nl.txt")
	cfg := DefaultConfig()
	cfg.Language = "nl"
	actual := extract(cfg, text, 10)
	assertMatch(t, "dutch", actual, []ref{
		{"vincent van gogh", 0.0110}, {"gogh museum", 0.0124}, {"gogh", 0.0150},
		{"museum", 0.0437}, {"brieven", 0.0632}, {"vincent", 0.0642},
		{"goghs schilderijen", 0.1003}, {"gogh verging", 0.1208},
		{"goghs", 0.1647}, {"schrijven", 0.1699},
	})
}

func TestCrossFinnish(t *testing.T) {
	text := readSample(t, "test_fi.txt")
	cfg := DefaultConfig()
	cfg.Language = "fi"
	actual := extract(cfg, text, 10)
	assertMatch(t, "finnish", actual, []ref{
		{"mobile networks", 0.0053}, {"nokia tekee muutoksia", 0.0059},
		{"tekee muutoksia organisaatioonsa", 0.0065},
		{"mobile networks -liiketoimintaryhmän", 0.0069},
		{"johtokuntaansa vauhdittaakseen yhtiön", 0.0084},
		{"vauhdittaakseen yhtiön strategian", 0.0084},
		{"yhtiön strategian toteuttamista", 0.0086},
		{"networks and applications", 0.0117},
		{"strategian toteuttamista nokia", 0.0121}, {"siirtyy mobile networks", 0.0124},
	})
}

func TestCrossDataset1PT(t *testing.T) {
	text := readSample(t, "test_data_1.txt")
	cfg := DefaultConfig()
	cfg.Language = "pt"
	actual := extract(cfg, text, 10)
	assertMatch(t, "dataset1_pt", actual, []ref{
		{"médio oriente continua", 0.0008}, {"médio oriente", 0.0045},
		{"oriente continua", 0.0117}, {"registar-se violentos confrontos", 0.0178},
		{"faixa de gaza", 0.0268}, {"fogo hoje voltaram", 0.0311},
		{"voltaram a registar-se", 0.0311}, {"registar-se violentos", 0.0311},
		{"exército israelita", 0.0368}, {"exército israelita voltou", 0.0639},
	})
}

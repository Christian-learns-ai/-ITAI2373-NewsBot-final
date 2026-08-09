from collections import Counter
from nltk.tokenize import word_tokenize
import spacy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class IntelligentSummarizer:
    """Abstractive summarization via BART with extractive sentence fallbacks."""

    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.nlp = spacy.load("en_core_web_sm")

    def _extractive_fallback(self, article_text: str, num_sentences: int = 3) -> str:
        doc = self.nlp(article_text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

        if not sentences:
            return article_text

        words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
        word_freq = Counter(words)

        sentence_scores = {}
        for sent in sentences:
            sent_doc = self.nlp(sent)
            score = sum(word_freq[token.text.lower()] for token in sent_doc if token.text.lower() in word_freq)
            sentence_scores[sent] = score / (len(sent_doc) + 1)

        top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        ordered_summary = [s for s in sentences if s in top_sentences]
        return " ".join(ordered_summary)

    def summarize_article(self, article_text: str, summary_type: str = "balanced") -> str:
        if not article_text or len(article_text.strip()) < 50:
            return article_text

        length_params = {
            "brief": {"max_length": 60, "min_length": 25},
            "balanced": {"max_length": 130, "min_length": 60},
            "detailed": {"max_length": 250, "min_length": 120},
        }
        params = length_params.get(summary_type, length_params["balanced"])

        try:
            inputs = self.tokenizer(
                article_text, return_tensors="pt", max_length=1024, truncation=True
            )
            summary_ids = self.model.generate(
                inputs.input_ids,
                max_length=params["max_length"],
                min_length=params["min_length"],
                num_beams=4,
                early_stopping=True,
            )
            return self.tokenizer.decode(
                summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        except Exception:
            num_sents = 2 if summary_type == "brief" else (4 if summary_type == "balanced" else 6)
            return self._extractive_fallback(article_text, num_sentences=num_sents)

    def summarize_multiple_articles(self, articles: list[str], focus_topic: str = None) -> str:
        if not articles:
            return ""
        filtered_articles = articles
        if focus_topic:
            filtered_articles = [art for art in articles if focus_topic.lower() in art.lower()] or articles

        individual_summaries = [
            self.summarize_article(art, summary_type="brief") for art in filtered_articles[:5]
        ]
        combined_text = "\n".join(individual_summaries)
        return self.summarize_article(combined_text, summary_type="balanced")

    def assess_summary_quality(self, original_text: str, summary: str) -> dict:
        orig_words = len(original_text.split())
        summ_words = len(summary.split())
        compression_ratio = round(summ_words / max(orig_words, 1), 4)

        orig_tokens = set(word_tokenize(original_text.lower()))
        summ_tokens = set(word_tokenize(summary.lower()))
        overlap_score = len(orig_tokens.intersection(summ_tokens)) / max(len(summ_tokens), 1)

        return {
            "original_word_count": orig_words,
            "summary_word_count": summ_words,
            "compression_ratio": compression_ratio,
            "content_coverage_overlap": round(overlap_score, 4),
        }

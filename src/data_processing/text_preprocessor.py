import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    """Text cleaning, normalization, and tokenization pipeline."""

    def __init__(self, spacy_model="en_core_web_sm"):
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        self.stop_words = set(stopwords.words("english"))
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception:
            self.nlp = None

    def clean(self, text: str) -> str:
        if not text:
            return ""
        return text.strip()

    def tokenize_and_clean(self, text: str) -> list[str]:
        tokens = word_tokenize(text.lower())
        return [
            token for token in tokens
            if token.isalpha() and token not in self.stop_words and len(token) > 2
        ]

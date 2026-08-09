from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    """TF-IDF and custom feature extraction for text documents."""

    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words="english", ngram_range=ngram_range
        )

    def fit_transform(self, raw_documents):
        return self.vectorizer.fit_transform(raw_documents)

    def transform(self, raw_documents):
        return self.vectorizer.transform(raw_documents)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

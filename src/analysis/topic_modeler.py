import nltk
import pandas as pd
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

class TopicDiscoveryEngine:
    """Topic modeling engine using Gensim LDA and scikit-learn NMF."""

    def __init__(self, n_topics=10, method="lda"):
        self.n_topics = n_topics
        self.method = method.lower()
        self.dictionary = None
        self.corpus = None
        self.model = None
        self.vectorizer = None

    def preprocess_documents(self, documents: list[str]) -> list[list[str]]:
        stop_words = set(stopwords.words("english"))
        tokenized_docs = []
        for doc in documents:
            tokens = word_tokenize(doc.lower())
            cleaned_tokens = [
                token for token in tokens
                if token.isalpha() and token not in stop_words and len(token) > 2
            ]
            tokenized_docs.append(cleaned_tokens)
        return tokenized_docs

    def fit_topics(self, documents: list[str]):
        processed_docs = self.preprocess_documents(documents)

        if self.method == "lda":
            self.dictionary = corpora.Dictionary(processed_docs)
            self.dictionary.filter_extremes(no_below=2, no_above=0.8)
            self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]

            self.model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=self.n_topics,
                random_state=42,
                passes=10,
                alpha="auto",
            )
            coherence_model = CoherenceModel(
                model=self.model, texts=processed_docs, dictionary=self.dictionary, coherence="c_v"
            )
            return coherence_model.get_coherence()

        elif self.method == "nmf":
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            self.model = NMF(n_components=self.n_topics, random_state=42, max_iter=200)
            self.model.fit(tfidf_matrix)

    def get_article_topics(self, article_text: str) -> list[tuple[int, float]]:
        processed = self.preprocess_documents([article_text])[0]
        if self.method == "lda" and self.model and self.dictionary:
            bow = self.dictionary.doc2bow(processed)
            return self.model.get_document_topics(bow)
        elif self.method == "nmf" and self.model and self.vectorizer:
            tfidf = self.vectorizer.transform([article_text])
            weights = self.model.transform(tfidf)[0]
            total_weight = sum(weights) or 1.0
            return [(idx, float(w / total_weight)) for idx, w in enumerate(weights)]
        return []

    def track_topic_trends(self, articles_with_dates: list[dict]) -> pd.DataFrame:
        records = []
        for item in articles_with_dates:
            text = item.get("text", "")
            date = item.get("date")
            topic_dist = dict(self.get_article_topics(text))
            row = {"date": pd.to_datetime(date)}
            for t_idx in range(self.n_topics):
                row[f"Topic_{t_idx}"] = topic_dist.get(t_idx, 0.0)
            records.append(row)

        df = pd.DataFrame(records)
        if not df.empty and "date" in df.columns:
            df = df.sort_values("date").set_index("date")
            return df.resample("W").mean().fillna(0)
        return df

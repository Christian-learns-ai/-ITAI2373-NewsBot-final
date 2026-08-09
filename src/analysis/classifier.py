import numpy as np
import sklearn.linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

class AdvancedNewsClassifier:
    """Enhanced news classification with confidence scoring and multi-label support."""

    def __init__(self, model_type="tfidf_lr"):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(1, 2)
        )
        self.model = sklearn.linear_model.LogisticRegression(class_weight="balanced")
        self.categories = []

        if self.model_type == "transformer":
            self.zero_shot = pipeline(
                "zero-shot-classification", model="facebook/bart-large-mnli"
            )

    def train(self, X_train, y_train):
        if self.model_type == "tfidf_lr":
            X_vec = self.vectorizer.fit_transform(X_train)
            self.model.fit(X_vec, y_train)
            self.categories = list(self.model.classes_)

    def predict_with_confidence(self, article_text: str) -> dict:
        if self.model_type == "transformer":
            candidate_labels = ["Politics", "Technology", "Business", "Sports", "Entertainment", "Health"]
            res = self.zero_shot(article_text, candidate_labels)
            return {
                "primary_category": res["labels"][0],
                "confidence_score": round(res["scores"][0], 4),
                "alternative_categories": {
                    label: round(score, 4) for label, score in zip(res["labels"][1:], res["scores"][1:])
                },
            }

        if not self.categories:
            return {"primary_category": "Unclassified", "confidence_score": 0.0, "alternative_categories": {}}

        X_vec = self.vectorizer.transform([article_text])
        probabilities = self.model.predict_proba(X_vec)[0]
        sorted_indices = np.argsort(probabilities)[::-1]

        primary_idx = sorted_indices[0]
        primary_category = self.categories[primary_idx]
        confidence_score = float(probabilities[primary_idx])

        alternatives = {
            self.categories[idx]: round(float(probabilities[idx]), 4)
            for idx in sorted_indices[1:]
        }

        return {
            "primary_category": primary_category,
            "confidence_score": round(confidence_score, 4),
            "alternative_categories": alternatives,
        }

    def explain_prediction(self, article_text: str) -> dict:
        if self.model_type == "transformer":
            return {"explanation": "Zero-shot classification pipeline used (transformer attention)."}

        if not self.categories:
            return {"predicted_category": "Untrained", "top_influential_words": []}

        X_vec = self.vectorizer.transform([article_text])
        probabilities = self.model.predict_proba(X_vec)[0]
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.categories[predicted_idx]

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        coefs = self.model.coef_[predicted_idx]

        nonzero_indices = X_vec.nonzero()[1]
        word_scores = [
            (feature_names[i], X_vec[0, i] * coefs[i]) for i in nonzero_indices
        ]
        word_scores.sort(key=lambda x: x[1], reverse=True)

        return {
            "predicted_category": predicted_class,
            "top_influential_words": word_scores[:10],
        }

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

class SentimentEvolutionTracker:
    """Advanced sentiment analysis with temporal, emotional, and anomaly tracking."""

    def __init__(self, model_type="vader"):
        self.model_type = model_type
        nltk.download("vader_lexicon", quiet=True)
        self.vader = SentimentIntensityAnalyzer()

        if self.model_type == "transformer":
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
            )

    def analyze_sentiment(self, article_text: str) -> dict:
        scores = self.vader.polarity_scores(article_text)
        compound = scores["compound"]

        if compound >= 0.05:
            overall = "positive"
        elif compound <= -0.05:
            overall = "negative"
        else:
            overall = "neutral"

        result = {
            "overall_sentiment": overall,
            "compound_score": round(compound, 4),
            "polarity_breakdown": {
                "positive": round(scores["pos"], 4),
                "negative": round(scores["neg"], 4),
                "neutral": round(scores["neu"], 4),
            },
            "emotions": {},
        }

        if self.model_type == "transformer":
            emotions_res = self.emotion_classifier(article_text[:512])[0]
            result["emotions"] = {
                item["label"]: round(item["score"], 4) for item in emotions_res
            }

        return result

    def track_sentiment_over_time(self, articles_with_dates: list[dict]) -> pd.DataFrame:
        records = []
        for item in articles_with_dates:
            text = item.get("text", "")
            date = item.get("date")
            res = self.analyze_sentiment(text)
            records.append({
                "date": pd.to_datetime(date),
                "compound": res["compound_score"],
                "pos": res["polarity_breakdown"]["positive"],
                "neg": res["polarity_breakdown"]["negative"],
                "neu": res["polarity_breakdown"]["neutral"],
            })

        df = pd.DataFrame(records)
        if not df.empty and "date" in df.columns:
            df = df.sort_values("date").set_index("date")
            return df.resample("D").mean().fillna(0)
        return df

    def detect_sentiment_anomalies(self, sentiment_timeline: pd.DataFrame, threshold_std: float = 2.0) -> pd.DataFrame:
        if sentiment_timeline.empty or "compound" not in sentiment_timeline.columns:
            return pd.DataFrame()

        mean_val = sentiment_timeline["compound"].mean()
        std_val = sentiment_timeline["compound"].std()

        if std_val == 0:
            sentiment_timeline["z_score"] = 0
            sentiment_timeline["is_anomaly"] = False
            return sentiment_timeline

        sentiment_timeline["z_score"] = (sentiment_timeline["compound"] - mean_val) / std_val
        sentiment_timeline["is_anomaly"] = sentiment_timeline["z_score"].abs() > threshold_std
        return sentiment_timeline[sentiment_timeline["is_anomaly"]]

from config.settings import NewsBot2Config
from src.analysis.classifier import AdvancedNewsClassifier
from src.analysis.sentiment_analyzer import SentimentEvolutionTracker

def test_system_integration():
    cfg = NewsBot2Config()
    classifier = AdvancedNewsClassifier()
    sentiment = SentimentEvolutionTracker()
    text = "Tesla stock rose today after strong battery earnings release."
    s_res = sentiment.analyze_sentiment(text)
    assert s_res["overall_sentiment"] in ["positive", "negative", "neutral"]

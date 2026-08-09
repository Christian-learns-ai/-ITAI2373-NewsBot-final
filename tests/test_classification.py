from src.analysis.classifier import AdvancedNewsClassifier

def test_classifier_init():
    classifier = AdvancedNewsClassifier(model_type="tfidf_lr")
    res = classifier.predict_with_confidence("Sample news text about technology.")
    assert "primary_category" in res

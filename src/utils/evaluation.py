import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class NewsBot2Evaluator:
    """Evaluation suite for NewsBot 2.0 system performance."""

    def __init__(self, newsbot_system):
        self.newsbot = newsbot_system

    def evaluate_classification_performance(self, test_data: list[dict]) -> dict:
        texts = [item["text"] for item in test_data]
        y_true = [item["true_label"] for item in test_data]

        y_pred = []
        confidences = []
        for text in texts:
            pred_info = self.newsbot.classifier.predict_with_confidence(text)
            y_pred.append(pred_info.get("primary_category"))
            confidences.append(pred_info.get("confidence_score", 0.0))

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)

        return {
            "accuracy": round(float(accuracy), 4),
            "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
            "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
            "average_confidence": round(float(np.mean(confidences)), 4),
        }

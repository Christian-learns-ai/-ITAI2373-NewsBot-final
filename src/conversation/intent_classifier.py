import re

class IntentClassifier:
    """Regex pattern matching for conversational user query intents."""

    def __init__(self):
        self.intent_patterns = {
            "search": [r"\bfind\b", r"\bsearch\b", r"\bshow\b", r"\bget\b", r"\bread\b"],
            "summarize": [r"\bsummarize\b", r"\bsummary\b", r"\bkey points\b", r"\boverview\b"],
            "analyze": [r"\banalyze\b", r"\bsentiment\b", r"\btrends?\b", r"\binsights?\b"],
            "compare": [r"\bcompare\b", r"\bdifference\b", r"\bversus\b", r"\bvs\b"],
            "explain": [r"\bexplain\b", r"\bhow are\b", r"\bconnection\b", r"\brelationship\b"],
        }

    def classify_intent(self, user_query: str) -> str:
        query_lower = user_query.lower()
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        return "search"

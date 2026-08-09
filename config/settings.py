import os

class NewsBot2Config:
    """Centralized configuration management for NewsBot 2.0."""

    def __init__(self):
        # API Keys and Endpoints
        self.news_api_key = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
        self.translation_api_url = "https://api-free.deepl.com/v2/translate"

        # Model Parameters
        self.spacy_model = "en_core_web_sm"
        self.huggingface_classifier_model = "facebook/bart-large-mnli"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.summarizer_model = "facebook/bart-large-cnn"
        self.num_topics = 10
        self.lda_passes = 15

        # File Paths and Directories
        self.data_dir = "./data"
        self.models_dir = "./data/models"
        self.cache_dir = "./cache"
        self.output_dir = "./data/results"

        # Processing Limits and Thresholds
        self.batch_size = 32
        self.confidence_threshold = 0.75
        self.max_articles_per_query = 50
        self.supported_languages = ["en", "es", "fr", "de"]

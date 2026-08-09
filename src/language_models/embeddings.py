from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearchEngine:
    """Dense vector semantic embedding index and clustering."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None
        self.documents = []

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        self.documents = documents
        self.doc_embeddings = self.model.encode(documents, convert_to_numpy=True)
        return self.doc_embeddings

    def find_similar_articles(self, query_article: str, top_k: int = 5) -> list[dict]:
        if self.doc_embeddings is None or not self.documents:
            raise ValueError("No indexed documents found. Call encode_documents() first.")

        query_vec = self.model.encode([query_article], convert_to_numpy=True)[0]
        similarities = cosine_similarity([query_vec], self.doc_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {"document": self.documents[idx], "similarity_score": round(float(similarities[idx]), 4)}
            for idx in top_indices
        ]

    def cluster_similar_content(self, articles: list[str] = None, n_clusters: int = 4) -> dict:
        if articles:
            embeddings = self.encode_documents(articles)
        else:
            embeddings = self.doc_embeddings
            articles = self.documents

        if embeddings is None:
            raise ValueError("No embeddings available for clustering.")

        num_clusters = min(n_clusters, len(articles))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        clusters = defaultdict(list)
        for label, doc in zip(cluster_labels, articles):
            clusters[f"Cluster_{label}"].append(doc)

        return dict(clusters)

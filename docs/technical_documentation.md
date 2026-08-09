# NewsBot 2.0 Technical Documentation

## 1. System Architecture
NewsBot 2.0 combines classical ML (TF-IDF + Logistic Regression, VADER, Gensim LDA, scikit-learn NMF) with modern deep learning transformers (BART-CNN, MiniLM sentence embeddings, MarianMT translation pipelines) under a unified orchestration layer.

## 2. Component Pipeline
1. **Data Preprocessing & Validation**: Input cleaning, tokenization, stopword filtering, length validation.
2. **Multilingual Engine**: Language detection (`langdetect`) and MarianMT neural machine translation to English.
3. **Content Analysis Engine**: Category predictions, sentiment polarity, spaCy NER, and NetworkX SVO knowledge graph creation.
4. **Language Models**: BART multi-length abstractive summarization with frequency-scored extractive fallback.
5. **Dense Semantic Index**: MiniLM vector encoding with cosine similarity search and KMeans clustering.
6. **Conversational Dialog**: Pattern-based intent classification with context carry-over across turns.

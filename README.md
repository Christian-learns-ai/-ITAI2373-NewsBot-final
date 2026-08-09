# ITAI 2373: NewsBot 2.0 Final Intelligence Platform

A modular, multi-tier Natural Language Processing (NLP) platform for news monitoring, topic modeling, multilingual translation, semantic search, abstractive summarization, knowledge graphing, and conversational interaction.

## 🏗️ Repository Architecture

```text
ITAI2373-NewsBot-Final/
├── README.md                      # Comprehensive project overview
├── requirements.txt               # All dependencies with versions
├── config/
│   ├── settings.py                # Configuration management
│   └── api_keys_template.txt      # API key template (no real keys!)
├── src/
│   ├── data_processing/           # Preprocessing, feature extraction, validator
│   ├── analysis/                  # Classification, sentiment, NER, topic modeling
│   ├── language_models/           # Summarization, generator, embeddings
│   ├── multilingual/              # Language detection, MarianMT translation
│   ├── conversation/              # Intent classification, slot parsing, response
│   └── utils/                     # Plotting, evaluation, export functions
├── notebooks/                     # Exploratory notebooks (01 to 07)
├── tests/                         # Pytest test suites
├── data/                          # Raw, processed, models, results
├── docs/                          # Technical specs, API, User Guide
└── reports/                       # Executive summary, Technical & Slides

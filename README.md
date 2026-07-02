# Text Intelligence Toolkit — Fake News Detection & Topic Analysis

NLP pipeline for supervised text classification and unsupervised topic discovery, applied to news and web corpora.

## Problem

Distinguishing fake from real news requires extracting robust signal from raw text — handling noise, stopwords, and linguistic variation. Topic modeling adds an unsupervised lens for exploring large corpora without labels.

## Projects

### Fake News Classifier
End-to-end binary classification pipeline: raw text → cleaning → tokenization → feature extraction → supervised model. Trained and evaluated on a labeled news dataset.

### Topic Discovery
Unsupervised extraction of latent topics from Wikipedia articles and news corpora. Identifies thematic clusters without labeled data.

### Multilingual Text Processing
Tokenization and stopword removal for English and Spanish corpora. Word cloud visualization of vocabulary distributions.

## Stack

`Python` `NLTK` `scikit-learn` `WordCloud` `matplotlib`

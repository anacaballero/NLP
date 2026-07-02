# Text Intelligence Toolkit — NLP Pipelines & Text Classification

End-to-end NLP pipelines for supervised text classification, feature engineering, and hyperparameter optimization, applied to news corpora and corporate message routing.

## Projects

### Fake News Detection & Topic Analysis
Binary classification pipeline applied to labeled news datasets — raw text through feature extraction to a trained classifier. Includes unsupervised topic discovery on Wikipedia and news corpora.

### Corporate Message Classification
3-class text classifier (Action / Dialogue / Information) combining TF-IDF lexical features with a custom POS-tag-based syntactic extractor via `FeatureUnion`. Full `GridSearchCV` optimization across pipeline components.

### Multilingual Text Processing
Tokenization and stopword removal for English and Spanish corpora. Word cloud visualization of vocabulary distributions.

## Notebooks

### Text Preprocessing
| Notebook | Description |
|----------|-------------|
| [`text_cleaning_tokenization.ipynb`](text_cleaning_tokenization.ipynb) | URL stripping (BeautifulSoup), Unicode normalization, case folding, tokenization, stopword removal. Reusable sklearn-compatible component. |
| [`bow_tfidf_feature_engineering.ipynb`](bow_tfidf_feature_engineering.ipynb) | Bag-of-Words vs TF-IDF representations. Analyzes vocabulary coverage, sparsity, and discriminative power of each. |

### Pipeline Architecture
| Notebook | Description |
|----------|-------------|
| [`nlp_sklearn_pipeline.ipynb`](nlp_sklearn_pipeline.ipynb) | Production-style sklearn `Pipeline`: CountVectorizer → TF-IDF → RandomForestClassifier. End-to-end fitting and cross-validation without data leakage. |
| [`nlp_feature_union.ipynb`](nlp_feature_union.ipynb) | `FeatureUnion` combining TF-IDF lexical features with a custom syntactic feature. Merges heterogeneous signals into a single feature matrix. |
| [`nlp_custom_transformer.ipynb`](nlp_custom_transformer.ipynb) | `BaseEstimator` + `TransformerMixin` implementation: POS-tag-based `StartingVerbExtractor` that integrates natively with sklearn `Pipeline` and `GridSearchCV`. |

### Model Selection & Optimization
| Notebook | Description |
|----------|-------------|
| [`nlp_gridsearch_pipeline.ipynb`](nlp_gridsearch_pipeline.ipynb) | `GridSearchCV` across CountVectorizer n-gram range, TF-IDF smoothing, and RandomForest depth/estimators simultaneously. Exposes nested pipeline parameters in the search grid. |
| [`nlp_classification_workflow.ipynb`](nlp_classification_workflow.ipynb) | End-to-end classification workflow: preprocessing → feature extraction → training → evaluation on a categorical message dataset. |

## Stack

`Python` `NLTK` `scikit-learn` `WordCloud` `BeautifulSoup` `pandas` `matplotlib`

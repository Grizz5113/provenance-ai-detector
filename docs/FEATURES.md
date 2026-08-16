# Feature Documentation

## Overview

The classifier operates on a fixed 40-dimensional feature vector produced by `EssayFeatureExtractor` (`backend/app/features/extractor.py`), across five groups: language-model, rhythm, lexical, repetition, and punctuation.

## Language-Model Features

From token-level negative log-likelihood (NLL) scores produced by `pythia-160m`.

- **token_count** — tokens scored by the language model
- **mean_nll** — mean NLL across all tokens
- **perplexity** — `exp(mean_nll)`
- **nll_std** — standard deviation of token-level NLL
- **nll_min / nll_max** — min/max token-level NLL
- **nll_median** — median token-level NLL
- **nll_p90** — 90th percentile of token-level NLL

## Rhythm (Sentence-Length) Features

- **sentence_count** — number of sentences
- **mean_sentence_length / median_sentence_length** — average/median sentence length in words
- **sentence_length_std** — standard deviation of sentence length
- **min_sentence_length / max_sentence_length** — shortest/longest sentence
- **sentence_length_cv** — coefficient of variation (std / mean)

## Lexical Features

- **unique_token_count** — distinct word types
- **type_token_ratio** — unique / total tokens (length-sensitive)
- **hapax_ratio** — proportion of tokens appearing exactly once
- **vocabulary_entropy** — Shannon entropy of token frequency distribution
- **repeated_token_ratio** — proportion of tokens repeating an earlier token

## Repetition Features

- **unique_bigrams / repeated_bigrams**
- **bigram_repetition_ratio**
- **unique_trigrams / repeated_trigrams**
- **trigram_repetition_ratio**
- **most_common_bigram_count**
- **most_common_trigram_count**

## Punctuation Features

- **punctuation_count**, **punctuation_density**, **punctuation_types**
- **comma_count**, **period_count**, **semicolon_count**, **colon_count**
- **question_count**, **exclamation_count**
- **parentheses_count**, **quotation_count**, **contraction_count**

## Feature Vector

```text
Text
 |
 +-- Language-model features
 +-- Lexical features
 +-- Sentence features
 +-- Repetition features
 +-- Punctuation features
 |
 v
40 numerical features
 |
 v
Random Forest
```

## Feature Distribution (training data)

```text
token_count:            min 107    mean 316.40   max 905
perplexity:              min 5.87   mean 26.01    max 100.68
sentence_count:          min 1      mean 12.58    max 47
mean_sentence_length:    min 5.83   mean 23.04    max 106
```

## Important Design Consideration

Very short text can produce feature values far outside the training distribution — very high type-token/hapax ratios, different sentence stats, different perplexity, sentence count of one. This is why the API uses context windows rather than scoring every short sentence as an independent full sample.

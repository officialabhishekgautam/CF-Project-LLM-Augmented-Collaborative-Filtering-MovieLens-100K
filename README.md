# LLM-Augmented Collaborative Filtering for Movie Rating Prediction

> **Group 36** | MovieLens 100K Dataset | M.Tech CSE — IIIT Delhi

A three-part rating prediction system that combines traditional Collaborative Filtering with semantic embeddings from a sentence transformer (LLM component) and blends both into a hybrid model evaluated across 5-fold cross-validation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Dataset](#dataset)
- [Three-Part System](#three-part-system)
  - [Part 1 — Collaborative Filtering Baseline](#part-1--collaborative-filtering-baseline)
  - [Part 2 — LLM-Based Prediction](#part-2--llm-based-prediction)
  - [Part 3 — Hybrid Blend](#part-3--hybrid-blend)
- [Running the Code](#running-the-code)
- [Evaluation Protocol](#evaluation-protocol)
- [Results](#results)
- [Design Decisions](#design-decisions)
- [Cold Start Handling](#cold-start-handling)
- [Why Sentence Transformer and Not GPT Directly?](#why-sentence-transformer-and-not-gpt-directly)

---

## Project Overview

This project answers the question:
> **Can a language model's semantic understanding of movies and users improve rating predictions beyond what collaborative filtering alone achieves?**

The approach follows a suggestion from the project review: use movie metadata (title, genres, release year) and user context (demographics, watch history) to construct a semantic profile, then use an LLM's embedding layer to measure content similarity. This is combined with a traditional CF model for final predictions.

**Core idea:**
- CF knows *how much* a user likes something (from historical rating patterns)
- The LLM knows *what is semantically similar* (from content understanding)
- The hybrid gets both signals → better ranking quality + cold-start capability

---

## Repository Structure

```
.
├── data/                        # Auto-downloaded MovieLens 100K (ml-100k/)
│   └── ml-100k/
│       ├── u1.base, u1.test    # Fold 1 train/test
│       ├── u2.base, u2.test    # Fold 2 train/test
│       ├── ...                  # Folds 3–5
│       └── u.item              # Movie metadata (title, genres, year)
│
├── src/
│   ├── data_loader.py          # Load and parse MovieLens files
│   ├── genre_baseline.py       # Genre-average fallback for cold-start movies
│   ├── user_profiler.py        # Build user preference text from watch history
│   ├── prompt_builder.py       # Construct movie + user text for embedding
│   ├── llm_client.py           # Sentence transformer embedding wrapper
│   ├── predictor.py            # Cosine similarity → percentile → rating
│   └── evaluator.py            # NDCG, HR, MRR, MAE, RMSE evaluation
│
├── part1_cf.py                 # Run CF baseline (user-user + item-item)
├── part2_llm.py                # Run LLM-based prediction pipeline
├── part3_hybrid.py             # Run hybrid blend (CF + LLM)
├── run_all_folds.py            # Run all three parts across u1–u5 folds
│
├── results/
│   ├── cf_summary.csv          # CF metrics per fold
│   ├── llm_summary.csv         # LLM metrics per fold
│   ├── blend_summary_per_fold.csv  # Hybrid metrics per fold
│   └── final_comparison.csv    # Side-by-side comparison of all three parts
│
├── tests/
│   ├── test_evaluator.py
│   ├── test_genre_baseline.py
│   └── test_user_profiler.py   # All tests run without an API key
│
├── .env.example                # Copy to .env and add your API key if needed
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/llm-cf-movielens.git
cd llm-cf-movielens
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | ≥2.2.0 | Semantic embedding model |
| `scikit-learn` | ≥1.3.0 | Cosine similarity, metrics |
| `numpy` | ≥1.24.0 | Numerical operations |
| `pandas` | ≥2.0.0 | Data loading and manipulation |
| `scipy` | ≥1.11.0 | NDCG computation |

### 4. Environment variables (optional)

```bash
cp .env.example .env
```

The LLM component uses `all-MiniLM-L6-v2` via the `sentence-transformers` library — **no API key required** for the default setup. If you swap in an external embedding API, add the key to `.env`.

---

## Dataset

The project uses the **MovieLens 100K** dataset — 100,000 ratings (1–5) from 943 users on 1,682 movies.

The dataset is **auto-downloaded** from GroupLens on first run:

```bash
python run_all_folds.py   # triggers download if data/ is missing
```

Alternatively, download manually:
```bash
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip -d data/
```

**Standard 5-fold splits used:** `u1.base / u1.test` through `u5.base / u5.test`

- Training set: ~80,000 ratings
- Test set: ~20,000 ratings
- Binarization threshold: ratings ≥ 4 are treated as "liked" for ranking metrics

---

## Three-Part System

### Part 1 — Collaborative Filtering Baseline

**File:** `part1_cf.py`

A bias-corrected hybrid of user-based and item-based collaborative filtering.

**Prediction formula:**

```
CF_pred(u, i) = α × item_score(u, i) + (1 - α) × user_score(u, i)
```

Where `α = 0.30` (item-weighted, tuned on validation split of u1.base).

- **User-user CF:** Pearson correlation similarity between users, top-K neighbors = 30
- **Item-item CF:** Adjusted cosine similarity between items, top-K = 30
- **Bias correction:** Global mean + user bias + item bias applied to all predictions

```bash
python part1_cf.py
```

---

### Part 2 — LLM-Based Prediction

**File:** `part2_llm.py`

The core contribution. Uses the semantic embedding layer of a sentence transformer to measure content alignment between a user's preference profile and a movie's text description.

**Pipeline:**

```
Movie metadata  →  text string  →  Sentence Transformer  →  384-dim vector
User watch hist →  text profile →  Sentence Transformer  →  384-dim vector
                                                              ↓
                                               cosine_similarity(v_user, v_movie)
                                                              ↓
                                               Percentile normalize → scale to [1–5]
                                                              ↓
                                                       Predicted rating
```

**Movie text construction:**
```python
"Star Wars (1977) — Action, Adventure, Sci-Fi"
```

**User profile construction:**
```python
# From top-rated movies (≥4 stars) in training set
"User enjoys: Crime, Drama, Thriller. 
 Favorite movies include Fargo, Pulp Fiction, Silence of the Lambs."
```

**Embedding model:** `all-MiniLM-L6-v2` (384 dimensions, ~80MB, runs on CPU)

**Why percentile normalization?**
Raw cosine similarities for MovieLens content cluster in a narrow band (≈0.53–0.68). Without normalization, every prediction collapses to ~3.0. Percentile mapping spreads scores across the full 1–5 range.

```python
percentile_rank = percentileofscore(training_similarities, raw_score) / 100
scaled_rating   = 1 + (percentile_rank * 4)
```

```bash
python part2_llm.py
```

---

### Part 3 — Hybrid Blend

**File:** `part3_hybrid.py`

Combines Part 1 and Part 2 scores with a tunable blend parameter β.

```
Hybrid_pred(u, i) = β × CF_pred(u, i) + (1 - β) × LLM_pred(u, i)
```

**Optimal β = 0.70** (tuned on validation fold — 70% CF, 30% LLM).

The 30% LLM weight acts as a semantic re-ranker: it slightly boosts movies that are content-aligned with the user's taste profile, even if those movies haven't been seen by similar users in the training data.

```bash
python part3_hybrid.py
```

---

## Running the Code

### Run a single part on one fold

```bash
python part1_cf.py --fold 1
python part2_llm.py --fold 1
python part3_hybrid.py --fold 1 --beta 0.70
```

### Run all parts across all 5 folds

```bash
python run_all_folds.py
```

Results are saved to `results/`.

### Run unit tests (no API key needed)

```bash
pytest tests/ -v
```

---

## Evaluation Protocol

All parts are evaluated identically across the standard 5-fold splits (u1–u5).

**Metrics computed:**

| Metric | Description |
|--------|-------------|
| `HR@K` | Hit Rate at K — does at least one liked movie appear in top K? |
| `NDCG@K` | Normalized Discounted Cumulative Gain — ranking quality at K |
| `MRR` | Mean Reciprocal Rank — how high does the first relevant item appear? |
| `P@10` | Precision at 10 |
| `R@10` | Recall at 10 |
| `F1@10` | Harmonic mean of P@10 and R@10 |
| `MAE` | Mean Absolute Error on raw rating predictions |
| `RMSE` | Root Mean Squared Error on raw rating predictions |

**Binarization:** A rating ≥ 4.0 is treated as "relevant" for ranking metrics (HR, NDCG, MRR).

**K values evaluated:** 5, 10, 20

---

## Results

### Per-Fold Summary

**Part 2 — LLM (Sentence Transformer):**

| Fold | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|------|------|-------|-------|--------|---------|---------|
| 1 | 0.9978 | 0.9978 | 1.0000 | 0.7907 | 0.7891 | 0.8093 |
| 2 | 0.9891 | 0.9984 | 0.9984 | 0.7501 | 0.7692 | 0.7997 |
| 3 | 0.9941 | 0.9988 | 0.9988 | 0.7569 | 0.7756 | 0.8086 |
| 4 | 0.9854 | 0.9989 | 1.0000 | 0.7548 | 0.7815 | 0.8152 |
| 5 | 0.9852 | 0.9977 | 1.0000 | 0.7495 | 0.7761 | 0.8130 |
| **AVG** | **0.9903** | **0.9983** | **0.9995** | **0.7604** | **0.7783** | **0.8092** |

**Part 3 — Hybrid (β = 0.70):**

| Fold | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|------|------|-------|-------|--------|---------|---------|-----|
| 1 | 0.9934 | 1.0000 | 1.0000 | 0.8694 | 0.8648 | 0.8740 | 0.9565 |
| 2 | 0.9938 | 0.9984 | 1.0000 | 0.8327 | 0.8403 | 0.8598 | 0.9161 |
| 3 | 0.9953 | 0.9988 | 1.0000 | 0.8352 | 0.8404 | 0.8626 | 0.9131 |
| 4 | 0.9944 | 0.9989 | 1.0000 | 0.8402 | 0.8530 | 0.8730 | 0.9253 |
| 5 | 0.9989 | 1.0000 | 1.0000 | 0.8357 | 0.8486 | 0.8735 | 0.9174 |
| **AVG** | **0.9952** | **0.9992** | **1.0000** | **0.8426** | **0.8494** | **0.8686** | **0.9257** |

### Final Comparison (Averaged Across All Folds)

| Metric | CF (β=1.0) | LLM (β=0.0) | Hybrid (β=0.70) | Δ vs CF |
|--------|-----------|------------|----------------|---------|
| HR@5 | 0.9947 | 0.9926 | **1.0000** | ↑ 0.0053 |
| HR@10 | 0.9979 | 1.0000 | **1.0000** | ↑ 0.0021 |
| HR@20 | 0.9989 | 1.0000 | **1.0000** | ↑ 0.0011 |
| NDCG@5 | 0.8142 | 0.7676 | **0.8619** | ↑ 0.0477 |
| NDCG@10 | 0.7990 | 0.7385 | **0.8318** | ↑ 0.0328 |
| NDCG@20 | 0.8107 | 0.7397 | **0.8267** | ↑ 0.0160 |
| MRR | 0.9061 | 0.8794 | **0.9480** | ↑ 0.0419 |
| MAE | **0.7462** | 1.1420 | 0.7753 | ↑ 0.0291 (trade-off) |
| RMSE | **0.9296** | 1.4402 | 0.9575 | ↑ 0.0279 (trade-off) |

**Key takeaway:** The hybrid improves NDCG@5 by **+5.86% relative** and MRR by **+4.62% relative** over CF alone, at the cost of a small MAE increase (0.746 → 0.775) — an acceptable trade-off for significantly better ranking quality.

---

## Design Decisions

### Why α = 0.30 for CF blend?

Tuned on a validation split (10% of u1.base). Item-based CF outperforms user-based CF on this dataset due to the higher density of item co-ratings. Higher item weight (α = 0.30 item-weighted) consistently reduced MAE.

### Why β = 0.70 for the hybrid?

Grid search over β ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} on a validation split. β = 0.70 maximized NDCG@5 — CF contributes strong personalization from rating history while the 30% LLM weight provides semantic re-ranking.

### Why `all-MiniLM-L6-v2`?

- Purpose-built for semantic similarity (contrastive training, not next-token prediction)
- Fast: encodes all 1,682 MovieLens movies in under 5 seconds on CPU
- Small: ~80MB download, no GPU required
- 384-dim vectors are sufficient for the content length in MovieLens metadata

---

## Cold Start Handling

### New movie (no ratings in training):

A genre baseline is computed from training data:

```python
genre_baseline(movie) = mean([avg_rating(genre) for genre in movie.genres])
```

This baseline is used as the initial rating estimate. Cosine similarity with the user profile is then applied as a personalization adjustment.

### New user (no rating history):

A demographic fallback is used:

```python
demographic_profile = avg_ratings_by(age_group, gender, occupation)
```

A user profile text is built from the genre preferences of demographically similar users. The LLM pipeline then proceeds normally with this proxy profile.

> **CF cannot handle either cold-start case** — it requires at least some overlap in the user-item matrix. The LLM component degrades gracefully to metadata-based estimation.

---

## Why Sentence Transformer and Not GPT Directly?

This is the most important architectural question.

| | Chat LLM (GPT, Llama) | Sentence Transformer |
|---|---|---|
| **Trained for** | Next token prediction | Semantic similarity (contrastive loss) |
| **Vectors calibrated for cosine sim?** | ❌ No | ✅ Yes |
| **Speed for 1,682 movies** | Very slow (full forward pass) | < 5 seconds on CPU |
| **Memory** | 4–70 GB | ~500 MB |
| **Cost** | API quota usage | Free, local |
| **Cold start** | Feasible but expensive | Fast and free |

A general-purpose chat LLM's hidden states *can* represent meaning, but this is a side effect — not the design goal. Its internal representations are optimized to predict the next token, not to cluster semantically similar items in vector space.

Sentence transformers are explicitly trained with contrastive objectives:
```
"Die Hard" and "The Rock"   → embeddings should be CLOSE  ✅
"Die Hard" and "Toy Story"  → embeddings should be FAR    ✅
```

Chat LLMs have no such training signal. Using a sentence transformer here is using the right tool for the job.

> **Note:** LLM embedding API endpoints (e.g., `text-embedding-004`) are purpose-built for similarity and would also work — they are just remotely hosted equivalents of what `all-MiniLM-L6-v2` does locally.

---

## License

MIT License — see `LICENSE` for details.

---

*Group 36 | M.Tech CSE | IIIT Delhi*


# =============================================================================
# PART 1 — Hybrid CF with Bias Correction
# Metrics: Hit Rate@K and NDCG@K (binarized ratings)
# Saves: cf_predictions_fold{i}.csv + cf_scores_fold{i}.npy for blending
# =============================================================================

import numpy as np
import pandas as pd
import pickle, os, time
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Tuple, List

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH   = r"C:\Users\Douglas\OneDrive\Desktop\CF_Project\ml-100k"
OUTPUT_PATH = r"C:\Users\Douglas\OneDrive\Desktop\CF_Project\part1_output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

BINARY_THRESHOLD = 4.0        # ratings >= 4 → relevant (1), else (0)
K_VALUES         = [5, 10, 20]
FOLDS            = [1, 2, 3, 4, 5]

# ── HYPERPARAMS (best from assignments) ───────────────────────────────────────
@dataclass
class CFConfig:
    lambda_bias:     float = 2.5
    bias_iters:      int   = 40
    k_user:          int   = 150
    k_item:          int   = 40
    alpha:           float = 0.30     # alpha*item + (1-alpha)*user
    shrink_user:     float = 750.0
    shrink_item:     float = 750.0
    min_common_user: int   = 2
    min_common_item: int   = 2
    gamma_user:      float = 1.2
    gamma_item:      float = 1.4
    pos_only_user:   bool  = True
    pos_only_item:   bool  = True
    cand_user:       int   = 350
    cand_item:       int   = 450

CFG = CFConfig()

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_ratings(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None,
                       names=["user", "item", "rating", "ts"],
                       usecols=["user", "item", "rating"])

def load_item_meta() -> pd.DataFrame:
    """Load u.item → movie id, title, year, genres"""
    genres = ["unknown","Action","Adventure","Animation","Childrens","Comedy",
              "Crime","Documentary","Drama","Fantasy","FilmNoir","Horror",
              "Musical","Mystery","Romance","SciFi","Thriller","War","Western"]
    cols = ["item","title","release","video","url"] + genres
    df = pd.read_csv(os.path.join(DATA_PATH, "u.item"), sep="|",
                     header=None, names=cols, encoding="latin-1",
                     usecols=["item","title","release"] + genres)
    df["genre_str"] = df[genres].apply(
        lambda r: "|".join(g for g, v in zip(genres, r) if v == 1), axis=1)
    df["year"] = df["release"].str.extract(r"\((\d{4})\)").fillna("")
    return df[["item","title","year","genre_str"]]

def load_user_meta() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_PATH, "u.user"), sep="|", header=None,
                       names=["user","age","gender","occupation","zip"],
                       usecols=["user","age","gender","occupation"])

# ── BIAS ESTIMATION (ALS / iterative) ─────────────────────────────────────────
def compute_biases(train: pd.DataFrame, cfg: CFConfig):
    mu = train["rating"].mean()
    users = train["user"].unique()
    items = train["item"].unique()
    b_u = dict.fromkeys(users, 0.0)
    b_i = dict.fromkeys(items, 0.0)

    u_ratings = train.groupby("user")[["item","rating"]].apply(
        lambda x: list(zip(x["item"], x["rating"]))).to_dict()
    i_ratings = train.groupby("item")[["user","rating"]].apply(
        lambda x: list(zip(x["user"], x["rating"]))).to_dict()

    for _ in range(cfg.bias_iters):
        for u, rlist in u_ratings.items():
            num = sum(r - mu - b_i.get(i, 0.0) for i, r in rlist)
            b_u[u] = num / (cfg.lambda_bias + len(rlist))
        for i, rlist in i_ratings.items():
            num = sum(r - mu - b_u.get(u, 0.0) for u, r in rlist)
            b_i[i] = num / (cfg.lambda_bias + len(rlist))
    return mu, b_u, b_i

# ── RESIDUAL MATRIX ───────────────────────────────────────────────────────────
def compute_residuals(train: pd.DataFrame, mu, b_u, b_i):
    df = train.copy()
    df["res"] = (df["rating"]
                 - mu
                 - df["user"].map(b_u).fillna(0)
                 - df["item"].map(b_i).fillna(0))
    # user→{item: residual}  and  item→{user: residual}
    u2i = df.groupby("user").apply(
        lambda x: dict(zip(x["item"], x["res"]))).to_dict()
    i2u = df.groupby("item").apply(
        lambda x: dict(zip(x["user"], x["res"]))).to_dict()
    return u2i, i2u

# ── USER-USER SIMILARITY ──────────────────────────────────────────────────────
def pearson(r1: dict, r2: dict, shrink: float) -> float:
    common = set(r1) & set(r2)
    n = len(common)
    if n == 0:
        return 0.0
    v1 = np.array([r1[k] for k in common])
    v2 = np.array([r2[k] for k in common])
    num = float(np.dot(v1, v2))
    den = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if den == 0:
        return 0.0
    sim = num / den
    return sim * n / (n + shrink)

def user_predict(u: int, i: int, u2i: dict, mu, b_u, b_i, cfg: CFConfig) -> float:
    if u not in u2i:
        return mu + b_u.get(u, 0) + b_i.get(i, 0)
    target_res = u2i[u]
    # candidate pool: users who rated item i
    from_item_raters = set()  # filled lazily below
    baseline = mu + b_u.get(u, 0) + b_i.get(i, 0)

    sims = []
    for v, v_res in u2i.items():
        if v == u:
            continue
        if i not in v_res:
            continue
        common = set(target_res) & set(v_res)
        if len(common) < cfg.min_common_user:
            continue
        sim = pearson(target_res, v_res, cfg.shrink_user)
        if cfg.pos_only_user and sim <= 0:
            continue
        # significance weighting
        w = sim * (min(len(common), 50) / 50) ** cfg.gamma_user
        sims.append((w, v_res[i]))
    if not sims:
        return baseline
    sims.sort(key=lambda x: -abs(x[0]))
    sims = sims[:cfg.k_user]
    num = sum(w * r for w, r in sims)
    den = sum(abs(w) for w, _ in sims) + 1e-9
    return baseline + num / den

# ── ITEM-ITEM SIMILARITY ──────────────────────────────────────────────────────
def adj_cosine(i1: int, i2: int, i2u: dict, u_mean: dict, shrink: float) -> float:
    r1, r2 = i2u.get(i1, {}), i2u.get(i2, {})
    common = set(r1) & set(r2)
    n = len(common)
    if n == 0:
        return 0.0
    v1 = np.array([r1[u] - u_mean.get(u, 0) for u in common])
    v2 = np.array([r2[u] - u_mean.get(u, 0) for u in common])
    num = float(np.dot(v1, v2))
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    if den == 0:
        return 0.0
    sim = num / den
    return sim * n / (n + shrink)

def item_predict(u: int, i: int, i2u: dict, u2i: dict, u_mean: dict,
                 mu, b_u, b_i, cfg: CFConfig) -> float:
    baseline = mu + b_u.get(u, 0) + b_i.get(i, 0)
    if u not in u2i:
        return baseline
    rated_by_u = u2i[u]   # items rated by target user
    sims = []
    for j, r_uj in rated_by_u.items():
        if j == i:
            continue
        common = set(i2u.get(i, {})) & set(i2u.get(j, {}))
        if len(common) < cfg.min_common_item:
            continue
        sim = adj_cosine(i, j, i2u, u_mean, cfg.shrink_item)
        if cfg.pos_only_item and sim <= 0:
            continue
        w = sim * (min(len(common), 50) / 50) ** cfg.gamma_item
        sims.append((w, r_uj))
    if not sims:
        return baseline
    sims.sort(key=lambda x: -abs(x[0]))
    sims = sims[:cfg.k_item]
    num = sum(w * r for w, r in sims)
    den = sum(abs(w) for w, _ in sims) + 1e-9
    return baseline + num / den

# ── HYBRID PREDICT ────────────────────────────────────────────────────────────
def hybrid_predict(u: int, i: int, u2i, i2u, u_mean, mu, b_u, b_i,
                   cfg: CFConfig) -> Tuple[float, float, float]:
    u_score = user_predict(u, i, u2i, mu, b_u, b_i, cfg)
    i_score = item_predict(u, i, i2u, u2i, u_mean, mu, b_u, b_i, cfg)
    hybrid  = cfg.alpha * i_score + (1 - cfg.alpha) * u_score
    return float(np.clip(hybrid, 1.0, 5.0)), float(u_score), float(i_score)

# ── RANKING METRICS ───────────────────────────────────────────────────────────
def dcg_at_k(rel: List[int], k: int) -> float:
    rel = rel[:k]
    return sum(r / np.log2(idx + 2) for idx, r in enumerate(rel))

def ndcg_at_k(rel: List[int], k: int) -> float:
    ideal = sorted(rel, reverse=True)
    idcg  = dcg_at_k(ideal, k)
    return dcg_at_k(rel, k) / idcg if idcg > 0 else 0.0

def hit_rate_at_k(rel: List[int], k: int) -> float:
    return float(any(r == 1 for r in rel[:k]))

def compute_ranking_metrics(user_preds: Dict[int, List[Tuple[float, int]]],
                             k_values: List[int]) -> Dict:
    """
    user_preds: {user: [(pred_score, actual_binary), ...]}
    Returns dict of metric→value
    """
    results = {f"HR@{k}": [] for k in k_values}
    results.update({f"NDCG@{k}": [] for k in k_values})

    for u, pairs in user_preds.items():
        if not any(ab == 1 for _, ab in pairs):
            continue   # skip users with no positive in test set
        # sort by predicted score descending
        pairs_sorted = sorted(pairs, key=lambda x: -x[0])
        rel = [ab for _, ab in pairs_sorted]
        for k in k_values:
            results[f"HR@{k}"].append(hit_rate_at_k(rel, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(rel, k))

    return {m: float(np.mean(v)) if v else 0.0 for m, v in results.items()}

# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def run_fold(fold: int, item_meta: pd.DataFrame, user_meta: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold}")
    print(f"{'='*60}")
    t0 = time.time()

    train = load_ratings(os.path.join(DATA_PATH, f"u{fold}.base"))
    test  = load_ratings(os.path.join(DATA_PATH, f"u{fold}.test"))

    print(f"  Train: {len(train):,} ratings | Test: {len(test):,} ratings")

    # 1. Biases
    print("  [1/4] Computing biases...")
    mu, b_u, b_i = compute_biases(train, CFG)

    # 2. Residual lookup structures
    print("  [2/4] Building residual structures...")
    u2i, i2u = compute_residuals(train, mu, b_u, b_i)
    u_mean = train.groupby("user")["rating"].mean().to_dict()

    # 3. Predict all test pairs
    print("  [3/4] Predicting test pairs...")
    records = []
    user_preds: Dict[int, List] = defaultdict(list)

    for _, row in test.iterrows():
        u, i, r_true = int(row["user"]), int(row["item"]), float(row["rating"])
        pred, u_pred, i_pred = hybrid_predict(u, i, u2i, i2u, u_mean, mu, b_u, b_i, CFG)
        r_bin  = int(r_true >= BINARY_THRESHOLD)
        p_bin  = int(pred   >= BINARY_THRESHOLD)
        records.append({
            "fold": fold, "user": u, "item": i,
            "cf_score": pred,
            "cf_user_score": u_pred,
            "cf_item_score": i_pred,
            "cf_binary_pred": p_bin,
            "actual_rating": r_true,
            "actual_binary": r_bin,
            "baseline": mu + b_u.get(u, 0) + b_i.get(i, 0),
        })
        user_preds[u].append((pred, r_bin))

    # 4. Metrics
    print("  [4/4] Computing Hit Rate & NDCG...")
    metrics = compute_ranking_metrics(user_preds, K_VALUES)

    df_preds = pd.DataFrame(records)
    elapsed  = time.time() - t0
    print(f"\n  ── Results (Fold {fold}) ──")
    for m, v in sorted(metrics.items()):
        print(f"    {m:10s} = {v:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    return df_preds, metrics

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  PART 1 — Hybrid CF  |  Hit Rate & NDCG Evaluation      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nData path : {DATA_PATH}")
    print(f"Output    : {OUTPUT_PATH}")
    print(f"\nConfig    : α={CFG.alpha} | k_user={CFG.k_user} | k_item={CFG.k_item}")
    print(f"            λ_bias={CFG.lambda_bias} | shrink_u={CFG.shrink_user} | shrink_i={CFG.shrink_item}")
    print(f"            binary_threshold={BINARY_THRESHOLD} | K={K_VALUES}\n")

    item_meta = load_item_meta()
    user_meta = load_user_meta()

    all_preds   = []
    all_metrics = defaultdict(list)

    for fold in FOLDS:
        df_fold, metrics = run_fold(fold, item_meta, user_meta)
        all_preds.append(df_fold)
        for m, v in metrics.items():
            all_metrics[m].append(v)

        # Save per-fold predictions (used in blending)
        out_csv = os.path.join(OUTPUT_PATH, f"cf_predictions_fold{fold}.csv")
        df_fold.to_csv(out_csv, index=False)

    # ── Combine & Save everything ────────────────────────────────────────────
    df_all = pd.concat(all_preds, ignore_index=True)
    df_all.to_csv(os.path.join(OUTPUT_PATH, "cf_predictions_all.csv"), index=False)

    # Save as pickle for blending script
    blend_data = {}
    for fold in FOLDS:
        df_f = df_all[df_all["fold"] == fold]
        blend_data[fold] = {
            (row["user"], row["item"]): {
                "cf_score":      row["cf_score"],
                "cf_user_score": row["cf_user_score"],
                "cf_item_score": row["cf_item_score"],
                "actual_rating": row["actual_rating"],
                "actual_binary": row["actual_binary"],
                "baseline":      row["baseline"],
            }
            for _, row in df_f.iterrows()
        }
    with open(os.path.join(OUTPUT_PATH, "cf_blend_data.pkl"), "wb") as f:
        pickle.dump(blend_data, f)

    # ── Summary Table ────────────────────────────────────────────────────────
    print("\n\n╔══════════════════════════════════════════════════════════╗")
    print("║                  FINAL RESULTS SUMMARY                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    rows = []
    for fold in FOLDS:
        df_f    = df_all[df_all["fold"] == fold]
        up      = defaultdict(list)
        for _, r in df_f.iterrows():
            up[r["user"]].append((r["cf_score"], r["actual_binary"]))
        m       = compute_ranking_metrics(up, K_VALUES)
        row     = {"Fold": fold}
        row.update(m)
        rows.append(row)

    avg_row = {"Fold": "AVG"}
    for m in all_metrics:
        avg_row[m] = round(float(np.mean(all_metrics[m])), 4)
    rows.append(avg_row)

    df_summary = pd.DataFrame(rows)
    metric_cols = sorted([c for c in df_summary.columns if c != "Fold"])
    df_summary  = df_summary[["Fold"] + metric_cols]

    # round
    for c in metric_cols:
        df_summary[c] = df_summary[c].apply(
            lambda x: round(float(x), 4) if isinstance(x, (float, np.floating)) else x)

    print(df_summary.to_string(index=False))
    df_summary.to_csv(os.path.join(OUTPUT_PATH, "cf_summary.csv"), index=False)

    # ── 5 Sample Predictions ─────────────────────────────────────────────────
    print("\n\n══ 5 Sample Predictions (Fold 1) ══")
    sample = df_all[df_all["fold"] == 1].head(5)
    im = item_meta.set_index("item")
    um = user_meta.set_index("user")

    for _, r in sample.iterrows():
        u, i = int(r["user"]), int(r["item"])
        title    = im.loc[i, "title"] if i in im.index else f"item_{i}"
        genre    = im.loc[i, "genre_str"] if i in im.index else "?"
        occ      = um.loc[u, "occupation"] if u in um.index else "?"
        age      = um.loc[u, "age"] if u in um.index else "?"
        gender   = um.loc[u, "gender"] if u in um.index else "?"
        print(f"\n  User {u:>4}  [{age}y {gender} {occ}]")
        print(f"  Movie : {title}  [{genre}]")
        print(f"  CF Score     : {r['cf_score']:.3f}  →  binary_pred={r['cf_binary_pred']}")
        print(f"  Actual Rating: {r['actual_rating']:.1f}  →  binary_true={r['actual_binary']}")
        print(f"  Baseline     : {r['baseline']:.3f} ...")
        verdict = "✅ CORRECT" if r["cf_binary_pred"] == r["actual_binary"] else "❌ WRONG"
        print(f"  Binary match : {verdict}")

    # ── Files Saved ──────────────────────────────────────────────────────────
    print("\n\n══ Files Saved ══")
    for fn in sorted(os.listdir(OUTPUT_PATH)):
        fp = os.path.join(OUTPUT_PATH, fn)
        sz = os.path.getsize(fp) / 1024
        print(f"  {fn:45s}  {sz:7.1f} KB")

    print("\n✅ Part 1 complete. Ready for Part 2 (LLM) and Part 3 (Blending).\n")

if __name__ == "__main__":
    main()

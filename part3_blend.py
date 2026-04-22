
# =============================================================================
# PART 3 — Hybrid Blending: CF + LLM Semantic Scores
# Pipeline:
#   1. Load cf_blend_data.pkl  (Part 1 output)
#   2. Load llm_blend_data.pkl (Part 2 output)
#   3. Grid-search beta in [0.0 → 1.0] to find optimal blend weight
#   4. Evaluate required metrics : HR@K, NDCG@K
#      Extra metrics added       : MAE, RMSE, Precision@K, Recall@K, F1@K, MRR
#   5. Final comparison table    : Part1 CF vs Part2 LLM vs Part3 Blend
#   6. 5 Sample predictions from best blend
#   7. Save all results to part3_output/
# =============================================================================

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH     = r"C:\Users\Douglas\OneDrive\Desktop\CF_Project\ml-100k"
PART1_OUTPUT  = r"C:\Users\Douglas\OneDrive\Desktop\CF_Project\part1_output"
PART2_OUTPUT  = r"C:\Users\Douglas\OneDrive\Desktop\CF_Project\part2_output"
OUTPUT_PATH   = r"C:\Users\Douglas\OneDrive\Desktop\CF_Project\part3_output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

BINARY_THRESHOLD = 4.0
K_VALUES         = [5, 10, 20]
FOLDS            = [1, 2, 3, 4, 5]
BETA_GRID        = [round(b * 0.05, 2) for b in range(21)]  # 0.00 to 1.00 step 0.05

# ── LOAD BLEND DATA ───────────────────────────────────────────────────────────
def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# ── METRICS ───────────────────────────────────────────────────────────────────
def dcg_at_k(rel: List[int], k: int) -> float:
    return sum(r / np.log2(idx + 2) for idx, r in enumerate(rel[:k]))

def ndcg_at_k(rel: List[int], k: int) -> float:
    ideal = sorted(rel, reverse=True)
    idcg  = dcg_at_k(ideal, k)
    return dcg_at_k(rel, k) / idcg if idcg > 0 else 0.0

def hit_rate_at_k(rel: List[int], k: int) -> float:
    return float(any(r == 1 for r in rel[:k]))

def precision_at_k(rel: List[int], k: int) -> float:
    return float(sum(rel[:k]) / k)

def recall_at_k(rel: List[int], k: int, n_relevant: int) -> float:
    return float(sum(rel[:k]) / n_relevant) if n_relevant > 0 else 0.0

def f1_at_k(prec: float, rec: float) -> float:
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

def reciprocal_rank(rel: List[int]) -> float:
    for idx, r in enumerate(rel):
        if r == 1:
            return 1.0 / (idx + 1)
    return 0.0

def compute_all_metrics(user_preds: Dict[int, List[Tuple[float, float, int]]],
                        k_values: List[int]) -> Dict:
    """
    user_preds: {user_id: [(blend_score, actual_rating, actual_binary), ...]}
    Returns dict of all metric names → mean value
    """
    results = {f"HR@{k}":        [] for k in k_values}
    results.update({f"NDCG@{k}": [] for k in k_values})
    results.update({f"P@{k}":    [] for k in k_values})
    results.update({f"R@{k}":    [] for k in k_values})
    results.update({f"F1@{k}":   [] for k in k_values})
    results["MRR"]  = []
    results["MAE"]  = []
    results["RMSE"] = []

    for u, triples in user_preds.items():
        if not triples:
            continue

        # MAE / RMSE over all predictions for this user
        scores   = [t[0] for t in triples]
        actuals  = [t[1] for t in triples]
        abs_errs = [abs(s - a) for s, a in zip(scores, actuals)]
        sq_errs  = [(s - a) ** 2 for s, a in zip(scores, actuals)]
        results["MAE"].append(float(np.mean(abs_errs)))
        results["RMSE"].append(float(np.sqrt(np.mean(sq_errs))))

        # Ranking metrics — skip users with no relevant items
        n_relevant = sum(t[2] for t in triples)
        if n_relevant == 0:
            continue

        sorted_triples = sorted(triples, key=lambda x: -x[0])
        rel = [t[2] for t in sorted_triples]

        results["MRR"].append(reciprocal_rank(rel))
        for k in k_values:
            results[f"HR@{k}"].append(hit_rate_at_k(rel, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(rel, k))
            p = precision_at_k(rel, k)
            r = recall_at_k(rel, k, n_relevant)
            results[f"P@{k}"].append(p)
            results[f"R@{k}"].append(r)
            results[f"F1@{k}"].append(f1_at_k(p, r))

    return {m: float(np.mean(v)) if v else 0.0 for m, v in results.items()}

# ── BLEND & EVALUATE ──────────────────────────────────────────────────────────
def evaluate_beta(cf_data: dict, llm_data: dict, beta: float) -> Dict:
    """
    beta=1.0 → pure CF, beta=0.0 → pure LLM
    final_score = beta * cf_score + (1-beta) * llm_score
    """
    all_user_preds = defaultdict(list)

    for fold in FOLDS:
        cf_fold  = cf_data.get(fold, {})
        llm_fold = llm_data.get(fold, {})
        common   = set(cf_fold.keys()) & set(llm_fold.keys())

        for key in common:
            u, i       = key
            cf_score   = float(cf_fold[key]["cf_score"])
            llm_score  = float(llm_fold[key]["llm_score"])
            act_rating = float(cf_fold[key]["actual_rating"])
            act_binary = int(cf_fold[key]["actual_binary"])

            blend = beta * cf_score + (1.0 - beta) * llm_score
            all_user_preds[u].append((blend, act_rating, act_binary))

    return compute_all_metrics(all_user_preds, K_VALUES)

# ── PER-FOLD EVALUATE (for final best beta) ───────────────────────────────────
def evaluate_beta_per_fold(cf_data: dict, llm_data: dict, beta: float):
    fold_metrics = {}
    for fold in FOLDS:
        cf_fold  = cf_data.get(fold, {})
        llm_fold = llm_data.get(fold, {})
        common   = set(cf_fold.keys()) & set(llm_fold.keys())
        up       = defaultdict(list)

        for key in common:
            u, i       = key
            cf_score   = float(cf_fold[key]["cf_score"])
            llm_score  = float(llm_fold[key]["llm_score"])
            act_rating = float(cf_fold[key]["actual_rating"])
            act_binary = int(cf_fold[key]["actual_binary"])
            blend      = beta * cf_score + (1.0 - beta) * llm_score
            up[u].append((blend, act_rating, act_binary))

        fold_metrics[fold] = compute_all_metrics(up, K_VALUES)
    return fold_metrics

# ── SAMPLE PREDICTIONS ────────────────────────────────────────────────────────
def show_samples(cf_data: dict, llm_data: dict, beta: float,
                 item_meta: pd.DataFrame, user_meta: pd.DataFrame, n: int = 5):
    im  = item_meta.set_index("item")
    um  = user_meta.set_index("user")
    cf1 = cf_data.get(1, {})
    ll1 = llm_data.get(1, {})
    common = list(set(cf1.keys()) & set(ll1.keys()))[:n]

    print(f"\n══ {n} Sample Predictions (Fold 1, β={beta:.2f}) ══\n")
    for (u, i) in common:
        cf_s   = float(cf1[(u,i)]["cf_score"])
        llm_s  = float(ll1[(u,i)]["llm_score"])
        blend  = beta * cf_s + (1.0 - beta) * llm_s
        r_true = float(cf1[(u,i)]["actual_rating"])
        r_bin  = int(cf1[(u,i)]["actual_binary"])
        p_bin  = int(blend >= BINARY_THRESHOLD)

        title  = im.loc[i, "title"]      if i in im.index else f"item_{i}"
        genre  = im.loc[i, "genre_str"]  if i in im.index else "?"
        age    = um.loc[u, "age"]        if u in um.index else "?"
        gender = um.loc[u, "gender"]     if u in um.index else "?"
        occ    = um.loc[u, "occupation"] if u in um.index else "?"

        verdict = "✅ CORRECT" if p_bin == r_bin else "❌ WRONG"
        print(f"  User {u:>4}  [{age}y {gender} {occ}]")
        print(f"  Movie : {title}  [{genre}]")
        print(f"  CF Score   : {cf_s:.3f}")
        print(f"  LLM Score  : {llm_s:.3f}")
        print(f"  Blend(β={beta:.2f}): {blend:.3f}  →  binary_pred={p_bin}")
        print(f"  Actual     : {r_true:.1f}            →  binary_true={r_bin}")
        print(f"  Match      : {verdict}\n")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_item_meta() -> pd.DataFrame:
    genres = ["unknown","Action","Adventure","Animation","Childrens","Comedy",
              "Crime","Documentary","Drama","Fantasy","FilmNoir","Horror",
              "Musical","Mystery","Romance","SciFi","Thriller","War","Western"]
    cols   = ["item","title","release","video","url"] + genres
    df     = pd.read_csv(os.path.join(DATA_PATH, "u.item"), sep="|",
                         header=None, names=cols, encoding="latin-1",
                         usecols=["item","title","release"] + genres)
    df["genre_str"] = df[genres].apply(
        lambda r: ", ".join(g for g, v in zip(genres, r) if v == 1), axis=1)
    return df[["item","title","genre_str"]]

def load_user_meta() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_PATH, "u.user"), sep="|",
                       header=None, names=["user","age","gender","occupation","zip"],
                       usecols=["user","age","gender","occupation"])

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    import time
    t_start = time.time()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  PART 3 — Hybrid Blending: CF + LLM                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Part 1 output : {PART1_OUTPUT}")
    print(f"  Part 2 output : {PART2_OUTPUT}")
    print(f"  Part 3 output : {OUTPUT_PATH}")
    print(f"  Beta grid     : {BETA_GRID}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("[1/5] Loading blend data ...")
    cf_data  = load_pkl(os.path.join(PART1_OUTPUT, "cf_blend_data.pkl"))
    llm_data = load_pkl(os.path.join(PART2_OUTPUT, "llm_blend_data.pkl"))

    cf_count  = sum(len(v) for v in cf_data.values())
    llm_count = sum(len(v) for v in llm_data.values())
    print(f"  CF  entries : {cf_count:,}")
    print(f"  LLM entries : {llm_count:,}")

    item_meta = load_item_meta()
    user_meta = load_user_meta()

    # ── Beta grid search ───────────────────────────────────────────────────────
    print("\n[2/5] Grid-searching beta (CF weight) ...")
    print(f"  {'Beta':>6}  {'HR@5':>7} {'HR@10':>7} {'HR@20':>7}  "
          f"{'NDCG@5':>8} {'NDCG@10':>8} {'NDCG@20':>8}  "
          f"{'P@10':>7} {'R@10':>7} {'F1@10':>7}  "
          f"{'MRR':>7} {'MAE':>7} {'RMSE':>7}")
    print("  " + "─" * 112)

    beta_results = {}
    for beta in BETA_GRID:
        m = evaluate_beta(cf_data, llm_data, beta)
        beta_results[beta] = m
        print(f"  {beta:>6.2f}  "
              f"{m['HR@5']:>7.4f} {m['HR@10']:>7.4f} {m['HR@20']:>7.4f}  "
              f"{m['NDCG@5']:>8.4f} {m['NDCG@10']:>8.4f} {m['NDCG@20']:>8.4f}  "
              f"{m['P@10']:>7.4f} {m['R@10']:>7.4f} {m['F1@10']:>7.4f}  "
              f"{m['MRR']:>7.4f} {m['MAE']:>7.4f} {m['RMSE']:>7.4f}")

    # ── Find best beta ─────────────────────────────────────────────────────────
    best_beta = max(beta_results, key=lambda b: beta_results[b]["NDCG@10"])
    best_m    = beta_results[best_beta]

    print(f"\n  ★ Best β = {best_beta:.2f}  (maximises NDCG@10 = {best_m['NDCG@10']:.4f})")

    # ── Per-fold results at best beta ──────────────────────────────────────────
    print("\n[3/5] Per-fold results at best β ...")
    fold_metrics = evaluate_beta_per_fold(cf_data, llm_data, best_beta)

    print(f"\n  Fold  HR@5   HR@10  HR@20  NDCG@5  NDCG@10  NDCG@20  "
          f"P@10   R@10   F1@10   MRR    MAE    RMSE")
    print("  " + "─" * 98)
    fold_rows = []
    for fold in FOLDS:
        m   = fold_metrics[fold]
        row = {"Fold": fold,
               "HR@5": round(m["HR@5"],4), "HR@10": round(m["HR@10"],4),
               "HR@20": round(m["HR@20"],4),
               "NDCG@5": round(m["NDCG@5"],4), "NDCG@10": round(m["NDCG@10"],4),
               "NDCG@20": round(m["NDCG@20"],4),
               "P@10": round(m["P@10"],4), "R@10": round(m["R@10"],4),
               "F1@10": round(m["F1@10"],4), "MRR": round(m["MRR"],4),
               "MAE": round(m["MAE"],4), "RMSE": round(m["RMSE"],4)}
        fold_rows.append(row)
        print(f"  {fold:>4}  {m['HR@5']:.4f} {m['HR@10']:.4f} {m['HR@20']:.4f}  "
              f"{m['NDCG@5']:.4f}  {m['NDCG@10']:.4f}   {m['NDCG@20']:.4f}  "
              f"{m['P@10']:.4f} {m['R@10']:.4f} {m['F1@10']:.4f}  "
              f"{m['MRR']:.4f} {m['MAE']:.4f} {m['RMSE']:.4f}")

    avgs = {k: round(float(np.mean([r[k] for r in fold_rows])), 4)
            for k in fold_rows[0] if k != "Fold"}
    avg_row = {"Fold": "AVG", **avgs}
    fold_rows.append(avg_row)
    print(f"  {'AVG':>4}  {avgs['HR@5']:.4f} {avgs['HR@10']:.4f} {avgs['HR@20']:.4f}  "
          f"{avgs['NDCG@5']:.4f}  {avgs['NDCG@10']:.4f}   {avgs['NDCG@20']:.4f}  "
          f"{avgs['P@10']:.4f} {avgs['R@10']:.4f} {avgs['F1@10']:.4f}  "
          f"{avgs['MRR']:.4f} {avgs['MAE']:.4f} {avgs['RMSE']:.4f}")

    # ── Save per-fold summary ──────────────────────────────────────────────────
    df_fold = pd.DataFrame(fold_rows)
    df_fold.to_csv(os.path.join(OUTPUT_PATH, "blend_summary_per_fold.csv"), index=False)

    # ── Save beta grid results ─────────────────────────────────────────────────
    beta_rows = [{"beta": b, **{k: round(v, 4) for k, v in m.items()}}
                 for b, m in beta_results.items()]
    df_beta = pd.DataFrame(beta_rows)
    df_beta.to_csv(os.path.join(OUTPUT_PATH, "beta_grid_search.csv"), index=False)

    # ── Build comparison table: Part1 vs Part2 vs Part3 ───────────────────────
    print("\n[4/5] Building Part1 vs Part2 vs Part3 comparison ...")

    # Re-evaluate pure CF (beta=1.0) and pure LLM (beta=0.0)
    m_cf  = beta_results[1.0]
    m_llm = beta_results[0.0]
    m_hyb = best_m

    comparison = {
        "Metric":  ["HR@5","HR@10","HR@20",
                    "NDCG@5","NDCG@10","NDCG@20",
                    "P@10","R@10","F1@10",
                    "MRR","MAE","RMSE"],
        "Part1 CF (β=1.0)":  [round(m_cf[k],4) for k in
                               ["HR@5","HR@10","HR@20",
                                "NDCG@5","NDCG@10","NDCG@20",
                                "P@10","R@10","F1@10",
                                "MRR","MAE","RMSE"]],
        "Part2 LLM (β=0.0)": [round(m_llm[k],4) for k in
                               ["HR@5","HR@10","HR@20",
                                "NDCG@5","NDCG@10","NDCG@20",
                                "P@10","R@10","F1@10",
                                "MRR","MAE","RMSE"]],
        f"Part3 Hybrid (β={best_beta:.2f})": [round(m_hyb[k],4) for k in
                               ["HR@5","HR@10","HR@20",
                                "NDCG@5","NDCG@10","NDCG@20",
                                "P@10","R@10","F1@10",
                                "MRR","MAE","RMSE"]],
    }
    df_comp = pd.DataFrame(comparison)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║              FINAL COMPARISON SUMMARY                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(df_comp.to_string(index=False))

    # Improvement column
    improvements = []
    for metric in comparison["Metric"]:
        cf_val  = m_cf[metric]
        hyb_val = m_hyb[metric]
        if metric in ["MAE","RMSE"]:
            delta = cf_val - hyb_val         # lower is better
            sym   = "↓" if delta > 0 else ("↑" if delta < 0 else "=")
        else:
            delta = hyb_val - cf_val         # higher is better
            sym   = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        improvements.append(f"{sym}{abs(delta):.4f}")
    df_comp[f"Δ vs CF"] = improvements

    print(f"\n  ★ Hybrid uses β = {best_beta:.2f}  "
          f"(CF weight = {best_beta:.0%}, LLM weight = {1-best_beta:.0%})\n")

    df_comp.to_csv(os.path.join(OUTPUT_PATH, "final_comparison.csv"), index=False)

    # ── Sample predictions ─────────────────────────────────────────────────────
    show_samples(cf_data, llm_data, best_beta, item_meta, user_meta, n=5)

    # ── Save blend predictions CSV ─────────────────────────────────────────────
    print("[5/5] Saving blended predictions ...")
    blend_rows = []
    for fold in FOLDS:
        cf_fold  = cf_data.get(fold, {})
        llm_fold = llm_data.get(fold, {})
        for key in set(cf_fold.keys()) & set(llm_fold.keys()):
            u, i       = key
            cf_s       = float(cf_fold[key]["cf_score"])
            llm_s      = float(llm_fold[key]["llm_score"])
            blend      = best_beta * cf_s + (1.0 - best_beta) * llm_s
            act_rating = float(cf_fold[key]["actual_rating"])
            act_binary = int(cf_fold[key]["actual_binary"])
            pred_binary = int(blend >= BINARY_THRESHOLD)
            blend_rows.append({
                "fold": fold, "user": u, "item": i,
                "cf_score": round(cf_s, 4),
                "llm_score": round(llm_s, 4),
                "blend_score": round(blend, 4),
                "blend_binary": pred_binary,
                "actual_rating": act_rating,
                "actual_binary": act_binary,
                "correct": int(pred_binary == act_binary),
            })

    df_blend = pd.DataFrame(blend_rows)
    df_blend.to_csv(os.path.join(OUTPUT_PATH, "blend_predictions_all.csv"), index=False)

    # ── Files saved ────────────────────────────────────────────────────────────
    print("\n══ Files Saved ══")
    for fn in sorted(os.listdir(OUTPUT_PATH)):
        fp = os.path.join(OUTPUT_PATH, fn)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp) / 1024
            print(f"  {fn:50s}  {sz:8.1f} KB")

    total_min = (time.time() - t_start) / 60
    print(f"\n✅ Part 3 complete in {total_min:.1f} minutes.")
    print(f"   Best β = {best_beta:.2f}  |  NDCG@10 = {best_m['NDCG@10']:.4f}")
    print("   All results saved to part3_output/\n")

if __name__ == "__main__":
    main()

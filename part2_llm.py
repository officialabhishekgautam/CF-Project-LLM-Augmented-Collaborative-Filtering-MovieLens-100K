
# =============================================================================
# PART 2 — LLM-Augmented Semantic Scoring using GPT-4.1 mini
# Pipeline:
#   1. GPT-4.1 mini generates rich movie descriptions (1682 calls, async)
#   2. GPT-4.1 mini generates user taste profiles   ( 943 calls, async)
#   3. sentence-transformers embeds all text locally (free, GPU)
#   4. Cosine similarity → semantic score (1-5 scale)
#   5. Saves llm_blend_data.pkl for Part 3 blending
#   6. Shows 5 demo predictions with actual GPT prompt
# =============================================================================

import asyncio
import json
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY   = "add your OpenAI API key here"
DATA_PATH        = r"C:\Users\Douglas\OneDrive\Desktop\CF_Project\ml-100k"
OUTPUT_PATH      = r"C:\Users\Douglas\OneDrive\Desktop\CF_Project\part2_output"
PART1_OUTPUT     = r"C:\Users\Douglas\OneDrive\Desktop\CF_Project\part1_output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

GPT_MODEL        = "gpt-4o-mini"     # gpt-4.1-mini maps to this in API
EMBED_MODEL      = "BAAI/bge-base-en-v1.5"
MAX_CONCURRENT   = 40                # parallel API calls (safe under 500 RPM)
BINARY_THRESHOLD = 4.0
K_VALUES         = [5, 10, 20]
FOLDS            = [1, 2, 3, 4, 5]

# Cache files (so re-runs don't cost extra API credits)
MOVIE_DESC_CACHE = os.path.join(OUTPUT_PATH, "movie_descriptions.json")
USER_PROF_CACHE  = os.path.join(OUTPUT_PATH, "user_profiles.json")

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_ratings(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None,
                       names=["user","item","rating","ts"],
                       usecols=["user","item","rating"])

def load_all_ratings() -> pd.DataFrame:
    return load_ratings(os.path.join(DATA_PATH, "u.data"))

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
    df["year"] = df["release"].str.extract(r"\((\d{4})\)").fillna("N/A")
    return df[["item","title","year","genre_str"]]

def load_user_meta() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_PATH, "u.user"), sep="|",
                       header=None, names=["user","age","gender","occupation","zip"],
                       usecols=["user","age","gender","occupation"])

# ── PROMPT BUILDERS ───────────────────────────────────────────────────────────
def build_movie_prompt(title: str, year: str, genres: str) -> str:
    return (
        "You are building a movie recommendation system. "
        "Write a concise semantic description (50-70 words) for this movie "
        "capturing its themes, mood, tone, and target audience. "
        "Focus on content characteristics useful for matching user preferences.\n\n"
        f"Title: {title} ({year})\n"
        f"Genres: {genres}\n\n"
        "Description:"
    )

def build_user_prompt(age: int, gender: str, occupation: str,
                      loved: List[str], liked: List[str], disliked: List[str]) -> str:
    loved_str    = ", ".join(loved[:6])    if loved    else "none"
    liked_str    = ", ".join(liked[:4])    if liked    else "none"
    disliked_str = ", ".join(disliked[:3]) if disliked else "none"
    return (
        "You are building a movie recommendation system. "
        "Write a 2-3 sentence taste profile for this user based on their ratings.\n\n"
        f"Demographics: {age} year old {gender} {occupation}\n"
        f"Loved (rated 5): {loved_str}\n"
        f"Liked (rated 4): {liked_str}\n"
        f"Disliked (rated 1-2): {disliked_str}\n\n"
        "Taste profile:"
    )

def build_demo_prompt(user_profile: str, movie_title: str, movie_desc: str,
                      genre: str, year: str) -> str:
    return (
        "You are a movie recommendation system. "
        "Predict the rating (1 to 5) this user would give to this movie.\n\n"
        f"User taste profile:\n{user_profile}\n\n"
        f"Movie: {movie_title} ({year})\n"
        f"Genres: {genre}\n"
        f"Description: {movie_desc}\n\n"
        "Instructions: Reply with ONLY a single number between 1 and 5 "
        "(decimals allowed, e.g. 3.5). Nothing else."
    )

# ── ASYNC GPT CALLER ──────────────────────────────────────────────────────────
async def call_gpt_async(client: AsyncOpenAI, prompt: str,
                         semaphore: asyncio.Semaphore,
                         max_tokens: int = 120,
                         retries: int = 4) -> str:
    async with semaphore:
        for attempt in range(retries):
            try:
                resp = await client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                wait = 2 ** attempt
                if attempt < retries - 1:
                    await asyncio.sleep(wait)
                else:
                    print(f"\n  ⚠ API error after {retries} retries: {e}")
                    return ""
        return ""

async def batch_call(client, prompts: Dict[int, str],
                     semaphore: asyncio.Semaphore,
                     desc: str = "Calling GPT") -> Dict[int, str]:
    results = {}
    tasks   = {k: call_gpt_async(client, p, semaphore) for k, p in prompts.items()}
    keys    = list(tasks.keys())
    coros   = [tasks[k] for k in keys]

    print(f"  {desc}: {len(coros)} calls ...")
    t0 = time.time()
    responses = []
    batch_size = 100
    for start in range(0, len(coros), batch_size):
        chunk = coros[start:start + batch_size]
        chunk_resp = await asyncio.gather(*chunk)
        responses.extend(chunk_resp)
        elapsed = time.time() - t0
        done    = start + len(chunk)
        print(f"    {done}/{len(coros)} done  ({elapsed:.0f}s elapsed)", end="\r")

    for k, r in zip(keys, responses):
        results[k] = r
    print(f"\n  ✅ Done in {time.time()-t0:.1f}s")
    return results

# ── GENERATE OR LOAD DESCRIPTIONS ─────────────────────────────────────────────
async def get_movie_descriptions(item_meta: pd.DataFrame) -> Dict[int, str]:
    if os.path.exists(MOVIE_DESC_CACHE):
        print("  ✅ Movie descriptions loaded from cache (no API calls needed)")
        with open(MOVIE_DESC_CACHE) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    client    = AsyncOpenAI(api_key=OPENAI_API_KEY)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    prompts   = {
        int(row["item"]): build_movie_prompt(row["title"], row["year"], row["genre_str"])
        for _, row in item_meta.iterrows()
    }
    descs = await batch_call(client, prompts, semaphore, "Generating movie descriptions")

    # Fallback: use raw metadata for empty responses
    for _, row in item_meta.iterrows():
        iid = int(row["item"])
        if not descs.get(iid):
            descs[iid] = f"{row['title']} ({row['year']}). Genres: {row['genre_str']}."

    with open(MOVIE_DESC_CACHE, "w") as f:
        json.dump({str(k): v for k, v in descs.items()}, f, indent=2)
    print(f"  Saved to {MOVIE_DESC_CACHE}")
    return descs

async def get_user_profiles(user_meta: pd.DataFrame,
                             all_ratings: pd.DataFrame,
                             item_meta: pd.DataFrame) -> Dict[int, str]:
    if os.path.exists(USER_PROF_CACHE):
        print("  ✅ User profiles loaded from cache (no API calls needed)")
        with open(USER_PROF_CACHE) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    title_map = item_meta.set_index("item")["title"].to_dict()
    client    = AsyncOpenAI(api_key=OPENAI_API_KEY)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    prompts = {}
    for _, row in user_meta.iterrows():
        uid  = int(row["user"])
        urat = all_ratings[all_ratings["user"] == uid]
        loved    = [title_map.get(int(r["item"]), "?")
                    for _, r in urat[urat["rating"] == 5].iterrows()]
        liked    = [title_map.get(int(r["item"]), "?")
                    for _, r in urat[urat["rating"] == 4].iterrows()]
        disliked = [title_map.get(int(r["item"]), "?")
                    for _, r in urat[urat["rating"] <= 2].iterrows()]
        prompts[uid] = build_user_prompt(
            int(row["age"]), str(row["gender"]), str(row["occupation"]),
            loved, liked, disliked
        )

    profiles = await batch_call(client, prompts, semaphore, "Generating user profiles")

    for _, row in user_meta.iterrows():
        uid = int(row["user"])
        if not profiles.get(uid):
            profiles[uid] = (f"{row['age']}yo {row['gender']} {row['occupation']} "
                             f"with diverse movie interests.")

    with open(USER_PROF_CACHE, "w") as f:
        json.dump({str(k): v for k, v in profiles.items()}, f, indent=2)
    print(f"  Saved to {USER_PROF_CACHE}")
    return profiles

# ── EMBEDDING ─────────────────────────────────────────────────────────────────
def embed_all(movie_descs: Dict[int, str],
              user_profiles: Dict[int, str]) -> Tuple[Dict, Dict]:
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    print(f"\n  Loading embedding model: {EMBED_MODEL}  (device={device})")
    model = SentenceTransformer(EMBED_MODEL, device=device)

    # Movie embeddings
    movie_ids   = sorted(movie_descs.keys())
    movie_texts = [movie_descs[i] for i in movie_ids]
    print(f"  Embedding {len(movie_ids)} movies ...")
    movie_embs  = model.encode(movie_texts, batch_size=128,
                               normalize_embeddings=True, show_progress_bar=True)
    movie_emb_map = {mid: movie_embs[idx] for idx, mid in enumerate(movie_ids)}

    # User embeddings
    user_ids    = sorted(user_profiles.keys())
    user_texts  = [user_profiles[u] for u in user_ids]
    print(f"  Embedding {len(user_ids)} users ...")
    user_embs   = model.encode(user_texts, batch_size=128,
                               normalize_embeddings=True, show_progress_bar=True)
    user_emb_map = {uid: user_embs[idx] for idx, uid in enumerate(user_ids)}

    return movie_emb_map, user_emb_map

# ── SCORE COMPUTATION ─────────────────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Both normalized → dot product = cosine similarity
    return float(np.dot(a, b))

def sim_to_rating(sim: float, lo: float, hi: float) -> float:
    """Map cosine similarity [lo, hi] → rating [1, 5]"""
    if hi == lo:
        return 3.0
    normed = (sim - lo) / (hi - lo)
    return float(np.clip(1.0 + 4.0 * normed, 1.0, 5.0))

# ── RANKING METRICS (same as Part 1) ─────────────────────────────────────────
def dcg_at_k(rel: List[int], k: int) -> float:
    return sum(r / np.log2(idx + 2) for idx, r in enumerate(rel[:k]))

def ndcg_at_k(rel: List[int], k: int) -> float:
    ideal = sorted(rel, reverse=True)
    idcg  = dcg_at_k(ideal, k)
    return dcg_at_k(rel, k) / idcg if idcg > 0 else 0.0

def hit_rate_at_k(rel: List[int], k: int) -> float:
    return float(any(r == 1 for r in rel[:k]))

def compute_metrics(user_preds: Dict[int, List[Tuple[float, int]]],
                    k_values: List[int]) -> Dict:
    results = {f"HR@{k}": [] for k in k_values}
    results.update({f"NDCG@{k}": [] for k in k_values})
    for u, pairs in user_preds.items():
        if not any(ab == 1 for _, ab in pairs):
            continue
        pairs_sorted = sorted(pairs, key=lambda x: -x[0])
        rel = [ab for _, ab in pairs_sorted]
        for k in k_values:
            results[f"HR@{k}"].append(hit_rate_at_k(rel, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(rel, k))
    return {m: float(np.mean(v)) if v else 0.0 for m, v in results.items()}

# ── DEMO PREDICTIONS ──────────────────────────────────────────────────────────
async def run_demo(user_profiles: Dict[int, str],
                   movie_descs: Dict[int, str],
                   item_meta: pd.DataFrame,
                   user_meta: pd.DataFrame,
                   fold1_test: pd.DataFrame,
                   movie_emb_map: Dict, user_emb_map: Dict,
                   sim_lo: float, sim_hi: float):
    client    = AsyncOpenAI(api_key=OPENAI_API_KEY)
    semaphore = asyncio.Semaphore(5)
    im        = item_meta.set_index("item")
    um        = user_meta.set_index("user")
    samples   = fold1_test.head(5)

    print("\n\n══ 5 Demo Predictions (Fold 1) ══")
    for _, row in samples.iterrows():
        u      = int(row["user"])
        i      = int(row["item"])
        r_true = float(row["rating"])
        r_bin  = int(r_true >= BINARY_THRESHOLD)

        title  = im.loc[i, "title"]    if i in im.index else f"item_{i}"
        genre  = im.loc[i, "genre_str"] if i in im.index else "?"
        year   = im.loc[i, "year"]     if i in im.index else "?"
        age    = um.loc[u, "age"]       if u in um.index else "?"
        gender = um.loc[u, "gender"]    if u in um.index else "?"
        occ    = um.loc[u, "occupation"] if u in um.index else "?"

        u_emb   = user_emb_map.get(u)
        i_emb   = movie_emb_map.get(i)
        sem_sim = cosine_sim(u_emb, i_emb) if (u_emb is not None and i_emb is not None) else 0.5
        sem_scr = sim_to_rating(sem_sim, sim_lo, sim_hi)
        sem_bin = int(sem_scr >= BINARY_THRESHOLD)

        # Actual GPT zero-shot rating prediction
        demo_prompt = build_demo_prompt(
            user_profiles.get(u, ""),
            title, movie_descs.get(i, ""),
            genre, year
        )
        gpt_raw = await call_gpt_async(client, demo_prompt, semaphore, max_tokens=5)
        try:
            gpt_score = float(gpt_raw.strip().split()[0])
            gpt_score = float(np.clip(gpt_score, 1.0, 5.0))
        except Exception:
            gpt_score = sem_scr
        gpt_bin = int(gpt_score >= BINARY_THRESHOLD)

        print(f"\n  User {u:>4}  [{age}y {gender} {occ}]")
        print(f"  Movie : {title} ({year})  [{genre}]")
        print(f"  ── User Profile (GPT-generated):")
        print(f"     {user_profiles.get(u,'')[:120]}...")
        print(f"  ── Movie Description (GPT-generated):")
        print(f"     {movie_descs.get(i,'')[:120]}...")
        print(f"  ── Semantic similarity : {sem_sim:.4f}  →  mapped score = {sem_scr:.2f}")
        print(f"  ── GPT zero-shot rating: {gpt_score:.1f}  (raw: '{gpt_raw}')")
        print(f"  ── Final LLM score used: {sem_scr:.2f}  →  binary_pred={sem_bin}")
        print(f"  ── Actual rating       : {r_true:.1f}   →  binary_true={r_bin}")
        verdict = "✅ CORRECT" if sem_bin == r_bin else "❌ WRONG"
        print(f"  ── Binary match        : {verdict}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
async def main():
    t_start = time.time()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  PART 2 — LLM Semantic Scoring (GPT-4.1 mini)           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  GPT model       : {GPT_MODEL}")
    print(f"  Embedding model : {EMBED_MODEL}")
    print(f"  Max concurrent  : {MAX_CONCURRENT}")
    print(f"  Output          : {OUTPUT_PATH}\n")

    # ── Load metadata ─────────────────────────────────────────────────────────
    print("[1/5] Loading data ...")
    item_meta   = load_item_meta()
    user_meta   = load_user_meta()
    all_ratings = load_all_ratings()
    print(f"  Movies: {len(item_meta)} | Users: {len(user_meta)} | "
          f"Total ratings: {len(all_ratings):,}")

    # ── GPT: Generate descriptions ────────────────────────────────────────────
    print("\n[2/5] GPT-4.1 mini — Movie Descriptions ...")
    movie_descs = await get_movie_descriptions(item_meta)

    print("\n[3/5] GPT-4.1 mini — User Taste Profiles ...")
    user_profiles = await get_user_profiles(user_meta, all_ratings, item_meta)

    # ── Embed ─────────────────────────────────────────────────────────────────
    print("\n[4/5] Local Embedding (sentence-transformers) ...")
    movie_emb_map, user_emb_map = embed_all(movie_descs, user_profiles)

    # ── Compute test scores for all 5 folds ───────────────────────────────────
    print("\n[5/5] Computing semantic scores for all folds ...")

    # Compute global similarity range for normalization
    sample_sims = []
    for uid, u_emb in list(user_emb_map.items())[:50]:
        for iid, i_emb in list(movie_emb_map.items())[:50]:
            sample_sims.append(cosine_sim(u_emb, i_emb))
    sim_lo = float(np.percentile(sample_sims, 5))
    sim_hi = float(np.percentile(sample_sims, 95))
    print(f"  Similarity range (5-95 pct): [{sim_lo:.3f}, {sim_hi:.3f}]")

    all_preds   = []
    all_metrics = defaultdict(list)
    blend_data  = {}
    fold1_test  = None

    for fold in FOLDS:
        test  = load_ratings(os.path.join(DATA_PATH, f"u{fold}.test"))
        if fold == 1:
            fold1_test = test.copy()

        records    = []
        user_preds = defaultdict(list)

        for _, row in test.iterrows():
            u      = int(row["user"])
            i      = int(row["item"])
            r_true = float(row["rating"])
            r_bin  = int(r_true >= BINARY_THRESHOLD)

            u_emb = user_emb_map.get(u)
            i_emb = movie_emb_map.get(i)

            if u_emb is not None and i_emb is not None:
                sim   = cosine_sim(u_emb, i_emb)
                score = sim_to_rating(sim, sim_lo, sim_hi)
            else:
                sim   = 0.5
                score = 3.0

            p_bin = int(score >= BINARY_THRESHOLD)
            records.append({
                "fold": fold, "user": u, "item": i,
                "llm_score":    score,
                "llm_sim":      sim,
                "llm_binary":   p_bin,
                "actual_rating": r_true,
                "actual_binary": r_bin,
            })
            user_preds[u].append((score, r_bin))

        metrics = compute_metrics(user_preds, K_VALUES)
        all_metrics_fold = metrics
        for m, v in metrics.items():
            all_metrics[m].append(v)

        print(f"\n  ── Fold {fold} Results ──")
        for m, v in sorted(metrics.items()):
            print(f"    {m:10s} = {v:.4f}")

        df_fold = pd.DataFrame(records)
        all_preds.append(df_fold)
        df_fold.to_csv(os.path.join(OUTPUT_PATH, f"llm_predictions_fold{fold}.csv"),
                       index=False)

        blend_data[fold] = {
            (int(r["user"]), int(r["item"])): {
                "llm_score":     r["llm_score"],
                "llm_sim":       r["llm_sim"],
                "actual_rating": r["actual_rating"],
                "actual_binary": r["actual_binary"],
            }
            for _, r in df_fold.iterrows()
        }

    # ── Save everything ───────────────────────────────────────────────────────
    df_all = pd.concat(all_preds, ignore_index=True)
    df_all.to_csv(os.path.join(OUTPUT_PATH, "llm_predictions_all.csv"), index=False)

    with open(os.path.join(OUTPUT_PATH, "llm_blend_data.pkl"), "wb") as f:
        pickle.dump(blend_data, f)

    # Save embeddings for potential reuse
    np.save(os.path.join(OUTPUT_PATH, "movie_embeddings.npy"),
            {k: v.tolist() for k, v in movie_emb_map.items()})
    np.save(os.path.join(OUTPUT_PATH, "user_embeddings.npy"),
            {k: v.tolist() for k, v in user_emb_map.items()})

    # ── Summary Table ─────────────────────────────────────────────────────────
    print("\n\n╔══════════════════════════════════════════════════════════╗")
    print("║                  FINAL RESULTS SUMMARY                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    rows = []
    for fold in FOLDS:
        df_f = df_all[df_all["fold"] == fold]
        up   = defaultdict(list)
        for _, r in df_f.iterrows():
            up[int(r["user"])].append((r["llm_score"], int(r["actual_binary"])))
        m   = compute_metrics(up, K_VALUES)
        row = {"Fold": fold}
        row.update({k: round(v, 4) for k, v in m.items()})
        rows.append(row)

    avg_row = {"Fold": "AVG"}
    for m in all_metrics:
        avg_row[m] = round(float(np.mean(all_metrics[m])), 4)
    rows.append(avg_row)

    df_summary = pd.DataFrame(rows)
    metric_cols = sorted([c for c in df_summary.columns if c != "Fold"])
    df_summary  = df_summary[["Fold"] + metric_cols]
    print(df_summary.to_string(index=False))
    df_summary.to_csv(os.path.join(OUTPUT_PATH, "llm_summary.csv"), index=False)

    # ── Demo Predictions ──────────────────────────────────────────────────────
    await run_demo(user_profiles, movie_descs, item_meta, user_meta,
                   fold1_test, movie_emb_map, user_emb_map, sim_lo, sim_hi)

    # ── Files Saved ───────────────────────────────────────────────────────────
    print("\n\n══ Files Saved ══")
    for fn in sorted(os.listdir(OUTPUT_PATH)):
        fp  = os.path.join(OUTPUT_PATH, fn)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp) / 1024
            print(f"  {fn:50s}  {sz:8.1f} KB")

    total_min = (time.time() - t_start) / 60
    print(f"\n✅ Part 2 complete in {total_min:.1f} minutes.")
    print("   Ready for Part 3 (Blending).\n")

if __name__ == "__main__":
    asyncio.run(main())

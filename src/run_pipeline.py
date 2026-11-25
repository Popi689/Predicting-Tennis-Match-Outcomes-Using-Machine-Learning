"""End-to-end leakage-free ATP tennis match outcome prediction pipeline.

Loads 2018-2026 ATP Tour singles matches, constructs strictly pre-match
chronological features (overall & surface Elo, ATP rankings, rolling serve/return
efficiency, form, head-to-head, and workload), tunes models on 2024 validation
data, evaluates on the 2025-2026 holdout test set, and generates figures and
structured metrics for the project blog.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "assets" / "figures"

ROUND_ORDER = {
    "ER": 0,
    "RR": 1,
    "R128": 2,
    "R64": 3,
    "R32": 4,
    "R16": 5,
    "QF": 6,
    "SF": 7,
    "BR": 8,
    "F": 9,
}

# League-average priors for cold-start smoothing
PRIOR_WIN_RATE = 0.50
PRIOR_ACE_RATE = 0.075
PRIOR_DF_RATE = 0.035
PRIOR_1ST_IN = 0.62
PRIOR_1ST_WON = 0.72
PRIOR_2ND_WON = 0.51
PRIOR_BP_SAVED = 0.62
PRIOR_RET_WON = 0.37


def load_matches(years: range) -> tuple[pd.DataFrame, dict]:
    """Load and clean ATP singles matches across the specified seasons."""
    frames = []
    for year in years:
        csv_path = DATA_DIR / f"atp_matches_{year}.csv"
        if csv_path.exists():
            df_y = pd.read_csv(csv_path, low_memory=False)
            df_y["season"] = year
            frames.append(df_y)

    raw_df = pd.concat(frames, ignore_index=True)
    raw_count = len(raw_df)

    # Filter out unplayed walkovers / defaults and missing player IDs
    score_str = raw_df["score"].fillna("").astype(str)
    is_walkover = score_str.str.contains(r"\bW/O\b|\bDEF\b|Walkover", case=False, regex=True)
    valid_ids = raw_df["winner_id"].notna() & raw_df["loser_id"].notna()
    df = raw_df.loc[~is_walkover & valid_ids].copy()

    df["tourney_date_dt"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    df["round_ord"] = df["round"].map(ROUND_ORDER).fillna(5)
    df["match_num_num"] = pd.to_numeric(df["match_num"], errors="coerce").fillna(0)
    df["surface_clean"] = df["surface"].where(df["surface"].isin(["Hard", "Clay", "Grass"]), "Hard")

    df = df.sort_values(
        ["tourney_date_dt", "tourney_id", "round_ord", "match_num_num"],
        kind="mergesort",
    ).reset_index(drop=True)

    summary = {
        "raw_matches": int(raw_count),
        "walkovers_removed": int(is_walkover.sum()),
        "played_matches": int(len(df)),
        "seasons": [int(y) for y in sorted(df["season"].unique())],
        "unique_players": int(pd.unique(pd.concat([df["winner_id"], df["loser_id"]])).size),
    }
    return df, summary


def expected_elo_prob(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def smoothed_mean(values: deque, prior: float, weight: float = 3.0) -> float:
    if not values:
        return prior
    return float((sum(values) + weight * prior) / (len(values) + weight))


def safe_ratio(num: float, den: float) -> float | None:
    if pd.isna(num) or pd.isna(den) or den <= 0:
        return None
    return float(num / den)


def build_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Construct strictly pre-match features in chronological order."""
    rng = np.random.default_rng(seed)

    elo_overall: dict[int, float] = defaultdict(lambda: 1500.0)
    elo_surface: dict[tuple[int, str], float] = defaultdict(lambda: 1500.0)
    matches_played: dict[int, int] = defaultdict(int)
    wins_all: dict[int, int] = defaultdict(int)

    recent_wins_10: dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
    recent_wins_25: dict[int, deque] = defaultdict(lambda: deque(maxlen=25))
    surface_wins_15: dict[tuple[int, str], deque] = defaultdict(lambda: deque(maxlen=15))

    ace_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=15))
    df_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=15))
    first_in_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=15))
    first_won_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=15))
    second_won_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=15))
    bp_saved_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=15))
    ret_won_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=15))

    h2h_wins: dict[tuple[int, int], int] = defaultdict(int)
    tourney_matches: dict[tuple[str, int], int] = defaultdict(int)
    match_dates: dict[int, deque] = defaultdict(lambda: deque(maxlen=40))

    rows = []

    for row in df.itertuples(index=False):
        w_id = int(row.winner_id)
        l_id = int(row.loser_id)
        surf = row.surface_clean
        t_id = str(row.tourney_id)
        t_date = row.tourney_date_dt
        t_level = str(row.tourney_level) if pd.notna(row.tourney_level) else "A"
        best_of = int(row.best_of) if pd.notna(row.best_of) else 3

        # Helper to extract pre-match snapshot for a player
        def snapshot(pid: int, opp_id: int, rank_val, pts_val, age_val, ht_val, hand_val) -> dict:
            r_val = float(rank_val) if pd.notna(rank_val) and rank_val > 0 else 500.0
            p_val = float(pts_val) if pd.notna(pts_val) and pts_val >= 0 else 10.0
            a_val = float(age_val) if pd.notna(age_val) else np.nan
            h_val = float(ht_val) if pd.notna(ht_val) else np.nan
            hd = str(hand_val) if pd.notna(hand_val) else "U"

            recent_14d = 0
            if pd.notna(t_date):
                for prev_d in match_dates[pid]:
                    if 0 <= (t_date - prev_d).days <= 21:
                        recent_14d += 1

            n_tot = matches_played[pid]
            career_wr = (wins_all[pid] + 3.0 * PRIOR_WIN_RATE) / (n_tot + 6.0)
            return {
                "id": pid,
                "elo": elo_overall[pid],
                "surf_elo": elo_surface[(pid, surf)],
                "rank": r_val,
                "log_rank": math.log(r_val),
                "rank_pts": p_val,
                "log_pts": math.log1p(p_val),
                "career_wr": career_wr,
                "wr_10": smoothed_mean(recent_wins_10[pid], PRIOR_WIN_RATE, 2.5),
                "wr_25": smoothed_mean(recent_wins_25[pid], PRIOR_WIN_RATE, 4.0),
                "surf_wr_15": smoothed_mean(surface_wins_15[(pid, surf)], PRIOR_WIN_RATE, 3.0),
                "ace_rate": smoothed_mean(ace_hist[pid], PRIOR_ACE_RATE, 3.0),
                "df_rate": smoothed_mean(df_hist[pid], PRIOR_DF_RATE, 3.0),
                "first_in": smoothed_mean(first_in_hist[pid], PRIOR_1ST_IN, 3.0),
                "first_won": smoothed_mean(first_won_hist[pid], PRIOR_1ST_WON, 3.0),
                "second_won": smoothed_mean(second_won_hist[pid], PRIOR_2ND_WON, 3.0),
                "bp_saved": smoothed_mean(bp_saved_hist[pid], PRIOR_BP_SAVED, 3.0),
                "ret_won": smoothed_mean(ret_won_hist[pid], PRIOR_RET_WON, 3.0),
                "h2h": h2h_wins[(pid, opp_id)],
                "matches_exp": math.log1p(n_tot),
                "tourney_load": tourney_matches[(t_id, pid)],
                "recent_21d_load": recent_14d,
                "age": a_val,
                "ht": h_val,
                "hand": hd,
            }

        w_snap = snapshot(w_id, l_id, row.winner_rank, row.winner_rank_points, row.winner_age, row.winner_ht, row.winner_hand)
        l_snap = snapshot(l_id, w_id, row.loser_rank, row.loser_rank_points, row.loser_age, row.loser_ht, row.loser_hand)

        # Symmetrize Player A vs Player B
        a_is_winner = bool(rng.random() < 0.5)
        a, b = (w_snap, l_snap) if a_is_winner else (l_snap, w_snap)
        target = 1 if a_is_winner else 0

        # Track whether higher-ranked player won (for EDA)
        higher_rank_won = int(w_snap["rank"] < l_snap["rank"]) if w_snap["rank"] != l_snap["rank"] else np.nan
        upset_flag = int(w_snap["rank"] > l_snap["rank"]) if w_snap["rank"] != l_snap["rank"] else 0

        rows.append({
            "season": int(row.season),
            "tourney_date": str(row.tourney_date),
            "tourney_name": str(row.tourney_name),
            "tourney_level": t_level,
            "surface": surf,
            "round": str(row.round),
            "best_of": best_of,
            "score": str(row.score),
            "winner_name": str(row.winner_name),
            "loser_name": str(row.loser_name),
            "winner_rank": w_snap["rank"],
            "loser_rank": l_snap["rank"],
            "player_a_name": str(row.winner_name) if a_is_winner else str(row.loser_name),
            "player_b_name": str(row.loser_name) if a_is_winner else str(row.winner_name),
            "higher_rank_won": higher_rank_won,
            "upset_flag": upset_flag,
            "target": target,
            # Features (Player A - Player B)
            "elo_diff": a["elo"] - b["elo"],
            "surf_elo_diff": a["surf_elo"] - b["surf_elo"],
            "blended_elo_diff": 0.6 * (a["elo"] - b["elo"]) + 0.4 * (a["surf_elo"] - b["surf_elo"]),
            "rank_diff": a["rank"] - b["rank"],
            "log_rank_diff": b["log_rank"] - a["log_rank"],  # positive => A ranked better
            "log_pts_diff": a["log_pts"] - b["log_pts"],
            "career_wr_diff": a["career_wr"] - b["career_wr"],
            "wr_10_diff": a["wr_10"] - b["wr_10"],
            "wr_25_diff": a["wr_25"] - b["wr_25"],
            "surf_wr_15_diff": a["surf_wr_15"] - b["surf_wr_15"],
            "ace_rate_diff": a["ace_rate"] - b["ace_rate"],
            "df_rate_diff": a["df_rate"] - b["df_rate"],
            "first_in_diff": a["first_in"] - b["first_in"],
            "first_won_diff": a["first_won"] - b["first_won"],
            "second_won_diff": a["second_won"] - b["second_won"],
            "bp_saved_diff": a["bp_saved"] - b["bp_saved"],
            "ret_won_diff": a["ret_won"] - b["ret_won"],
            "h2h_diff": a["h2h"] - b["h2h"],
            "exp_diff": a["matches_exp"] - b["matches_exp"],
            "tourney_load_diff": a["tourney_load"] - b["tourney_load"],
            "recent_21d_load_diff": a["recent_21d_load"] - b["recent_21d_load"],
            "age_diff": a["age"] - b["age"],
            "ht_diff": a["ht"] - b["ht"],
            "a_is_lefty": int(a["hand"] == "L") - int(b["hand"] == "L"),
            "is_best_of_5": int(best_of == 5),
            "is_clay": int(surf == "Clay"),
            "is_grass": int(surf == "Grass"),
            "is_grand_slam": int(t_level == "G"),
            "is_masters": int(t_level == "M"),
        })

        # ---- Update post-match states AFTER recording pre-match snapshot ----
        k_overall = 36.0 if t_level == "G" else (32.0 if t_level == "M" else 28.0)
        p_w = expected_elo_prob(elo_overall[w_id], elo_overall[l_id])
        elo_overall[w_id] += k_overall * (1.0 - p_w)
        elo_overall[l_id] += k_overall * (0.0 - (1.0 - p_w))

        k_surf = 30.0
        p_ws = expected_elo_prob(elo_surface[(w_id, surf)], elo_surface[(l_id, surf)])
        elo_surface[(w_id, surf)] += k_surf * (1.0 - p_ws)
        elo_surface[(l_id, surf)] += k_surf * (0.0 - (1.0 - p_ws))

        matches_played[w_id] += 1
        matches_played[l_id] += 1
        wins_all[w_id] += 1

        recent_wins_10[w_id].append(1)
        recent_wins_10[l_id].append(0)
        recent_wins_25[w_id].append(1)
        recent_wins_25[l_id].append(0)
        surface_wins_15[(w_id, surf)].append(1)
        surface_wins_15[(l_id, surf)].append(0)

        h2h_wins[(w_id, l_id)] += 1
        tourney_matches[(t_id, w_id)] += 1
        tourney_matches[(t_id, l_id)] += 1
        if pd.notna(t_date):
            match_dates[w_id].append(t_date)
            match_dates[l_id].append(t_date)

        # Serve & return stats update
        def update_serve(pid: int, ace, df_v, svpt, first_in, first_won, second_won, bp_saved, bp_faced,
                         opp_svpt, opp_first_won, opp_second_won):
            r_ace = safe_ratio(ace, svpt)
            r_df = safe_ratio(df_v, svpt)
            r_1in = safe_ratio(first_in, svpt)
            r_1w = safe_ratio(first_won, first_in)
            r_2w = safe_ratio(second_won, (svpt - first_in) if pd.notna(svpt) and pd.notna(first_in) else np.nan)
            r_bps = safe_ratio(bp_saved, bp_faced)
            opp_won_total = (opp_first_won + opp_second_won) if pd.notna(opp_first_won) and pd.notna(opp_second_won) else np.nan
            r_ret = safe_ratio((opp_svpt - opp_won_total) if pd.notna(opp_svpt) and pd.notna(opp_won_total) else np.nan, opp_svpt)

            if r_ace is not None:
                ace_hist[pid].append(r_ace)
            if r_df is not None:
                df_hist[pid].append(r_df)
            if r_1in is not None:
                first_in_hist[pid].append(r_1in)
            if r_1w is not None:
                first_won_hist[pid].append(r_1w)
            if r_2w is not None:
                second_won_hist[pid].append(r_2w)
            if r_bps is not None:
                bp_saved_hist[pid].append(r_bps)
            if r_ret is not None:
                ret_won_hist[pid].append(r_ret)

        update_serve(w_id, row.w_ace, row.w_df, row.w_svpt, row.w_1stIn, row.w_1stWon, row.w_2ndWon,
                     row.w_bpSaved, row.w_bpFaced, row.l_svpt, row.l_1stWon, row.l_2ndWon)
        update_serve(l_id, row.l_ace, row.l_df, row.l_svpt, row.l_1stIn, row.l_1stWon, row.l_2ndWon,
                     row.l_bpSaved, row.l_bpFaced, row.w_svpt, row.w_1stWon, row.w_2ndWon)

    return pd.DataFrame(rows)


FEATURE_COLS = [
    "elo_diff",
    "surf_elo_diff",
    "blended_elo_diff",
    "log_rank_diff",
    "log_pts_diff",
    "career_wr_diff",
    "wr_10_diff",
    "wr_25_diff",
    "surf_wr_15_diff",
    "ace_rate_diff",
    "df_rate_diff",
    "first_in_diff",
    "first_won_diff",
    "second_won_diff",
    "bp_saved_diff",
    "ret_won_diff",
    "h2h_diff",
    "exp_diff",
    "tourney_load_diff",
    "recent_21d_load_diff",
    "age_diff",
    "ht_diff",
    "a_is_lefty",
    "is_best_of_5",
    "is_clay",
    "is_grass",
    "is_grand_slam",
    "is_masters",
]

FEATURE_LABELS = {
    "elo_diff": "Overall Elo Diff",
    "surf_elo_diff": "Surface Elo Diff",
    "blended_elo_diff": "Blended Elo Diff",
    "log_rank_diff": "Log ATP Rank Diff",
    "log_pts_diff": "Log ATP Points Diff",
    "career_wr_diff": "Career Win Rate Diff",
    "wr_10_diff": "Last-10 Form Diff",
    "wr_25_diff": "Last-25 Form Diff",
    "surf_wr_15_diff": "Surface Form (15) Diff",
    "ace_rate_diff": "Ace Rate Diff",
    "df_rate_diff": "Double-Fault Rate Diff",
    "first_in_diff": "1st Serve % Diff",
    "first_won_diff": "1st Serve Won % Diff",
    "second_won_diff": "2nd Serve Won % Diff",
    "bp_saved_diff": "Break Points Saved % Diff",
    "ret_won_diff": "Return Points Won % Diff",
    "h2h_diff": "Head-to-Head Diff",
    "exp_diff": "Tour Experience Diff",
    "tourney_load_diff": "Current Tourney Load Diff",
    "recent_21d_load_diff": "21-Day Match Load Diff",
    "age_diff": "Age Diff",
    "ht_diff": "Height Diff",
    "a_is_lefty": "Left-Handed Matchup",
    "is_best_of_5": "Best-of-5 Flag",
    "is_clay": "Clay Surface Flag",
    "is_grass": "Grass Surface Flag",
    "is_grand_slam": "Grand Slam Flag",
    "is_masters": "Masters 1000 Flag",
}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_prob_clipped = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    y_pred = (y_prob_clipped >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob_clipped)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob_clipped)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob_clipped)), 4),
    }


def setup_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#f8fafc",
        "axes.edgecolor": "#cbd5e1",
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def generate_eda_figures(feat_df: pd.DataFrame) -> dict:
    setup_plot_style()
    eda_df = feat_df[feat_df["season"] >= 2019].copy()

    # 1. Surface & tournament level favorite win rates
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    surf_stats = (
        eda_df.dropna(subset=["higher_rank_won"])
        .groupby("surface")
        .agg(matches=("target", "count"), fav_win=("higher_rank_won", "mean"))
        .reindex(["Hard", "Clay", "Grass"])
    )
    colors_surf = ["#2563eb", "#c2410c", "#15803d"]
    bars = axes[0].bar(surf_stats.index, surf_stats["fav_win"] * 100, color=colors_surf, width=0.55, edgecolor="#0f172a", linewidth=0.6)
    axes[0].set_ylim(50, 75)
    axes[0].set_title("Higher-Ranked Player Win Rate by Surface (2019-2026)")
    axes[0].set_ylabel("Higher-Ranked Win Rate (%)")
    for bar, (_, r) in zip(bars, surf_stats.iterrows()):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                     f"{r['fav_win']*100:.1f}%\n(n={int(r['matches']):,})", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    level_map = {"G": "Grand Slam (Bo5)", "M": "Masters 1000", "A": "ATP 250/500"}
    eda_df["level_label"] = eda_df["tourney_level"].map(level_map)
    lvl_stats = (
        eda_df.dropna(subset=["higher_rank_won", "level_label"])
        .groupby("level_label")
        .agg(matches=("target", "count"), fav_win=("higher_rank_won", "mean"))
        .reindex(["Grand Slam (Bo5)", "Masters 1000", "ATP 250/500"])
    )
    bars2 = axes[1].bar(lvl_stats.index, lvl_stats["fav_win"] * 100, color=["#0f172a", "#0284c7", "#64748b"], width=0.55, edgecolor="#0f172a", linewidth=0.6)
    axes[1].set_ylim(50, 78)
    axes[1].set_title("Predictability by Tournament Tier (2019-2026)")
    axes[1].set_ylabel("Higher-Ranked Win Rate (%)")
    for bar, (_, r) in zip(bars2, lvl_stats.iterrows()):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                     f"{r['fav_win']*100:.1f}%\n(n={int(r['matches']):,})", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_surface_upsets.png", dpi=180)
    plt.close(fig)

    # 2. Empirical win rate by Elo difference and Log-Rank difference
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    elo_bins = pd.qcut(eda_df["elo_diff"], q=12, duplicates="drop")
    elo_curve = eda_df.groupby(elo_bins, observed=True).agg(
        mean_diff=("elo_diff", "mean"),
        win_rate=("target", "mean"),
        n=("target", "count"),
    )
    axes[0].plot(elo_curve["mean_diff"], elo_curve["win_rate"] * 100, marker="o", linewidth=2.4, color="#15803d")
    axes[0].axhline(50, color="#94a3b8", linestyle=":", linewidth=1.2)
    axes[0].axvline(0, color="#94a3b8", linestyle=":", linewidth=1.2)
    axes[0].set_title("Player A Win Rate vs. Pre-Match Elo Difference")
    axes[0].set_xlabel("Pre-Match Elo Difference (Player A - Player B)")
    axes[0].set_ylabel("Empirical Win Rate (%)")
    axes[0].set_ylim(10, 92)

    rank_bins = pd.qcut(eda_df["log_rank_diff"], q=12, duplicates="drop")
    rank_curve = eda_df.groupby(rank_bins, observed=True).agg(
        mean_diff=("log_rank_diff", "mean"),
        win_rate=("target", "mean"),
    )
    axes[1].plot(rank_curve["mean_diff"], rank_curve["win_rate"] * 100, marker="s", linewidth=2.4, color="#2563eb")
    axes[1].axhline(50, color="#94a3b8", linestyle=":", linewidth=1.2)
    axes[1].axvline(0, color="#94a3b8", linestyle=":", linewidth=1.2)
    axes[1].set_title("Player A Win Rate vs. Log ATP Rank Advantage")
    axes[1].set_xlabel("log(Rank_B) - log(Rank_A)  [positive = A ranked higher]")
    axes[1].set_ylabel("Empirical Win Rate (%)")
    axes[1].set_ylim(10, 92)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_rank_vs_winrate.png", dpi=180)
    plt.close(fig)

    return {
        "surface_summary": {
            idx: {"matches": int(r["matches"]), "higher_rank_win_rate": round(float(r["fav_win"]), 4)}
            for idx, r in surf_stats.iterrows()
        },
        "tier_summary": {
            idx: {"matches": int(r["matches"]), "higher_rank_win_rate": round(float(r["fav_win"]), 4)}
            for idx, r in lvl_stats.iterrows()
        },
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df, data_summary = load_matches(range(2018, 2027))
    feat_df = build_features(df, seed=42)
    eda_stats = generate_eda_figures(feat_df)

    # Temporal split: 2018 = warm-up, 2019-2023 = train, 2024 = validation, 2025-2026 = holdout test
    train_df = feat_df[(feat_df["season"] >= 2019) & (feat_df["season"] <= 2023)].copy()
    val_df = feat_df[feat_df["season"] == 2024].copy()
    test_df = feat_df[feat_df["season"] >= 2025].copy()

    X_train, y_train = train_df[FEATURE_COLS], train_df["target"].to_numpy()
    X_val, y_val = val_df[FEATURE_COLS], val_df["target"].to_numpy()
    X_test, y_test = test_df[FEATURE_COLS], test_df["target"].to_numpy()

    # Baseline 1: ATP Rank Heuristic (tie -> 0.5, scaled smoothly via logistic of log_rank_diff for log-loss/AUC)
    # Plus strict deterministic rank-based accuracy
    rank_prob_test = 1.0 / (1.0 + np.exp(-0.85 * test_df["log_rank_diff"].to_numpy()))
    rank_prob_val = 1.0 / (1.0 + np.exp(-0.85 * val_df["log_rank_diff"].to_numpy()))

    # Baseline 2: Pure Pre-Match Elo Probability
    elo_prob_val = 1.0 / (1.0 + 10.0 ** (-val_df["elo_diff"].to_numpy() / 400.0))
    elo_prob_test = 1.0 / (1.0 + 10.0 ** (-test_df["elo_diff"].to_numpy() / 400.0))

    models_results = {}
    test_probs = {
        "Baseline: ATP Rank": rank_prob_test,
        "Baseline: Pure Elo": elo_prob_test,
    }
    models_results["Baseline: ATP Rank"] = {
        "type": "Heuristic Baseline",
        "val": compute_metrics(y_val, rank_prob_val),
        "test": compute_metrics(y_test, rank_prob_test),
    }
    models_results["Baseline: Pure Elo"] = {
        "type": "Rating Baseline",
        "val": compute_metrics(y_val, elo_prob_val),
        "test": compute_metrics(y_test, elo_prob_test),
    }

    # Tune and train ML models on train -> select best C / hyperparams on 2024 val -> refit on train+val -> evaluate on 2025-2026 test
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val])

    # 1. Logistic Regression
    best_c, best_ll = 0.1, float("inf")
    for c_val in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=c_val, max_iter=1000, random_state=42)),
        ])
        pipe.fit(X_train, y_train)
        ll = log_loss(y_val, pipe.predict_proba(X_val)[:, 1])
        if ll < best_ll:
            best_ll, best_c = ll, c_val

    lr_val_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=best_c, max_iter=1000, random_state=42)),
    ]).fit(X_train, y_train)
    lr_final = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=best_c, max_iter=1000, random_state=42)),
    ]).fit(X_trainval, y_trainval)
    lr_test_p = lr_final.predict_proba(X_test)[:, 1]
    test_probs["Logistic Regression"] = lr_test_p
    models_results["Logistic Regression"] = {
        "type": f"Linear (C={best_c})",
        "val": compute_metrics(y_val, lr_val_pipe.predict_proba(X_val)[:, 1]),
        "test": compute_metrics(y_test, lr_test_p),
    }

    # 2. Random Forest
    best_rf_cfg, best_rf_ll = (250, 10, 15), float("inf")
    for max_d in [8, 10, 14]:
        for min_leaf in [10, 20]:
            rf = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=250, max_depth=max_d, min_samples_leaf=min_leaf, random_state=42, n_jobs=-1)),
            ])
            rf.fit(X_train, y_train)
            ll = log_loss(y_val, rf.predict_proba(X_val)[:, 1])
            if ll < best_rf_ll:
                best_rf_ll = ll
                best_rf_cfg = (250, max_d, min_leaf)

    rf_val = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=best_rf_cfg[0], max_depth=best_rf_cfg[1], min_samples_leaf=best_rf_cfg[2], random_state=42, n_jobs=-1)),
    ]).fit(X_train, y_train)
    rf_final = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=best_rf_cfg[0], max_depth=best_rf_cfg[1], min_samples_leaf=best_rf_cfg[2], random_state=42, n_jobs=-1)),
    ]).fit(X_trainval, y_trainval)
    rf_test_p = rf_final.predict_proba(X_test)[:, 1]
    test_probs["Random Forest"] = rf_test_p
    models_results["Random Forest"] = {
        "type": f"Bagged Trees (depth={best_rf_cfg[1]})",
        "val": compute_metrics(y_val, rf_val.predict_proba(X_val)[:, 1]),
        "test": compute_metrics(y_test, rf_test_p),
    }

    # 3. HistGradientBoostingClassifier
    best_hgb_cfg, best_hgb_ll = (0.03, 4, 40), float("inf")
    for lr in [0.03, 0.05]:
        for max_d in [3, 4, 6]:
            for min_leaf in [30, 60]:
                hgb = HistGradientBoostingClassifier(
                    learning_rate=lr, max_depth=max_d, max_iter=250, min_samples_leaf=min_leaf,
                    l2_regularization=1.0, random_state=42
                )
                hgb.fit(X_train, y_train)
                ll = log_loss(y_val, hgb.predict_proba(X_val)[:, 1])
                if ll < best_hgb_ll:
                    best_hgb_ll = ll
                    best_hgb_cfg = (lr, max_d, min_leaf)

    hgb_val = HistGradientBoostingClassifier(
        learning_rate=best_hgb_cfg[0], max_depth=best_hgb_cfg[1], max_iter=250,
        min_samples_leaf=best_hgb_cfg[2], l2_regularization=1.0, random_state=42
    ).fit(X_train, y_train)
    hgb_final = HistGradientBoostingClassifier(
        learning_rate=best_hgb_cfg[0], max_depth=best_hgb_cfg[1], max_iter=250,
        min_samples_leaf=best_hgb_cfg[2], l2_regularization=1.0, random_state=42
    ).fit(X_trainval, y_trainval)
    hgb_test_p = hgb_final.predict_proba(X_test)[:, 1]
    test_probs["HistGradientBoosting"] = hgb_test_p
    models_results["HistGradientBoosting"] = {
        "type": f"Gradient Boosting (depth={best_hgb_cfg[1]})",
        "val": compute_metrics(y_val, hgb_val.predict_proba(X_val)[:, 1]),
        "test": compute_metrics(y_test, hgb_test_p),
    }

    # 4. MLP Neural Network
    mlp_val = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.5, max_iter=300, early_stopping=True, random_state=42)),
    ]).fit(X_train, y_train)
    mlp_final = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.5, max_iter=300, early_stopping=True, random_state=42)),
    ]).fit(X_trainval, y_trainval)
    mlp_test_p = mlp_final.predict_proba(X_test)[:, 1]
    test_probs["MLP Neural Net"] = mlp_test_p
    models_results["MLP Neural Net"] = {
        "type": "Neural Network (64-32)",
        "val": compute_metrics(y_val, mlp_val.predict_proba(X_val)[:, 1]),
        "test": compute_metrics(y_test, mlp_test_p),
    }

    # Pick best model by lowest test log loss (or val log loss)
    best_model_name = min(models_results.keys(), key=lambda m: models_results[m]["val"]["log_loss"])
    best_probs = test_probs[best_model_name]

    # ---- Figure 3: Model Comparison ----
    setup_plot_style()
    names = list(models_results.keys())
    accs = [models_results[m]["test"]["accuracy"] * 100 for m in names]
    aucs = [models_results[m]["test"]["roc_auc"] for m in names]
    losses = [models_results[m]["test"]["log_loss"] for m in names]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    palette = ["#94a3b8", "#64748b", "#0284c7", "#0d9488", "#15803d", "#7c3aed"]

    for ax, vals, title, fmt, ylim in [
        (axes[0], accs, "Test Accuracy (%) ↑", "{:.2f}%", (60, max(accs) + 3.5)),
        (axes[1], aucs, "Test ROC-AUC ↑", "{:.4f}", (0.65, max(aucs) + 0.04)),
        (axes[2], losses, "Test Log Loss ↓", "{:.4f}", (min(losses) - 0.03, max(losses) + 0.04)),
    ]:
        bars = ax.barh(names[::-1], vals[::-1], color=palette[::-1], edgecolor="#0f172a", linewidth=0.5)
        ax.set_xlim(ylim)
        ax.set_title(title)
        for bar, v in zip(bars, vals[::-1]):
            ax.text(v + (ylim[1] - ylim[0]) * 0.015, bar.get_y() + bar.get_height() / 2,
                    fmt.format(v), va="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    # ---- Figure 4: ROC Curve & Calibration Curve ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    line_styles = {
        "Baseline: ATP Rank": ("#94a3b8", "--"),
        "Baseline: Pure Elo": ("#475569", ":"),
        "Logistic Regression": ("#0284c7", "-"),
        "Random Forest": ("#0d9488", "-."),
        "HistGradientBoosting": ("#15803d", "-"),
        "MLP Neural Net": ("#7c3aed", "-"),
    }
    for m in names:
        fpr, tpr, _ = roc_curve(y_test, test_probs[m])
        c, ls = line_styles[m]
        axes[0].plot(fpr, tpr, color=c, linestyle=ls, linewidth=2.0,
                     label=f"{m} (AUC={models_results[m]['test']['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k:", alpha=0.4)
    axes[0].set_title("ROC Curves (2025-2026 Holdout Test Set)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=7.8, loc="lower right")

    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
    for m in ["Baseline: Pure Elo", "Logistic Regression", "Random Forest", "HistGradientBoosting"]:
        prob_true, prob_pred = calibration_curve(y_test, test_probs[m], n_bins=10, strategy="quantile")
        c, ls = line_styles[m]
        axes[1].plot(prob_pred, prob_true, marker="o", markersize=4.5, color=c, linestyle=ls, linewidth=2.0, label=m)
    axes[1].set_title("Probability Calibration (2025-2026 Test Set)")
    axes[1].set_xlabel("Mean Predicted Win Probability")
    axes[1].set_ylabel("Observed Win Frequency")
    axes[1].legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_and_calibration.png", dpi=180)
    plt.close(fig)

    # ---- Figure 5: Permutation Feature Importance on 2025-2026 Test Set ----
    best_estimator = {"Logistic Regression": lr_final, "Random Forest": rf_final, "HistGradientBoosting": hgb_final, "MLP Neural Net": mlp_final}[best_model_name]
    perm = permutation_importance(best_estimator, X_test, y_test, n_repeats=8, random_state=42, scoring="neg_log_loss")
    imp_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "label": [FEATURE_LABELS[f] for f in FEATURE_COLS],
        "importance": perm.importances_mean,
        "std": perm.importances_std,
    }).sort_values("importance", ascending=False)

    top_imp = imp_df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.barh(top_imp["label"], top_imp["importance"], xerr=top_imp["std"], color="#15803d", edgecolor="#0f172a", linewidth=0.5, capsize=3)
    ax.set_title(f"Top 12 Pre-Match Features by Permutation Importance ({best_model_name})")
    ax.set_xlabel("Mean Increase in Log Loss When Permuted (Test Set)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)

    # ---- Figure 6: Surface & Format Breakdown ----
    test_eval_df = test_df.copy()
    test_eval_df["pred_prob"] = best_probs
    test_eval_df["pred_correct"] = ((best_probs >= 0.5).astype(int) == y_test).astype(int)
    test_eval_df["elo_correct"] = ((elo_prob_test >= 0.5).astype(int) == y_test).astype(int)

    surf_bk = test_eval_df.groupby("surface").agg(
        matches=("target", "count"),
        model_acc=("pred_correct", "mean"),
        elo_acc=("elo_correct", "mean"),
    ).reindex(["Hard", "Clay", "Grass"])

    bo_bk = test_eval_df.groupby("best_of").agg(
        matches=("target", "count"),
        model_acc=("pred_correct", "mean"),
        elo_acc=("elo_correct", "mean"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(surf_bk))
    w = 0.34
    b1 = axes[0].bar(x - w / 2, surf_bk["elo_acc"] * 100, w, label="Pure Elo Baseline", color="#64748b")
    b2 = axes[0].bar(x + w / 2, surf_bk["model_acc"] * 100, w, label=best_model_name, color="#15803d")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{s}\n(n={int(surf_bk.loc[s, 'matches'])})" for s in surf_bk.index])
    axes[0].set_ylim(55, 78)
    axes[0].set_title("Test Accuracy by Court Surface (2025-2026)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    for b in list(b1) + list(b2):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    bo_labels = [f"Best-of-{int(k)}\n(n={int(bo_bk.loc[k, 'matches'])})" for k in bo_bk.index]
    x2 = np.arange(len(bo_bk))
    b3 = axes[1].bar(x2 - w / 2, bo_bk["elo_acc"] * 100, w, label="Pure Elo Baseline", color="#64748b")
    b4 = axes[1].bar(x2 + w / 2, bo_bk["model_acc"] * 100, w, label=best_model_name, color="#0284c7")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(bo_labels)
    axes[1].set_ylim(55, 82)
    axes[1].set_title("Test Accuracy by Match Format (2025-2026)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    for b in list(b3) + list(b4):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "surface_accuracy_breakdown.png", dpi=180)
    plt.close(fig)

    # Extract notable real Grand Slam Final / Semi-final test predictions for case studies
    slam_finals = test_eval_df[(test_eval_df["tourney_level"] == "G") & (test_eval_df["round"] == "F")].copy()
    case_studies = []
    for _, r in slam_finals.head(8).iterrows():
        p_a = float(r["pred_prob"])
        pred_winner = r["player_a_name"] if p_a >= 0.5 else r["player_b_name"]
        pred_conf = p_a if p_a >= 0.5 else (1.0 - p_a)
        case_studies.append({
            "date": r["tourney_date"],
            "tournament": r["tourney_name"],
            "surface": r["surface"],
            "round": r["round"],
            "winner": r["winner_name"],
            "loser": r["loser_name"],
            "winner_rank": int(r["winner_rank"]),
            "loser_rank": int(r["loser_rank"]),
            "score": r["score"],
            "predicted_winner": pred_winner,
            "predicted_prob": round(pred_conf, 4),
            "correct": bool(pred_winner == r["winner_name"]),
        })

    output_payload = {
        "dataset": data_summary,
        "splits": {
            "warmup_2018": int((feat_df["season"] == 2018).sum()),
            "train_2019_2023": int(len(train_df)),
            "val_2024": int(len(val_df)),
            "test_2025_2026": int(len(test_df)),
        },
        "eda": eda_stats,
        "best_model": best_model_name,
        "models": models_results,
        "top_features": [
            {"feature": r["feature"], "label": r["label"], "importance": round(float(r["importance"]), 5)}
            for _, r in imp_df.head(10).iterrows()
        ],
        "surface_test_breakdown": {
            s: {
                "matches": int(surf_bk.loc[s, "matches"]),
                "elo_acc": round(float(surf_bk.loc[s, "elo_acc"]), 4),
                "model_acc": round(float(surf_bk.loc[s, "model_acc"]), 4),
            }
            for s in surf_bk.index
        },
        "format_test_breakdown": {
            f"Best-of-{int(k)}": {
                "matches": int(bo_bk.loc[k, "matches"]),
                "elo_acc": round(float(bo_bk.loc[k, "elo_acc"]), 4),
                "model_acc": round(float(bo_bk.loc[k, "model_acc"]), 4),
            }
            for k in bo_bk.index
        },
        "grand_slam_final_cases": case_studies,
    }

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print("Pipeline complete! Best model:", best_model_name)
    for m, res in models_results.items():
        print(f"  {m:24s} | Test Acc: {res['test']['accuracy']*100:5.2f}% | AUC: {res['test']['roc_auc']:.4f} | LogLoss: {res['test']['log_loss']:.4f}")


if __name__ == "__main__":
    main()

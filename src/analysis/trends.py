"""Trend detection: weekly issue counts, z-score and EWMA spike detection against a trailing baseline."""
import numpy as np
import pandas as pd


def assign_timeline(df: pd.DataFrame, cluster_col: str, spikes: list[dict], n_weeks: int = 12, seed: int = 0) -> pd.Series:
    """Give every review a week 0..n_weeks-1. Reviews are spread uniformly, then for each planted spike
    {cluster, week, factor} extra reviews of that cluster are moved into that week until its count is
    about `factor` times the cluster's normal weekly count."""
    rng = np.random.default_rng(seed)
    week = pd.Series(rng.integers(0, n_weeks, len(df)), index=df.index)
    for sp in spikes:
        idx = df.index[df[cluster_col] == sp["cluster"]]
        normal = len(idx) / n_weeks
        target = int(round(sp["factor"] * normal))
        already = int((week[idx] == sp["week"]).sum())
        movable = [i for i in idx if week[i] != sp["week"]]
        move = rng.choice(movable, size=min(max(target - already, 0), len(movable)), replace=False)
        week.loc[move] = sp["week"]
    return week


def weekly_counts(df: pd.DataFrame, week: pd.Series, group_col: str, n_weeks: int = 12) -> pd.DataFrame:
    """Rows = weeks, columns = groups (e.g. clusters)."""
    return (pd.crosstab(week, df[group_col]).reindex(range(n_weeks), fill_value=0))


def detect_spikes(counts: pd.DataFrame, window: int = 4, z_thresh: float = 3.5, min_count: int = 8,
                  alpha: float = 0.3, ewma_L: float = 3.0) -> pd.DataFrame:
    """For each group and week t >= window, compare with the trailing `window` weeks.
    z-score rule: z >= z_thresh and count >= min_count.
    EWMA rule: EWMA of counts (weight alpha) above mean + L * sd * sqrt(alpha / (2 - alpha)).
    sd is floored at sqrt(mean) (Poisson) so quiet groups do not produce huge z-scores."""
    rows = []
    for g in counts.columns:
        x = counts[g].to_numpy(dtype=float)
        ewma = x[:window].mean()
        for t in range(window):
            ewma = alpha * x[t] + (1 - alpha) * ewma
        for t in range(window, len(x)):
            base = x[t - window:t]
            mu = base.mean()
            sd = max(base.std(ddof=1), np.sqrt(max(mu, 1.0)))
            z = (x[t] - mu) / sd
            ewma = alpha * x[t] + (1 - alpha) * ewma
            ucl = mu + ewma_L * sd * np.sqrt(alpha / (2 - alpha))
            rows.append({
                "group": g, "week": t, "count": int(x[t]), "prev_week": int(x[t - 1]),
                "wow_delta": int(x[t] - x[t - 1]), "baseline_mean": round(mu, 2), "z": round(float(z), 2),
                "flag_z": bool(z >= z_thresh and x[t] >= min_count), "flag_ewma": bool(ewma > ucl and x[t] >= min_count),
            })
    return pd.DataFrame(rows)


def evaluate(flags: pd.DataFrame, spikes: list[dict], rule: str) -> dict:
    """Precision / recall of flagged (group, week) pairs vs planted spikes, plus Precision@k (k = number planted)."""
    planted = {(sp["cluster"], sp["week"]) for sp in spikes}
    flagged = {(r.group, r.week) for r in flags.itertuples() if getattr(r, rule)}
    tp = len(flagged & planted)
    top = flags.sort_values("z", ascending=False).head(len(planted))
    p_at_k = sum((r.group, r.week) in planted for r in top.itertuples()) / len(planted)
    return {"tp": tp, "fp": len(flagged - planted), "fn": len(planted - flagged), "n_flagged": len(flagged),
            "precision": tp / len(flagged) if flagged else float("nan"), "recall": tp / len(planted),
            "precision_at_k": p_at_k}

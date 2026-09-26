import numpy as np
import pandas as pd

from src.analysis import trends as tr


def flat_counts(weeks=12, groups=("a", "b"), level=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({g: rng.poisson(level, weeks) for g in groups})


def test_planted_spike_is_flagged():
    counts = flat_counts()
    counts.loc[9, "a"] = 30
    flags = tr.detect_spikes(counts)
    hit = flags[(flags.group == "a") & (flags.week == 9)].iloc[0]
    assert hit.flag_z and hit.wow_delta > 15


def test_quiet_series_has_no_alerts_at_default_threshold():
    flags = tr.detect_spikes(flat_counts(level=6, seed=3))
    assert not flags.flag_z.any()


def test_small_counts_never_alert_even_with_high_z():
    counts = flat_counts(level=0)
    counts.loc[8, "a"] = 5  # z is large against a zero baseline but below min_count=8
    assert not tr.detect_spikes(counts).flag_z.any()


def test_no_baseline_weeks_are_not_scored():
    flags = tr.detect_spikes(flat_counts())
    assert flags.week.min() == 4


def test_timeline_moves_reviews_into_spike_week():
    df = pd.DataFrame({"cluster": [0] * 60 + [1] * 60})
    spikes = [{"cluster": 0, "week": 7, "factor": 5.0}]
    week = tr.assign_timeline(df, "cluster", spikes, seed=1)
    counts = tr.weekly_counts(df, week, "cluster")
    assert counts.loc[7, 0] >= 20 and counts.loc[7, 1] < 15

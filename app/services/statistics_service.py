from __future__ import annotations

from math import atanh, tanh, sqrt, erfc
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


TRANSFORMS = {
    "Percentage change": "pct",
    "Absolute change": "diff",
}


def normalize_series(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or not {"date", "value"}.issubset(frame.columns):
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype="float64")})
    out = frame[["date", "value"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "value"])
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


STATISTICAL_CHANGE_MODES = {
    "Log return (%)": "log",
    "Z-score of log return": "zlog",
    "Percentage change (%)": "pct",
    "Z-score of percentage change": "zpct",
    "Absolute change": "diff",
    "Z-score of absolute change": "zdiff",
}


def _change_values(values: pd.Series, window: int, mode: str) -> pd.Series:
    """Calculate one variable's change series on the supplied observation grid."""
    prior = values.shift(window)
    if mode == "pct":
        result = (values / prior - 1.0) * 100.0
        result = result.where(prior > 0)
    elif mode == "log":
        valid = (values > 0) & (prior > 0)
        result = pd.Series(np.nan, index=values.index, dtype=float)
        result.loc[valid] = np.log(values.loc[valid] / prior.loc[valid]) * 100.0
    elif mode == "diff":
        result = values.diff(window)
    else:
        raise ValueError("Unsupported base change mode")
    return result.replace([np.inf, -np.inf], np.nan)


def change_series(
    frame: pd.DataFrame,
    window: int = 1,
    mode: str = "pct",
    standardize_start_date=None,
    standardize_end_date=None,
) -> pd.DataFrame:
    """Calculate changes; z-score modes scale using the selected study period.

    For pair analyses, raw prices are timestamp-matched before this function's
    equivalent transform is applied, so both returns span the same matched grid.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    out = normalize_series(frame)
    if out.empty:
        out["change"] = pd.Series(dtype="float64")
        return out[["date", "change"]]
    standardized = mode.startswith("z")
    base_mode = mode[1:] if standardized else mode
    if base_mode not in {"pct", "log", "diff"}:
        raise ValueError("mode must be one of pct, log, diff, zpct, zlog, zdiff")
    out["change"] = _change_values(out["value"], window, base_mode)
    if standardized:
        reference = out["change"].copy()
        if standardize_start_date is not None:
            reference = reference[out["date"] >= pd.Timestamp(standardize_start_date)]
        if standardize_end_date is not None:
            end = pd.Timestamp(standardize_end_date)
            reference = reference[out["date"] < end + pd.Timedelta(days=1)] if end == end.normalize() else reference[out["date"] <= end]
        mean = float(reference.mean()) if reference.notna().any() else np.nan
        std = float(reference.std(ddof=1)) if reference.notna().sum() >= 2 else np.nan
        if np.isfinite(std) and std > 0:
            out["change"] = (out["change"] - mean) / std
        else:
            out["change"] = np.nan
    return out[["date", "change"]].replace([np.inf, -np.inf], np.nan)


def _date_filter(frame: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:
    """Filter by inclusive calendar dates, retaining all intraday bars on end_date."""
    out = frame.copy()
    if start_date is not None:
        out = out[out["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        end = pd.Timestamp(end_date)
        if end == end.normalize():
            out = out[out["date"] < end + pd.Timedelta(days=1)]
        else:
            out = out[out["date"] <= end]
    return out


def paired_change_data(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    window: int = 1,
    mode: str = "pct",
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Align raw levels on exact timestamps, then calculate both changes on that grid.

    Calculating returns before alignment can compare unequal elapsed intervals when one
    series has missing bars. This function deliberately intersects timestamps first.
    """
    p = normalize_series(primary).rename(columns={"value": "primary_value"})
    c = normalize_series(comparison).rename(columns={"value": "comparison_value"})
    joined = p.merge(c, on="date", how="inner").sort_values("date").reset_index(drop=True)
    joined = _date_filter(joined, start_date, end_date)
    # Keep the prior matched observation available for the first selected return.
    if start_date is not None:
        raw_all = normalize_series(primary).rename(columns={"value": "primary_value"}).merge(
            normalize_series(comparison).rename(columns={"value": "comparison_value"}), on="date", how="inner"
        ).sort_values("date").reset_index(drop=True)
        cutoff = pd.Timestamp(start_date)
        before = raw_all[raw_all["date"] < cutoff].tail(window)
        joined = pd.concat([before, joined], ignore_index=True).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    standardized = mode.startswith("z")
    base_mode = mode[1:] if standardized else mode
    if base_mode not in {"pct", "log", "diff"}:
        raise ValueError("mode must be one of pct, log, diff, zpct, zlog, zdiff")
    joined["primary_change"] = _change_values(joined["primary_value"], window, base_mode)
    joined["comparison_change"] = _change_values(joined["comparison_value"], window, base_mode)
    # Standardization is estimated only inside the chosen study dates, separately for
    # each series. This gives both scatter axes unitless, comparable sigma units.
    if standardized:
        for column in ("primary_change", "comparison_change"):
            reference = joined[column].copy()
            if start_date is not None:
                reference = reference[joined["date"] >= pd.Timestamp(start_date)]
            if end_date is not None:
                end = pd.Timestamp(end_date)
                reference = reference[joined["date"] < end + pd.Timedelta(days=1)] if end == end.normalize() else reference[joined["date"] <= end]
            mean = float(reference.mean()) if reference.notna().any() else np.nan
            std = float(reference.std(ddof=1)) if reference.notna().sum() >= 2 else np.nan
            joined[column] = (joined[column] - mean) / std if np.isfinite(std) and std > 0 else np.nan
    joined = _date_filter(joined, start_date, end_date)
    return joined.dropna(subset=["primary_change", "comparison_change"])[["date", "primary_change", "comparison_change"]].reset_index(drop=True)


def _fisher_correlation_interval(r: float | None, n: float) -> tuple[float | None, float | None]:
    """Approximate Fisher-z 95% interval using the supplied effective sample size."""
    if r is None or n < 4 or not np.isfinite(r):
        return None, None
    # Exact +/-1 correlations in short windows make Fisher z infinite. Return a
    # conservative full-range interval rather than implying perfect certainty.
    if abs(r) >= 1.0:
        return -1.0, 1.0
    clipped = float(np.clip(r, -0.999999, 0.999999))
    z = atanh(clipped)
    margin = 1.96 / sqrt(n - 3)
    return tanh(z - margin), tanh(z + margin)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation for finite arrays without pandas overhead."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return None
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    denom = float(np.sqrt(np.dot(xc, xc) * np.dot(yc, yc)))
    if denom <= 0 or not np.isfinite(denom):
        return None
    return float(np.clip(np.dot(xc, yc) / denom, -1.0, 1.0))


def _acf_at_lag(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    if lag <= 0 or len(values) <= lag + 2:
        return 0.0
    return _safe_corr(values[:-lag], values[lag:]) or 0.0


def effective_sample_size_for_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_acf_lag: int | None = None,
) -> float:
    """Bartlett-style effective sample-size approximation for serially dependent pairs.

    It discounts nominal sample size when both series have positive serial dependence.
    This is an approximation for uncertainty guidance, not an exact correction for
    arbitrary non-stationarity, overlapping returns, or selection across many pairs.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    n = int(min(len(x), len(y)))
    if n < 4:
        return float(n)
    k_max = min(max_acf_lag if max_acf_lag is not None else min(10, n // 4), n // 4)
    denominator = 1.0
    for lag in range(1, max(0, k_max) + 1):
        taper = 1.0 - lag / (k_max + 1.0) if k_max >= 1 else 0.0
        denominator += 2.0 * taper * _acf_at_lag(x, lag) * _acf_at_lag(y, lag)
    if not np.isfinite(denominator) or denominator <= 0:
        return float(n)
    return float(np.clip(n / denominator, 1.0, float(n)))


def pair_statistics(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    window: int = 1,
    mode: str = "pct",
    start_date=None,
    end_date=None,
) -> dict[str, float | int | None]:
    joined = paired_change_data(primary, comparison, window, mode, start_date, end_date)
    n = int(len(joined))
    if n < 2:
        return {"observations": n, "effective_observations": float(n), "pearson": None, "spearman": None, "slope": None, "pearson_ci_low": None, "pearson_ci_high": None}
    x = joined["primary_change"].to_numpy(dtype=float)
    y = joined["comparison_change"].to_numpy(dtype=float)
    pearson = _safe_corr(x, y)
    spearman = joined["primary_change"].corr(joined["comparison_change"], method="spearman")
    slope = float(np.polyfit(x, y, 1)[0]) if np.std(x) > 0 else None
    effective_n = effective_sample_size_for_correlation(x, y, max_acf_lag=min(max(10, window - 1), n // 4))
    ci_low, ci_high = _fisher_correlation_interval(pearson, effective_n)
    # Approximate two-sided Fisher-z p-value using the same effective-n estimate as
    # the interval. It is explicitly approximate for dependent financial observations.
    p_value = None
    if pearson is not None and effective_n > 3 and abs(pearson) < 1:
        fisher_z = abs(atanh(float(pearson))) * sqrt(max(effective_n - 3.0, 0.0))
        p_value = float(erfc(fisher_z / sqrt(2.0)))
    elif pearson is not None and abs(pearson) >= 1 and effective_n > 3:
        p_value = 0.0
    return {
        "observations": n,
        "effective_observations": effective_n,
        "pearson": pearson,
        "spearman": float(spearman) if pd.notna(spearman) else None,
        "slope": slope if slope is None or np.isfinite(slope) else None,
        "pearson_ci_low": ci_low,
        "pearson_ci_high": ci_high,
        "approx_p_value": p_value,
    }


def correlation_strength(correlation: float | None) -> str:
    """Cohen-style conventional magnitude labels; these are not universal cutoffs."""
    if correlation is None or not np.isfinite(correlation):
        return "N/A"
    magnitude = abs(float(correlation))
    if magnitude < 0.10:
        return "Negligible"
    if magnitude < 0.30:
        return "Small"
    if magnitude < 0.50:
        return "Moderate"
    return "Large"


def relationship_diagnostic(
    stats: Mapping[str, float | int | None],
    family_size: int = 1,
    alpha: float = 0.05,
    minimum_abs_correlation: float = 0.30,
    minimum_observations: int = 60,
    minimum_effective_observations: float = 30.0,
) -> dict[str, float | int | str | None]:
    """Classify pair evidence using magnitude, uncertainty, support and multiplicity.

    GREEN is deliberately strict: medium-or-larger effect (|r| >= .30), approximate
    95% interval excluding zero, and Bonferroni-adjusted Fisher-z p <= alpha, with
    at least 60 matched observations and effective n >= 30. These are transparent
    conservative UI gates, not universal scientific laws or trading rules.
    """
    r = stats.get("pearson")
    low = stats.get("pearson_ci_low")
    high = stats.get("pearson_ci_high")
    p = stats.get("approx_p_value")
    n = int(stats.get("observations") or 0)
    n_eff = float(stats.get("effective_observations") or 0.0)
    family = max(1, int(family_size))
    adjusted_p = min(1.0, float(p) * family) if p is not None and np.isfinite(p) else None
    enough_data = n >= minimum_observations and n_eff >= minimum_effective_observations
    ci_excludes_zero = low is not None and high is not None and (float(low) > 0 or float(high) < 0)
    practical = r is not None and np.isfinite(r) and abs(float(r)) >= minimum_abs_correlation
    significant = adjusted_p is not None and adjusted_p <= alpha
    if not enough_data or r is None or adjusted_p is None or low is None or high is None:
        status = "GREY"
    elif practical and significant and ci_excludes_zero:
        status = "GREEN"
    else:
        status = "RED"
    return {
        "status": status,
        "strength": correlation_strength(float(r) if r is not None else None),
        "family_size": family,
        "adjusted_p_value": adjusted_p,
        "enough_data": enough_data,
        "ci_excludes_zero": bool(ci_excludes_zero),
        "practical_effect": bool(practical),
        "statistically_supported": bool(significant and ci_excludes_zero),
    }

def correlation_matrix(
    series: Mapping[str, pd.DataFrame],
    window: int = 1,
    mode: str = "pct",
    method: str = "pearson",
    start_date=None,
    end_date=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return pairwise correlations after exact raw-timestamp alignment.

    Each pair gets its own matched timestamp grid. It does not calculate changes on
    each source's native grid and then accidentally compare unequal observation spans.
    """
    labels = list(series)
    corr = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    counts = pd.DataFrame(0, index=labels, columns=labels, dtype="Int64")
    for i, left in enumerate(labels):
        for j in range(i, len(labels)):
            right = labels[j]
            paired = paired_change_data(series[left], series[right], window, mode, start_date, end_date)
            x = paired["primary_change"]
            y = paired["comparison_change"]
            n = len(paired)
            value = None
            if n >= 3:
                value = _safe_corr(x.to_numpy(dtype=float), y.to_numpy(dtype=float)) if method == "pearson" else x.corr(y, method=method)
                if pd.isna(value):
                    value = None
            corr.loc[left, right] = value
            corr.loc[right, left] = value
            counts.loc[left, right] = n
            counts.loc[right, left] = n
    return corr, counts

def rolling_pair_correlation(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    window: int = 1,
    mode: str = "pct",
    rolling_observations: int = 30,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Rolling Pearson correlation with a Bartlett-effective-n Fisher interval.

    The effective-n adjustment accounts approximately for serial dependence in each
    rolling window. Bands remain pointwise and overlapping windows are not independent.
    """
    if rolling_observations < 4:
        raise ValueError("rolling_observations must be >= 4")
    joined = paired_change_data(primary, comparison, window, mode, start_date, end_date)
    x_all = joined["primary_change"].to_numpy(dtype=float)
    y_all = joined["comparison_change"].to_numpy(dtype=float)
    corr_values: list[float | None] = []
    ci_lows: list[float | None] = []
    ci_highs: list[float | None] = []
    effective_ns: list[float | None] = []
    for end in range(len(joined)):
        begin = end - rolling_observations + 1
        if begin < 0:
            corr_values.append(None); ci_lows.append(None); ci_highs.append(None); effective_ns.append(None)
            continue
        x = x_all[begin:end + 1]
        y = y_all[begin:end + 1]
        corr = _safe_corr(x, y)
        eff_n = effective_sample_size_for_correlation(x, y, max_acf_lag=min(max(10, window - 1), rolling_observations // 4))
        low, high = _fisher_correlation_interval(corr, eff_n)
        corr_values.append(corr); ci_lows.append(low); ci_highs.append(high); effective_ns.append(eff_n)
    joined["rolling_correlation"] = corr_values
    joined["rolling_ci_low"] = ci_lows
    joined["rolling_ci_high"] = ci_highs
    joined["effective_observations"] = effective_ns
    joined["rolling_observations"] = rolling_observations
    return joined[["date", "rolling_correlation", "rolling_ci_low", "rolling_ci_high", "rolling_observations", "effective_observations"]]

def _corr_for_offset(x: np.ndarray, y: np.ndarray, offset: int) -> tuple[float | None, int]:
    """Align x[t] with y[t+offset]; positive offset means y is later."""
    n = min(len(x), len(y))
    if abs(offset) >= n - 2:
        return None, 0
    if offset > 0:
        xa, ya = x[:n - offset], y[offset:n]
    elif offset < 0:
        xa, ya = x[-offset:n], y[:n + offset]
    else:
        xa, ya = x[:n], y[:n]
    valid = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[valid], ya[valid]
    return _safe_corr(xa, ya), int(len(xa))


def _moving_block_lag_intervals(
    x: np.ndarray,
    y: np.ndarray,
    offsets: list[int],
    window: int,
    replicates: int = 199,
) -> tuple[dict[int, tuple[float | None, float | None]], int | None]:
    """Moving-block bootstrap intervals for lag correlations; resamples paired blocks."""
    n = len(x)
    # A lagged correlation depends on paired observations separated by the largest
    # tested offset. Blocks shorter than that split the relevant pair across artificial
    # block seams and can badly bias its bootstrap distribution. Require at least three
    # blocks that cover the full lag span plus the change window.
    required_block = max((abs(int(offset)) for offset in offsets), default=0) + int(window)
    if n < max(30, 3 * required_block) or replicates < 50:
        return {offset: (None, None) for offset in offsets}, None
    proposed = max(required_block, int(np.ceil(n ** (1.0 / 3.0))), 2)
    block_length = min(proposed, max(2, n // 3))
    blocks_needed = int(np.ceil(n / block_length))
    rng = np.random.default_rng(20261016 + n + window + max(offsets or [0]))
    collected: dict[int, list[float]] = {offset: [] for offset in offsets}
    for _ in range(replicates):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = np.concatenate([(np.arange(start, start + block_length) % n) for start in starts])[:n]
        xb, yb = x[indices], y[indices]
        for offset in offsets:
            corr, _ = _corr_for_offset(xb, yb, offset)
            if corr is not None and np.isfinite(corr):
                collected[offset].append(corr)
    intervals: dict[int, tuple[float | None, float | None]] = {}
    for offset, samples in collected.items():
        if len(samples) >= max(50, int(replicates * 0.5)):
            intervals[offset] = (float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))
        else:
            intervals[offset] = (None, None)
    return intervals, block_length


def _circular_shift_maxstat_pvalues(
    x: np.ndarray,
    y: np.ndarray,
    offsets: list[int],
    permutations: int = 399,
) -> tuple[dict[int, float | None], int]:
    """Family-wise p-values from a circular-shift max-statistic null.

    A circular shift preserves each series' broad autocorrelation structure while
    breaking its original alignment. Shifts close to the tested lag range are excluded.
    The procedure assumes the aligned change process is sufficiently stationary for
    circular shifts to be meaningful; it is exploratory, not causal identification.
    """
    n = min(len(x), len(y))
    max_offset = max((abs(value) for value in offsets), default=0)
    exclusion = max_offset + 1
    candidates = np.arange(exclusion + 1, n - exclusion, dtype=int)
    if len(candidates) < 20 or permutations < 20:
        return {offset: None for offset in offsets}, 0
    rng = np.random.default_rng(20261017 + n + max_offset)
    count = min(int(permutations), len(candidates))
    shifts = rng.choice(candidates, size=count, replace=False)
    null_maxima: list[float] = []
    for shift in shifts:
        shifted_y = np.roll(y, int(shift))
        magnitudes: list[float] = []
        for offset in offsets:
            corr, _ = _corr_for_offset(x, shifted_y, offset)
            if corr is not None and np.isfinite(corr):
                magnitudes.append(abs(corr))
        if magnitudes:
            null_maxima.append(max(magnitudes))
    if not null_maxima:
        return {offset: None for offset in offsets}, 0
    null = np.asarray(null_maxima, dtype=float)
    adjusted: dict[int, float | None] = {}
    for offset in offsets:
        corr, _ = _corr_for_offset(x, y, offset)
        if corr is None:
            adjusted[offset] = None
        else:
            # Comparing every lag to the maximum null statistic controls selection
            # of the strongest lag over the tested family (family-wise, conservative).
            adjusted[offset] = float((1 + np.count_nonzero(null >= abs(corr))) / (len(null) + 1))
    return adjusted, len(null)


def lead_lag_statistics(
    primary: pd.DataFrame,
    comparison: pd.DataFrame,
    window: int = 1,
    mode: str = "pct",
    max_lag: int = 10,
    start_date=None,
    end_date=None,
    bootstrap_replicates: int = 199,
    permutations: int = 399,
) -> pd.DataFrame:
    """Lead/lag correlation using non-overlapping return-window offsets.

    `max_lag` means the number of change windows to test in either direction. If
    window=5 and max_lag=3, tested offsets are -15,-10,-5,0,+5,+10,+15 matched
    observations. Each adjacent tested alignment is shifted by a full change window,
    avoiding overlap between the primary and comparison change intervals. This helps
    with interval overlap but does not eliminate autocorrelation or nonstationarity.

    CIs use moving-block bootstrap intervals. Family-wise adjusted p-values compare
    each observed lag to the maximum absolute correlation across lags under circular
    shifts of the comparison series, preserving serial structure approximately.
    """
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")
    if window < 1:
        raise ValueError("window must be >= 1")
    joined = paired_change_data(primary, comparison, window, mode, start_date, end_date)
    x = joined["primary_change"].to_numpy(dtype=float)
    y = joined["comparison_change"].to_numpy(dtype=float)
    date_diffs = joined["date"].sort_values().diff().dt.total_seconds().div(60.0).dropna()
    median_step_minutes = float(date_diffs[date_diffs > 0].median()) if (date_diffs > 0).any() else np.nan
    lag_windows = list(range(-max_lag, max_lag + 1))
    offsets = [step * window for step in lag_windows]
    boot_ci, block_length = _moving_block_lag_intervals(x, y, offsets, window, bootstrap_replicates)
    adjusted_p, null_count = _circular_shift_maxstat_pvalues(x, y, offsets, permutations)
    records = []
    for step, offset in zip(lag_windows, offsets):
        corr, n = _corr_for_offset(x, y, offset)
        ci_low, ci_high = boot_ci.get(offset, (None, None))
        records.append({
            "lag_windows": step,
            "lag_observations": offset,
            "lag_minutes_approx": float(offset * median_step_minutes) if np.isfinite(median_step_minutes) else None,
            "correlation": corr,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "observations": int(n),
            "familywise_adjusted_p_value": adjusted_p.get(offset),
            "tested_lags_count": len(offsets),
            "null_shift_replicates": null_count,
            "bootstrap_block_length": block_length,
            "change_window_observations": window,
        })
    return pd.DataFrame(records)

def _future_response(frame: pd.DataFrame, horizon: int, mode: str) -> pd.DataFrame:
    out = normalize_series(frame)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    future = out["value"].shift(-horizon)
    if mode == "pct":
        response = (future / out["value"] - 1.0) * 100.0
        response = response.mask(out["value"] <= 0)
    elif mode == "diff":
        response = future - out["value"]
    else:
        raise ValueError("mode must be 'pct' or 'diff'")
    return pd.DataFrame({"date": out["date"], "response": response}).replace([np.inf, -np.inf], np.nan)


def _sample_summary(sample: pd.Series, prefix: str = "") -> dict:
    values = pd.to_numeric(sample, errors="coerce").dropna().astype(float)
    n = int(values.size)
    mean = float(values.mean()) if n else None
    std = float(values.std(ddof=1)) if n > 1 else None
    # Conventional approximation only. Event returns overlap and may be dependent.
    half_width = 1.96 * std / sqrt(n) if n > 1 and std is not None else None
    return {
        f"{prefix}observations": n,
        f"{prefix}mean": mean,
        f"{prefix}median": float(values.median()) if n else None,
        f"{prefix}positive_pct": float((values > 0).mean() * 100) if n else None,
        f"{prefix}negative_pct": float((values < 0).mean() * 100) if n else None,
        f"{prefix}std": std,
        f"{prefix}ci_low": mean - half_width if mean is not None and half_width is not None else None,
        f"{prefix}ci_high": mean + half_width if mean is not None and half_width is not None else None,
    }


def _block_bootstrap_event_comparison(
    aligned: pd.DataFrame,
    event_direction: str,
    horizon: int,
    replicates: int = 400,
) -> dict[str, float | int | None]:
    """Moving-block bootstrap intervals for event mean, baseline mean and difference.

    Resamples contiguous blocks from the chronological matched-observation sequence.
    This approximately preserves some local dependence and overlapping-horizon effects;
    it is not a substitute for a pre-specified inferential design. Returns no interval
    for samples too small to support a minimally useful block bootstrap.
    """
    clean = aligned[["direction", "response"]].copy().reset_index(drop=True)
    clean["response"] = pd.to_numeric(clean["response"], errors="coerce")
    n = len(clean)
    if n < 20:
        return {
            "block_bootstrap_event_ci_low": None,
            "block_bootstrap_event_ci_high": None,
            "block_bootstrap_baseline_ci_low": None,
            "block_bootstrap_baseline_ci_high": None,
            "block_bootstrap_difference_ci_low": None,
            "block_bootstrap_difference_ci_high": None,
            "bootstrap_valid_replicates": 0,
            "bootstrap_block_length": None,
        }

    # Forward-horizon returns overlap for up to `horizon` observations. Keep blocks
    # at least horizon-long when possible, but retain at least three blocks overall.
    block_length = max(2, min(int(horizon), max(2, n // 3)))
    directions = clean["direction"].to_numpy(dtype=object)
    responses = clean["response"].to_numpy(dtype=float)
    rng = np.random.default_rng(20261016 + int(horizon) + (1 if event_direction == "Increase" else 2))
    event_means: list[float] = []
    baseline_means: list[float] = []
    differences: list[float] = []
    blocks_needed = int(np.ceil(n / block_length))

    for _ in range(int(replicates)):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = np.concatenate([
            (np.arange(start, start + block_length) % n) for start in starts
        ])[:n]
        sampled_directions = directions[indices]
        sampled_responses = responses[indices]
        event_mask = (sampled_directions == event_direction) & np.isfinite(sampled_responses)
        baseline_mask = (sampled_directions == "No event") & np.isfinite(sampled_responses)
        if event_mask.sum() < 2 or baseline_mask.sum() < 2:
            continue
        event_mean = float(sampled_responses[event_mask].mean())
        baseline_mean = float(sampled_responses[baseline_mask].mean())
        event_means.append(event_mean)
        baseline_means.append(baseline_mean)
        differences.append(event_mean - baseline_mean)

    if len(differences) < max(50, int(replicates * 0.25)):
        return {
            "block_bootstrap_event_ci_low": None,
            "block_bootstrap_event_ci_high": None,
            "block_bootstrap_baseline_ci_low": None,
            "block_bootstrap_baseline_ci_high": None,
            "block_bootstrap_difference_ci_low": None,
            "block_bootstrap_difference_ci_high": None,
            "bootstrap_valid_replicates": len(differences),
            "bootstrap_block_length": block_length,
        }

    q = lambda values, prob: float(np.quantile(np.asarray(values, dtype=float), prob))
    return {
        "block_bootstrap_event_ci_low": q(event_means, 0.025),
        "block_bootstrap_event_ci_high": q(event_means, 0.975),
        "block_bootstrap_baseline_ci_low": q(baseline_means, 0.025),
        "block_bootstrap_baseline_ci_high": q(baseline_means, 0.975),
        "block_bootstrap_difference_ci_low": q(differences, 0.025),
        "block_bootstrap_difference_ci_high": q(differences, 0.975),
        "bootstrap_valid_replicates": len(differences),
        "bootstrap_block_length": block_length,
    }


def event_study(
    primary: pd.DataFrame,
    target: pd.DataFrame,
    measurement_window: int = 5,
    threshold: float = 2.0,
    primary_mode: str = "pct",
    response_mode: str = "pct",
    horizons: Iterable[int] = (1, 3, 5, 20),
    start_date=None,
    end_date=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare subsequent target responses after events with a matched non-event baseline.

    Event dates are in the requested start/end window. Target forward responses may
    extend past end_date so long as those future observations are already available.
    Event-vs-baseline uncertainty is estimated with a circular moving-block bootstrap
    over chronologically matched observations to preserve some local dependence.
    The interval is exploratory, can be unavailable for small/sparse samples, and does
    not correct for selecting thresholds/horizons after looking at results.
    """
    primary_changes = change_series(primary, measurement_window, primary_mode).rename(columns={"change": "primary_change"})
    primary_changes = _date_filter(primary_changes, start_date, end_date)
    target_values = normalize_series(target)
    matched = primary_changes.dropna(subset=["primary_change"]).merge(target_values, on="date", how="inner").sort_values("date").reset_index(drop=True)
    matched["is_event"] = (matched["primary_change"] >= threshold) | (matched["primary_change"] <= -threshold)
    matched["direction"] = np.where(
        matched["primary_change"] >= threshold, "Increase",
        np.where(matched["primary_change"] <= -threshold, "Decrease", "No event"),
    )

    observations: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for horizon in sorted(set(int(h) for h in horizons if int(h) > 0)):
        response = _future_response(target, horizon, response_mode)
        aligned = matched.merge(response, on="date", how="left").sort_values("date").reset_index(drop=True)
        aligned["horizon"] = horizon
        aligned["event_value_change"] = aligned["primary_change"]
        event_horizon = aligned[aligned["is_event"]].copy()
        observations.append(event_horizon[["date", "direction", "primary_change", "horizon", "value", "response"]].copy())
        baseline_sample = aligned.loc[aligned["direction"] == "No event", "response"].dropna().astype(float)
        baseline_summary = _sample_summary(baseline_sample, prefix="baseline_")
        for direction in ("Increase", "Decrease"):
            event_sample = aligned.loc[aligned["direction"] == direction, "response"].dropna().astype(float)
            stats = _sample_summary(event_sample)
            n_events = int((aligned["direction"] == direction).sum())
            boot = _block_bootstrap_event_comparison(aligned, direction, horizon)
            row = {
                "horizon_observations": horizon,
                "primary_direction": direction,
                "events_detected": n_events,
                "realized_responses": stats["observations"],
                "mean_response": stats["mean"],
                "median_response": stats["median"],
                "positive_response_pct": stats["positive_pct"],
                "negative_response_pct": stats["negative_pct"],
                "response_std": stats["std"],
                # Retain legacy fields for compatibility, but populate them with
                # block-bootstrap intervals rather than independent-observation CIs.
                "approx_mean_ci_low": boot["block_bootstrap_event_ci_low"],
                "approx_mean_ci_high": boot["block_bootstrap_event_ci_high"],
                "block_bootstrap_event_ci_low": boot["block_bootstrap_event_ci_low"],
                "block_bootstrap_event_ci_high": boot["block_bootstrap_event_ci_high"],
                "block_bootstrap_baseline_ci_low": boot["block_bootstrap_baseline_ci_low"],
                "block_bootstrap_baseline_ci_high": boot["block_bootstrap_baseline_ci_high"],
                "block_bootstrap_difference_ci_low": boot["block_bootstrap_difference_ci_low"],
                "block_bootstrap_difference_ci_high": boot["block_bootstrap_difference_ci_high"],
                "bootstrap_valid_replicates": boot["bootstrap_valid_replicates"],
                "bootstrap_block_length": boot["bootstrap_block_length"],
                **baseline_summary,
            }
            baseline_mean = baseline_summary["baseline_mean"]
            row["mean_difference_vs_baseline"] = (
                stats["mean"] - baseline_mean
                if stats["mean"] is not None and baseline_mean is not None else None
            )
            summaries.append(row)
    observation_frame = pd.concat(observations, ignore_index=True) if observations else pd.DataFrame(
        columns=["date", "direction", "primary_change", "horizon", "value", "response"]
    )
    return pd.DataFrame(summaries), observation_frame


def variable_summary(frame: pd.DataFrame, start_date=None, end_date=None) -> dict[str, float | int | str | None]:
    """Summarize first-to-last change inside requested dates, not sum of daily changes."""
    out = normalize_series(frame)
    if not out.empty:
        out = _date_filter(out, start_date, end_date)
    if out.empty:
        return {"observations": 0, "first_date": None, "first_value": None, "latest_date": None, "latest_value": None, "change": None, "percent_change": None, "percent_change_available": False, "historical_percentile": None}
    first, latest = out.iloc[0], out.iloc[-1]
    start_value, end_value = float(first["value"]), float(latest["value"])
    change = end_value - start_value
    pct = ((end_value / start_value) - 1.0) * 100.0 if start_value > 0 else None
    pct_rank = float((out["value"] <= end_value).mean() * 100)
    return {
        "observations": int(len(out)), "first_date": pd.Timestamp(first["date"]).date().isoformat(),
        "first_value": start_value, "latest_date": pd.Timestamp(latest["date"]).date().isoformat(),
        "latest_value": end_value, "change": float(change),
        "percent_change": float(pct) if pct is not None else None,
        "percent_change_available": pct is not None, "historical_percentile": pct_rank,
    }


def distribution_frame(frame: pd.DataFrame, mode: str = "levels", start_date=None, end_date=None) -> pd.DataFrame:
    """Return selected distribution observations; changes use prior loaded observation."""
    normalized = normalize_series(frame)
    if mode == "levels":
        out = normalized.copy()
    elif mode in ("absolute_change", "percentage_change"):
        change_mode = "diff" if mode == "absolute_change" else "pct"
        out = change_series(normalized, window=1, mode=change_mode).rename(columns={"change": "value"})
    else:
        raise ValueError("mode must be 'levels', 'absolute_change', or 'percentage_change'")
    if not out.empty:
        out = _date_filter(out, start_date, end_date)
    return out.reset_index(drop=True)


def distribution_summary(frame: pd.DataFrame, start_date=None, end_date=None) -> dict[str, float | int | None]:
    """Descriptive distribution statistics for observations in the specified period."""
    out = normalize_series(frame)
    if not out.empty:
        out = _date_filter(out, start_date, end_date)
    values = out["value"].astype(float) if not out.empty else pd.Series(dtype=float)
    if values.empty:
        return {"observations": 0, "mean": None, "median": None, "std_dev": None, "min": None, "p05": None, "p25": None, "p75": None, "p95": None, "max": None, "skewness": None, "excess_kurtosis": None}
    skew = values.skew() if values.size >= 3 else np.nan
    kurt = values.kurt() if values.size >= 4 else np.nan
    return {
        "observations": int(values.size), "mean": float(values.mean()), "median": float(values.median()),
        "std_dev": float(values.std(ddof=1)) if values.size > 1 else None,
        "min": float(values.min()), "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)), "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)), "max": float(values.max()),
        "skewness": float(skew) if pd.notna(skew) else None,
        "excess_kurtosis": float(kurt) if pd.notna(kurt) else None,
    }


def distribution_interpretation(summary: dict, mode: str) -> str:
    """Generate cautious plain-language descriptions from descriptive summaries."""
    n = int(summary.get("observations") or 0)
    if n == 0:
        return "No valid observations are available in this period."
    if n < 20:
        caution = f"Only {n} observations are available, so the shape and tail estimates are unstable. "
    else:
        caution = f"The summary uses {n} observations in the selected period. "
    mean, median = summary.get("mean"), summary.get("median")
    skew = summary.get("skewness")
    p05, p95 = summary.get("p05"), summary.get("p95")
    spread = f"The central 90% observed range is approximately {p05:,.5g} to {p95:,.5g}. " if p05 is not None and p95 is not None else ""
    if mean is not None and median is not None:
        if abs(mean - median) <= max(abs(summary.get("std_dev") or 0) * 0.1, 1e-12):
            centre = "The mean and median are close, so the centre is not strongly shifted by a difference between these two summaries. "
        elif mean > median:
            centre = "The mean is above the median; relatively large upper observations may be pulling the average upward. "
        else:
            centre = "The mean is below the median; relatively low observations may be pulling the average downward. "
    else:
        centre = ""
    if skew is not None and skew >= 0.5:
        shape = "The distribution is right-skewed: the upper tail extends farther or is more influential. "
    elif skew is not None and skew <= -0.5:
        shape = "The distribution is left-skewed: the lower tail extends farther or is more influential. "
    elif skew is not None:
        shape = "Skewness is not pronounced by the simple ±0.5 descriptive rule used here. "
    else:
        shape = "There are too few observations to describe skewness reliably. "
    if mode == "levels":
        purpose = "This describes observed levels; for a trending series, the histogram can mostly reflect the trend and is not a return distribution. "
    elif mode == "absolute_change":
        purpose = "This describes absolute step-to-step changes, including their sign and size. "
    else:
        purpose = "This describes step-to-step percentage changes, using only observations with a positive prior value. "
    return caution + purpose + centre + spread + shape + "These are descriptive patterns, not a forecast or a test of whether the series is good or bad."

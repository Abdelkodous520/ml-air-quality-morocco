"""Compare air-quality forecasting models (multi-horizon) for Moroccan cities.

Improvements over v1:
  * hourly grid per city (exact gaps), lags 0..168h, rolling mean/std
  * cyclic encodings (hour, weekday, month, wind direction), weekend, Ramadan
  * optional dust/aerosol features (auto-detected) and extra columns
  * several horizons (default +1h, +6h, +24h)
  * walk-forward validation with a gap between train and test
  * metrics per model, per horizon, per fold, per city + exceedance (peaks)
  * uncertainty: quantile regression (q10/q50/q90) + conformal calibration (CQR)
  * optional leave-one-city-out (sparse-sensor scenario)

Note: the default Open-Meteo/CAMS dataset contains MODELED pollution values,
not station measurements. Interpret scores accordingly.

Examples
--------
python temporal_models_v2.py
python temporal_models_v2.py --target pm10_ug_m3 --horizons 1 24 --loco
python temporal_models_v2.py --future-weather --n-splits 4
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "Data" / "morocco_city_pollution_weather_hourly.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "Data"

WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
]
LAGS = (0, 1, 2, 3, 6, 12, 24, 48, 168)
ROLL_WINDOWS = (6, 24, 168)

# Approximate Ramadan periods (start, end inclusive). Morocco's moon sighting
# can shift these by one day. Extend the list if your data goes beyond 2027.
RAMADAN_PERIODS = [
    ("2019-05-06", "2019-06-04"),
    ("2020-04-24", "2020-05-23"),
    ("2021-04-13", "2021-05-12"),
    ("2022-04-02", "2022-05-01"),
    ("2023-03-23", "2023-04-20"),
    ("2024-03-11", "2024-04-09"),
    ("2025-03-01", "2025-03-30"),
    ("2026-02-18", "2026-03-19"),
    ("2027-02-08", "2027-03-09"),
]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input hourly CSV")
    p.add_argument("--target", default="pm2_5_ug_m3", help="Pollutant column to predict")
    p.add_argument("--group", default="city", help="City/station column")
    p.add_argument("--datetime", default="datetime_local", help="Datetime column")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24], help="Forecast horizons in hours")
    p.add_argument("--test-fraction", type=float, default=0.2, help="Latest time fraction used for testing")
    p.add_argument("--n-splits", type=int, default=3, help="Walk-forward folds inside the test period")
    p.add_argument("--gap-hours", type=int, default=24, help="Minimum gap between train end and test start")
    p.add_argument("--alpha", type=float, default=0.2, help="Interval miscoverage (0.2 -> 80%% interval)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Exceedance threshold (default: 90th percentile of training target)")
    p.add_argument("--extra-features", nargs="*", default=[], help="Extra numeric columns to use as features")
    p.add_argument("--future-weather", action="store_true",
                   help="Add weather at t+h as features (simulates a perfect weather forecast)")
    p.add_argument("--skip-uncertainty", action="store_true", help="Skip quantile / conformal intervals")
    p.add_argument("--loco", action="store_true", help="Run leave-one-city-out evaluation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Folder for outputs")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def detect_extra_features(columns, target: str, user_extra) -> list[str]:
    """User-provided extras (if present) + auto-detected dust/aerosol columns."""
    cols = list(columns)
    extras = []
    for c in user_extra:
        if c in cols:
            extras.append(c)
        else:
            print(f"Warning: extra feature '{c}' not found in CSV; ignored.")
    for c in cols:
        if c != target and any(k in c.lower() for k in ("dust", "aerosol")):
            extras.append(c)
    return list(dict.fromkeys(extras))


def prepare_hourly(raw: pd.DataFrame, args, numeric_cols: list[str]) -> pd.DataFrame:
    """Clean, deduplicate and reindex every group on a complete hourly grid."""
    gcol, tcol, target = args.group, args.datetime, args.target
    data = raw.copy()
    data[tcol] = pd.to_datetime(data[tcol], errors="coerce")
    if getattr(data[tcol].dt, "tz", None) is not None:
        data[tcol] = data[tcol].dt.tz_localize(None)
    data[tcol] = data[tcol].dt.floor("h")
    for c in numeric_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=[tcol, gcol])
    data[gcol] = data[gcol].astype(str)

    # Average duplicate readings for a group/hour.
    data = data[[gcol, tcol] + numeric_cols].groupby([gcol, tcol], as_index=False).mean()

    frames = []
    for g, df in data.groupby(gcol, sort=False):
        df = df.set_index(tcol).sort_index()
        full = pd.date_range(df.index.min(), df.index.max(), freq="h")
        df = df.reindex(full)
        df[gcol] = g
        df.index.name = tcol
        frames.append(df.reset_index())
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values([gcol, tcol]).reset_index(drop=True)

    n_target = int(data[target].notna().sum())
    print(f"Hourly grid: {len(data):,} rows, {n_target:,} with a target value "
          f"({n_target / max(len(data), 1):.1%}).")
    return data


def build_base_features(data: pd.DataFrame, args, extras: list[str]):
    """Features known at time t (no information from the future)."""
    gcol, tcol, target = args.group, args.datetime, args.target
    gt = data.groupby(gcol, sort=False)[target]
    new: dict[str, pd.Series] = {}
    feats: list[str] = []

    for lag in LAGS:
        name = f"{target}_lag_{lag}h"
        new[name] = gt.shift(lag)
        feats.append(name)

    for w in ROLL_WINDOWS:
        roll = gt.rolling(w, min_periods=max(2, int(w * 0.8)))
        new[f"{target}_roll_mean_{w}h"] = roll.mean().reset_index(level=0, drop=True)
        new[f"{target}_roll_std_{w}h"] = roll.std().reset_index(level=0, drop=True)
        feats += [f"{target}_roll_mean_{w}h", f"{target}_roll_std_{w}h"]

    weather = [c for c in WEATHER_FEATURES if c in data.columns]
    future_cols = []
    for c in weather:
        if c == "wind_direction_10m":
            rad = np.deg2rad(data[c])
            new["wind_dir_sin"] = np.sin(rad)
            new["wind_dir_cos"] = np.cos(rad)
            feats += ["wind_dir_sin", "wind_dir_cos"]
            future_cols += ["wind_dir_sin", "wind_dir_cos"]
        else:
            feats.append(c)
            future_cols.append(c)
    for c in extras:
        feats.append(c)
        future_cols.append(c)

    ts = data[tcol]
    for name, values, period in (
        ("hour", ts.dt.hour, 24),
        ("dow", ts.dt.dayofweek, 7),
        ("month", ts.dt.month - 1, 12),
    ):
        new[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        new[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
        feats += [f"{name}_sin", f"{name}_cos"]
    new["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    ramadan = pd.Series(False, index=data.index)
    for start, end in RAMADAN_PERIODS:
        ramadan |= (ts >= pd.Timestamp(start)) & (ts < pd.Timestamp(end) + pd.Timedelta(days=1))
    new["is_ramadan"] = ramadan.astype(int)
    feats += ["is_weekend", "is_ramadan"]

    data = pd.concat([data, pd.DataFrame(new, index=data.index)], axis=1)

    loc = pd.get_dummies(data[gcol], prefix=gcol, dtype=int)
    data = pd.concat([data, loc], axis=1)
    loc_cols = loc.columns.tolist()
    feats += loc_cols

    miss = data[feats].isna().mean()
    bad = miss[miss > 0.2]
    if not bad.empty:
        print("Warning: features with >20% missing values (kept; trees handle NaN, others impute):")
        print(bad.round(3).to_string())
    return data, feats, loc_cols, future_cols


def make_horizon_dataset(data: pd.DataFrame, feats: list[str], future_cols: list[str], h: int, args):
    """Add the target at t+h (and optionally the weather at t+h)."""
    ds = data.copy()
    g = ds.groupby(args.group, sort=False)
    ds["y_target"] = g[args.target].shift(-h)
    f = list(feats)
    if args.future_weather and future_cols:
        fut = {f"{c}_fut": g[c].shift(-h) for c in future_cols}
        ds = pd.concat([ds, pd.DataFrame(fut, index=ds.index)], axis=1)
        f += list(fut)
    lag0 = f"{args.target}_lag_0h"
    before = len(ds)
    ds = ds.dropna(subset=["y_target", lag0]).reset_index(drop=True)
    print(f"[h={h}h] rows kept: {len(ds):,} / {before:,} (dropped only rows without target or current value)")
    return ds, f


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def make_folds(times: pd.Series, test_fraction: float, n_splits: int, gap_hours: int, horizon: int):
    """Walk-forward folds: expanding train window, consecutive test blocks."""
    ut = np.sort(pd.unique(times))
    n = len(ut)
    if n < 20:
        return []
    test_start = max(1, int(n * (1 - test_fraction)))
    edges = np.linspace(test_start, n, n_splits + 1).astype(int)
    gap = pd.Timedelta(hours=max(gap_hours, horizon))
    folds = []
    for i in range(n_splits):
        a, b = edges[i], edges[i + 1]
        if a >= b:
            continue
        start = pd.Timestamp(ut[a])
        end = pd.Timestamp(ut[b]) if b < n else None
        folds.append((i + 1, start - gap, start, end))
    return folds


def window_mask(times: pd.Series, start, end) -> pd.Series:
    mask = times >= start
    if end is not None:
        mask &= times < end
    return mask


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_models(seed: int) -> dict:
    models = {
        "LinearRegression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, objective="reg:squarederror", random_state=seed, n_jobs=-1,
        )
    except ImportError:
        if not getattr(build_models, "_warned", False):
            print("XGBoost not installed; skipping it (pip install xgboost).")
            build_models._warned = True
    return models


def make_quantile_model(q: float, seed: int):
    try:
        import xgboost as xgb

        if int(xgb.__version__.split(".")[0]) >= 2:
            return xgb.XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=q, n_estimators=300, max_depth=5,
                learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=-1,
            )
    except ImportError:
        pass
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=300, learning_rate=0.05, random_state=seed
    )


def fit_quantile_intervals(Xtr, ytr, times_tr, Xte, alpha, gap, seed):
    """Quantile regression + conformalized quantile regression (CQR).

    The last 15% of the training period (chronologically) is held out to
    calibrate the intervals. Coverage guarantees are approximate for time series.
    """
    ut = np.sort(pd.unique(times_tr))
    if len(ut) < 50:
        return None
    cal_start = pd.Timestamp(ut[int(len(ut) * 0.85)])
    fit_mask = (times_tr < cal_start - gap).to_numpy()
    cal_mask = (times_tr >= cal_start).to_numpy()
    if fit_mask.sum() < 100 or cal_mask.sum() < 50:
        return None

    taus = (alpha / 2, 0.5, 1 - alpha / 2)
    cal_preds, test_preds = [], []
    for tau in taus:
        m = make_quantile_model(tau, seed)
        m.fit(Xtr.loc[fit_mask], ytr.loc[fit_mask])
        cal_preds.append(m.predict(Xtr.loc[cal_mask]))
        test_preds.append(m.predict(Xte))
    cal = np.sort(np.vstack(cal_preds), axis=0)    # enforce non-crossing quantiles
    tst = np.sort(np.vstack(test_preds), axis=0)

    y_cal = ytr.loc[cal_mask].to_numpy()
    scores = np.maximum(cal[0] - y_cal, y_cal - cal[2])
    n = len(scores)
    k = min(n, math.ceil((n + 1) * (1 - alpha)))
    qhat = float(np.sort(scores)[k - 1])

    return {
        "q_lo": np.maximum(tst[0], 0),
        "q_50": np.maximum(tst[1], 0),
        "q_hi": np.maximum(tst[2], 0),
        "cqr_lo": np.maximum(tst[0] - qhat, 0),
        "cqr_hi": np.maximum(tst[2] + qhat, 0),
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def score(y, pred) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
    }


def exceed_metrics(y, pred, thr) -> dict:
    yt = np.asarray(y) > np.asarray(thr)
    yp = np.asarray(pred) > np.asarray(thr)
    if yt.sum() == 0:
        return {"precision_exceed": np.nan, "recall_exceed": np.nan, "f1_exceed": np.nan}
    p, r, f, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    return {"precision_exceed": float(p), "recall_exceed": float(r), "f1_exceed": float(f)}


def pinball(y, q, tau) -> float:
    d = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def summarize(all_pred: pd.DataFrame, model_cols: list[str], gcol: str):
    rows, rows_group = [], []
    for h, dfh in all_pred.groupby("horizon_h"):
        parts = [("all", dfh)] + [(str(k), d) for k, d in dfh.groupby("fold")]
        for label, d in parts:
            for m in model_cols:
                rows.append({
                    "horizon_h": h, "fold": label, "model": m, "n": len(d),
                    **score(d["y_true"], d[m]),
                    **exceed_metrics(d["y_true"], d[m], d["threshold"]),
                })
        for grp, d in dfh.groupby(gcol):
            for m in model_cols:
                rows_group.append({
                    "horizon_h": h, gcol: grp, "model": m, "n": len(d),
                    **score(d["y_true"], d[m]),
                    **exceed_metrics(d["y_true"], d[m], d["threshold"]),
                })
    comp = pd.DataFrame(rows)
    pers = comp.loc[comp["model"] == "Persistence", ["horizon_h", "fold", "MAE"]].rename(
        columns={"MAE": "MAE_persistence"})
    comp = comp.merge(pers, on=["horizon_h", "fold"], how="left")
    comp["MAE_skill_vs_persistence"] = 1 - comp["MAE"] / comp["MAE_persistence"]
    comp = comp.drop(columns="MAE_persistence")
    return comp, pd.DataFrame(rows_group)


def summarize_uncertainty(all_pred: pd.DataFrame, gcol: str, alpha: float) -> pd.DataFrame:
    if "q_lo" not in all_pred.columns:
        return pd.DataFrame()
    rows = []
    taus = (alpha / 2, 0.5, 1 - alpha / 2)
    for h, dfh in all_pred.groupby("horizon_h"):
        subsets = [("ALL", dfh)] + [(g, d) for g, d in dfh.groupby(gcol)]
        for label, d in subsets:
            d = d.dropna(subset=["q_lo", "q_hi", "cqr_lo", "cqr_hi"])
            if d.empty:
                continue
            for method, lo, hi in (("quantile_raw", "q_lo", "q_hi"), ("quantile_conformal", "cqr_lo", "cqr_hi")):
                inside = (d["y_true"] >= d[lo]) & (d["y_true"] <= d[hi])
                row = {
                    "horizon_h": h, gcol: label, "method": method, "n": len(d),
                    "target_coverage": 1 - alpha,
                    "empirical_coverage": float(inside.mean()),
                    "mean_width": float((d[hi] - d[lo]).mean()),
                }
                if method == "quantile_raw":
                    row["pinball_mean"] = float(np.mean([
                        pinball(d["y_true"], d["q_lo"], taus[0]),
                        pinball(d["y_true"], d["q_50"], taus[1]),
                        pinball(d["y_true"], d["q_hi"], taus[2]),
                    ]))
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Leave-one-city-out
# --------------------------------------------------------------------------- #
def run_loco(ds, feats, loc_cols, args, h, model_name) -> list[dict]:
    groups = sorted(ds[args.group].unique())
    if len(groups) < 2:
        print("LOCO skipped: needs at least 2 groups.")
        return []
    folds = make_folds(ds[args.datetime], args.test_fraction, 1, args.gap_hours, h)
    if not folds:
        return []
    _, train_end, test_start, test_end = folds[0]
    base = [c for c in feats if c not in loc_cols]
    hist = [c for c in base if c.startswith(f"{args.target}_lag_") or c.startswith(f"{args.target}_roll_")]
    variants = {"with_history": base, "no_history": [c for c in base if c not in hist]}
    lag0 = f"{args.target}_lag_0h"

    train_time = ds[args.datetime] < train_end
    test_time = window_mask(ds[args.datetime], test_start, test_end)
    rows = []
    for g in groups:
        train = ds[train_time & (ds[args.group] != g)]
        test = ds[test_time & (ds[args.group] == g)]
        if train.empty or test.empty:
            continue
        rows.append({"horizon_h": h, "held_out": g, "variant": "reference", "model": "Persistence",
                     "n_test": len(test), **score(test["y_target"], test[lag0])})
        for vname, cols in variants.items():
            if not cols:
                continue
            model = build_models(args.seed)[model_name]
            model.fit(train[cols], train["y_target"])
            pred = np.maximum(model.predict(test[cols]), 0)
            rows.append({"horizon_h": h, "held_out": g, "variant": vname, "model": model_name,
                         "n_test": len(test), **score(test["y_target"], pred)})
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    if not 0.05 <= args.test_fraction <= 0.5:
        raise ValueError("--test-fraction must be between 0.05 and 0.5")
    if not 0.01 <= args.alpha <= 0.5:
        raise ValueError("--alpha must be between 0.01 and 0.5")
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    raw = pd.read_csv(args.input, low_memory=False)
    missing = {args.group, args.datetime, args.target} - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {', '.join(sorted(missing))}")

    extras = detect_extra_features(raw.columns, args.target, args.extra_features)
    if extras:
        print(f"Extra features used: {extras}")
    weather = [c for c in WEATHER_FEATURES if c in raw.columns]
    numeric_cols = list(dict.fromkeys([args.target] + weather + extras))

    data = prepare_hourly(raw, args, numeric_cols)
    data, feats, loc_cols, future_cols = build_base_features(data, args, extras)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lag0 = f"{args.target}_lag_0h"
    all_frames, loco_rows = [], []
    model_names: list[str] = []

    for h in args.horizons:
        ds, fcols = make_horizon_dataset(data, feats, future_cols, h, args)
        folds = make_folds(ds[args.datetime], args.test_fraction, args.n_splits, args.gap_hours, h)
        if not folds:
            print(f"[h={h}h] not enough timestamps; skipped.")
            continue

        for fold_id, train_end, test_start, test_end in folds:
            train = ds[ds[args.datetime] < train_end]
            test = ds[window_mask(ds[args.datetime], test_start, test_end)]
            if len(train) < 200 or test.empty:
                print(f"[h={h}h fold {fold_id}] too little data; skipped.")
                continue
            print(f"[h={h}h fold {fold_id}] train={len(train):,} test={len(test):,} "
                  f"(test starts {test_start})")

            Xtr, ytr = train[fcols], train["y_target"]
            Xte, yte = test[fcols], test["y_target"]

            pred = test[[args.group, args.datetime]].copy()
            pred["horizon_h"] = h
            pred["fold"] = fold_id
            pred["y_true"] = yte.to_numpy()
            pred["threshold"] = args.threshold if args.threshold is not None else float(ytr.quantile(0.9))
            pred["Persistence"] = Xte[lag0].to_numpy()

            models = build_models(args.seed)
            model_names = list(models)
            for name, model in models.items():
                model.fit(Xtr, ytr)
                pred[name] = np.maximum(model.predict(Xte), 0)

            if not args.skip_uncertainty:
                res = fit_quantile_intervals(
                    Xtr, ytr, train[args.datetime], Xte, args.alpha,
                    pd.Timedelta(hours=max(args.gap_hours, h)), args.seed,
                )
                if res is None:
                    print(f"[h={h}h fold {fold_id}] not enough data for calibrated intervals.")
                else:
                    for k, v in res.items():
                        pred[k] = v
            all_frames.append(pred)

        if args.loco:
            preferred = "XGBoost" if "XGBoost" in build_models(args.seed) else "RandomForest"
            loco_rows += run_loco(ds, fcols, loc_cols, args, h, preferred)

    if not all_frames:
        raise RuntimeError("No predictions were produced. Check data size, horizons and test fraction.")

    all_pred = pd.concat(all_frames, ignore_index=True)
    model_cols = ["Persistence"] + model_names
    comp, by_group = summarize(all_pred, model_cols, args.group)
    unc = summarize_uncertainty(all_pred, args.group, args.alpha)

    out = args.output_dir
    comp.to_csv(out / "temporal_v2_comparison.csv", index=False)
    by_group.to_csv(out / "temporal_v2_by_group.csv", index=False)
    all_pred.to_csv(out / "temporal_v2_predictions.csv", index=False)
    if not unc.empty:
        unc.to_csv(out / "temporal_v2_uncertainty.csv", index=False)
    if loco_rows:
        pd.DataFrame(loco_rows).to_csv(out / "temporal_v2_loco.csv", index=False)

    fmt = lambda v: f"{v:.3f}"
    cols = ["horizon_h", "model", "MAE", "RMSE", "R2", "MAE_skill_vs_persistence", "f1_exceed"]
    print("\n=== Scores (all folds pooled; skill > 0 means better than persistence) ===")
    print(comp[comp["fold"] == "all"][cols].sort_values(["horizon_h", "RMSE"]).to_string(index=False, float_format=fmt))

    if not unc.empty:
        print(f"\n=== Intervals (target coverage {1 - args.alpha:.0%}) ===")
        u = unc[unc[args.group] == "ALL"][["horizon_h", "method", "empirical_coverage", "mean_width"]]
        print(u.to_string(index=False, float_format=fmt))

    if loco_rows:
        print("\n=== Leave-one-city-out ===")
        print(pd.DataFrame(loco_rows).to_string(index=False, float_format=fmt))

    print(f"\nSaved outputs in: {out}")
    print("Caution: verify whether the target is measured or model-estimated before interpreting results.")


if __name__ == "__main__":
    main()
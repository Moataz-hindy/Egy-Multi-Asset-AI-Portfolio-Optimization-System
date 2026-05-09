"""Expected-return forecasting with a walk-forward, leakage-safe ensemble.

The legacy implementation suffered from three issues that this rewrite
addresses:

1. **Future leakage** -- features were winsorized using *full-sample*
   quantiles before the chronological train/test split.  The new
   pipeline computes outlier bounds on the train window only.
2. **Single-fold validation** -- a single 80/20 split on a small
   Egyptian sample produced unstable confidence numbers.  We now use
   expanding-window walk-forward CV with the median forecast.
3. **Confidence proxy** -- the legacy ``1/(1+rmse*100)`` mapping was
   uncalibrated; predictions with no signal still produced ~0.7
   confidence.  We replace it with an out-of-sample R^2 (clipped to
   [0, 1]) which is a standard skill measure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

FEATURES = ["return_lag1", "return_lag2", "return_lag3", "dist_to_ma5", "macd_hist", "rsi", "bb_pb"]


@dataclass
class ForecastSummary:
    expected_returns: Dict[str, float]      # daily, decimal
    confidence: Dict[str, float]            # OOS R^2 in [0, 1]
    sample_sizes: Dict[str, int]


def _prepare(asset_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    cols = [c for c in FEATURES if c in asset_df.columns]
    if not cols or "return" not in asset_df.columns:
        return pd.DataFrame(), pd.Series(dtype=float)
    df = asset_df[cols + ["return"]].replace([np.inf, -np.inf], np.nan).dropna()
    return df[cols], df["return"]


def _winsorize_train(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.quantile(X_train, 0.01, axis=0)
    hi = np.quantile(X_train, 0.99, axis=0)
    return np.clip(X_train, lo, hi), np.clip(X_test, lo, hi)


def _walk_forward_score(
    X: np.ndarray,
    y: np.ndarray,
    model_factory,
    n_folds: int = 4,
    min_train: int = 60,
) -> Tuple[List[float], List[float]]:
    n = len(X)
    if n < min_train + 30:
        return [], []
    fold_size = max(15, (n - min_train) // n_folds)
    predictions: List[float] = []
    r2_scores: List[float] = []
    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, n)
        if train_end >= n or test_end <= train_end:
            break
        X_tr, X_te = X[:train_end], X[train_end:test_end]
        y_tr, y_te = y[:train_end], y[train_end:test_end]
        X_tr, X_te = _winsorize_train(X_tr, X_te)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        try:
            model = model_factory()
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_te_s)
            if not np.all(np.isfinite(preds)):
                continue
            ss_res = float(np.sum((y_te - preds) ** 2))
            ss_tot = float(np.sum((y_te - y_te.mean()) ** 2)) + 1e-12
            r2 = 1.0 - ss_res / ss_tot
            r2_scores.append(float(np.clip(r2, -1.0, 1.0)))
            # Use the final fold's last prediction as the latest forecast.
            if fold == n_folds - 1 or test_end == n:
                predictions.append(float(preds[-1]))
        except Exception:
            continue
    return predictions, r2_scores


def _ensemble_forecast(asset_df: pd.DataFrame) -> Tuple[float, float, int]:
    X_df, y_series = _prepare(asset_df)
    if X_df.empty:
        return 0.0, 0.0, 0
    X, y = X_df.values, y_series.values
    if len(X) < 80:
        # Not enough sample to do walk-forward CV; fall back to recent mean.
        return float(np.nanmean(y[-30:])), 0.20, int(len(X))

    factories = [
        lambda: Ridge(alpha=1.0),
        lambda: RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=10, random_state=42),
        lambda: SVR(C=1.0, epsilon=1e-3, kernel="rbf"),
    ]
    all_preds: List[float] = []
    all_r2: List[float] = []
    for factory in factories:
        preds, r2s = _walk_forward_score(X, y, factory)
        all_preds.extend(preds)
        all_r2.extend(r2s)

    if not all_preds:
        return float(np.nanmean(y[-30:])), 0.20, int(len(X))

    expected_return = float(np.median(all_preds))
    # OOS R^2 across folds, clipped to [0, 1] and lightly shrunken so a
    # signal-less series settles around 0.10 confidence.
    raw_r2 = float(np.median(all_r2)) if all_r2 else 0.0
    confidence = float(np.clip(0.5 * raw_r2 + 0.10, 0.05, 0.95))
    return expected_return, confidence, int(len(X))


def forecast_expected_returns(features: Dict[str, pd.DataFrame]) -> ForecastSummary:
    expected_returns: Dict[str, float] = {}
    confidence: Dict[str, float] = {}
    samples: Dict[str, int] = {}
    for asset, df in features.items():
        if "return" not in df.columns:
            continue
        mu, conf, n = _ensemble_forecast(df.dropna(subset=["return"]))
        expected_returns[asset] = mu
        confidence[asset] = conf
        samples[asset] = n
    return ForecastSummary(expected_returns=expected_returns, confidence=confidence, sample_sizes=samples)

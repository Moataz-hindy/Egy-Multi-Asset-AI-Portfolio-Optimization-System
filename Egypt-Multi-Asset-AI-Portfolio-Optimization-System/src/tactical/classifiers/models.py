from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def _dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Use tactical lag/technical columns for short-horizon classification.
    use_cols = [c for c in ["return_lag1", "return_lag2", "return_lag3", "macd_hist", "rsi", "bb_pb"] if c in df.columns]
    X = df[use_cols].replace([np.inf, -np.inf], np.nan).dropna()
    y = _build_target(df.loc[X.index].copy())
    aligned_idx = X.index.intersection(y.index)
    return X.loc[aligned_idx].values, y.loc[aligned_idx].values


def _build_target(df: pd.DataFrame) -> pd.Series:
    if "label" not in df.columns:
        return pd.Series(dtype=int)
    y = df["label"].astype(int).copy()
    unique = sorted(y.dropna().unique().tolist())
    if len(unique) >= 3:
        return y
    # Build leakage-safe 3-class tactical labels from next-day return quantiles.
    if "return" in df.columns:
        nxt = pd.to_numeric(df["return"], errors="coerce").shift(-1)
        lo, hi = float(nxt.quantile(0.33)), float(nxt.quantile(0.67))
        tri = pd.Series(0, index=df.index, dtype=int)
        tri[nxt <= lo] = -1
        tri[nxt >= hi] = 1
        tri = tri.dropna()
        if len(tri.unique()) >= 3:
            return tri.astype(int)
    return y


def _time_folds(n: int, n_splits: int = 5, min_train: int = 120, min_test: int = 30) -> list[tuple[int, int, int, int]]:
    folds: list[tuple[int, int, int, int]] = []
    if n < min_train + min_test:
        return folds
    test_size = max(min_test, (n - min_train) // n_splits)
    for i in range(n_splits):
        tr_end = min_train + i * test_size
        te_end = min(n, tr_end + test_size)
        if te_end - tr_end < min_test or tr_end < min_train:
            continue
        folds.append((0, tr_end, tr_end, te_end))
    return folds


def _clip_features(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.nanquantile(train, 0.01, axis=0)
    hi = np.nanquantile(train, 0.99, axis=0)
    return np.clip(train, lo, hi), np.clip(test, lo, hi)


def classify_tactical_signal(df: pd.DataFrame) -> Dict[str, object]:
    # Prepare matrix for tactical buy/hold/sell classifier stack.
    X, y = _dataset(df)
    if len(X) < 180:
        # Conservative neutral fallback on limited sample history.
        return {
            "signal": 0,
            "confidence": 0.3,
            "evaluation": {
                "status": "insufficient_history",
                "holdout_size": 0,
            },
        }

    # Use last point as the live prediction target; all diagnostics are OOS on history.
    X_hist, y_hist = X[:-1], y[:-1]
    X_live = X[-1:]
    if len(X_hist) < 150:
        return {"signal": 0, "confidence": 0.3, "evaluation": {"status": "insufficient_history", "holdout_size": 0}}

    classes = sorted(np.unique(y_hist))
    map_to = {c: i for i, c in enumerate(classes)}
    map_back = {i: c for c, i in map_to.items()}
    y_hist_e = np.array([map_to[v] for v in y_hist])

    bc = np.bincount(y_hist_e, minlength=len(classes))
    min_class = int(bc.min()) if bc.size else 0
    min_required = max(12, int(0.02 * len(y_hist_e)))
    if min_class < min_required:
        return {
            "signal": 0,
            "confidence": 0.28,
            "evaluation": {
                "status": "class_imbalance",
                "min_class_count": min_class,
                "min_required": min_required,
            },
        }

    oof_probabilities = []
    latest_probabilities = []
    model_metrics = []
    f1_scores = []
    fold_diagnostics = []
    folds = _time_folds(len(X_hist), n_splits=5, min_train=120, min_test=30)
    if not folds:
        return {"signal": 0, "confidence": 0.3, "evaluation": {"status": "insufficient_history", "holdout_size": 0}}

    # Model 1: logistic classifier baseline.
    models = [
        ("logistic_regression", LogisticRegression(max_iter=1200, class_weight="balanced")),
        ("svm_rbf", SVC(kernel="rbf", C=1.2, probability=True, class_weight="balanced")),
    ]
    if XGBClassifier is not None:
        models.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="multi:softprob" if len(classes) > 2 else "binary:logistic",
                    eval_metric="mlogloss" if len(classes) > 2 else "logloss",
                    random_state=42,
                    verbosity=0,
                ),
            )
        )

    for model_name, model in models:
        oof_pred = []
        oof_true = []
        oof_proba_rows = []
        for fold_id, (_, tr_end, te_start, te_end) in enumerate(folds):
            X_train, y_train = X_hist[:tr_end], y_hist_e[:tr_end]
            X_test, y_test = X_hist[te_start:te_end], y_hist_e[te_start:te_end]
            X_train, X_test = _clip_features(X_train, X_test)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            try:
                model.fit(X_train_s, y_train)
                pred = model.predict(X_test_s)
                prob = model.predict_proba(X_test_s)
            except Exception:
                continue
            oof_pred.extend(pred.tolist())
            oof_true.extend(y_test.tolist())
            oof_proba_rows.extend(prob.tolist())
            precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
                y_test, pred, average="weighted", zero_division=0
            )
            fold_diagnostics.append(
                {
                    "model": model_name,
                    "fold": int(fold_id),
                    "train_end_idx": int(tr_end - 1),
                    "test_size": int(len(y_test)),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "precision_weighted": float(precision_w),
                    "recall_weighted": float(recall_w),
                    "f1_weighted": float(f1_w),
                }
            )

        if not oof_pred:
            continue

        y_true = np.array(oof_true)
        y_pred = np.array(oof_pred)
        prob_arr = np.array(oof_proba_rows)
        oof_probabilities.append(prob_arr)

        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        f1_scores.append(float(f1_w))
        fold_f1_vals = [float(fd["f1_weighted"]) for fd in fold_diagnostics if fd.get("model") == model_name]
        fold_f1_std = float(np.std(fold_f1_vals)) if len(fold_f1_vals) >= 2 else 0.0
        model_metrics.append(
            {
                "model": model_name,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision_weighted": float(precision_w),
                "recall_weighted": float(recall_w),
                "f1_weighted": float(f1_w),
                "fold_f1_std": round(fold_f1_std, 6),
            }
        )
        try:
            scaler_live = StandardScaler()
            X_full, X_last = _clip_features(X_hist, X_live)
            X_full_s = scaler_live.fit_transform(X_full)
            X_live_s = scaler_live.transform(X_last)
            model.fit(X_full_s, y_hist_e)
            latest_probabilities.append(model.predict_proba(X_live_s)[0])
        except Exception:
            continue

    if not oof_probabilities or not latest_probabilities:
        return {
            "signal": 0,
            "confidence": 0.25,
            "evaluation": {
                "status": "model_fit_failed",
                "holdout_size": 0,
            },
        }

    # Ensemble picks class with highest averaged posterior probability.
    min_len = min(prob.shape[0] for prob in oof_probabilities)
    avg_prob_full = np.mean(np.stack([prob[-min_len:] for prob in oof_probabilities], axis=0), axis=0)
    avg_latest_prob = np.mean(np.vstack(latest_probabilities), axis=0)
    ensemble_pred = np.argmax(avg_prob_full, axis=1)
    cls = int(np.argmax(avg_latest_prob))
    signal = int(map_back[cls])
    confidence = float(np.clip(np.mean(f1_scores) * float(np.max(avg_latest_prob)), 0.05, 0.95))

    fold_std_vals = [float(m.get("fold_f1_std", 0.0)) for m in model_metrics if "fold_f1_std" in m]
    max_fold_std = max(fold_std_vals) if fold_std_vals else 0.0
    if max_fold_std > 0.10:
        confidence *= float(np.clip(1.2 - 2.2 * max_fold_std, 0.55, 1.0))

    labels_encoded = list(range(len(classes)))
    labels_original = [int(map_back[idx]) for idx in labels_encoded]
    y_eval = y_hist_e[-min_len:]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_eval,
        ensemble_pred,
        labels=labels_encoded,
        average=None,
        zero_division=0,
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_eval,
        ensemble_pred,
        average="weighted",
        zero_division=0,
    )
    cm = confusion_matrix(y_eval, ensemble_pred, labels=labels_encoded)
    calibration_error = float(np.mean(np.abs(np.max(avg_prob_full, axis=1) - (ensemble_pred == y_eval).astype(float))))
    if calibration_error > 0.28:
        confidence *= 0.84

    live_votes = [int(np.argmax(lp)) for lp in latest_probabilities]
    ensemble_disagreement = len(live_votes) >= 2 and len(set(live_votes)) > 1
    if ensemble_disagreement:
        confidence *= 0.87

    confidence = float(np.clip(confidence, 0.05, 0.95))

    if float(f1_w) < 0.42:
        # Enforce neutral stance when tactical edge is not statistically credible.
        signal = 0
        confidence = min(confidence, 0.35)

    per_class = {
        str(label): {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
        }
        for label, p, r, f in zip(labels_original, precision, recall, f1)
    }

    return {
        "signal": signal,
        "confidence": confidence,
        "evaluation": {
            "status": "ok",
            "labels": labels_original,
            "holdout_size": int(len(y_eval)),
            "accuracy": float(accuracy_score(y_eval, ensemble_pred)),
            "precision_weighted": float(precision_w),
            "recall_weighted": float(recall_w),
            "f1_weighted": float(f1_w),
            "confidence_calibration_error": calibration_error,
            "fold_f1_std_max": round(float(max_fold_std), 6),
            "ensemble_disagreement": ensemble_disagreement,
            "confusion_matrix": cm.astype(int).tolist(),
            "per_class": per_class,
            "model_metrics": model_metrics,
            "fold_metrics": fold_diagnostics,
        },
    }

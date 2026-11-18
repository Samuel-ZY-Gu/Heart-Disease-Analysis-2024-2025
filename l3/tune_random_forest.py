"""Reproduce L2 and tune RF on 2024 only. All generated files stay under l3/rf_search."""
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "l3" / "rf_search"
FEATURES = [
    "_AGEG5YR", "_BMI5", "_SMOKER3", "_TOTINDA", "DIABETE4", "_EDUCAG",
    "_INCOMG1", "_RFHLTH", "_LTASTH1", "_DRDXAR2", "_RFBING6", "_AIDTST4", "_STATE",
]
JOBS = 4
# Same preprocessing, cohort, labels, folds and unweighted scoring as the root L2 script.
PREPROCESSOR = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]), ["_BMI5"]),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), [column for column in FEATURES if column != "_BMI5"]),
])
BASELINE = dict(n_estimators=60, max_depth=18, min_samples_leaf=10)
SPACE = {
    "max_depth": [8, 12, 18, 26, None],
    "min_samples_leaf": [1, 5, 10, 25, 50, 100],
    "max_features": ["sqrt", 0.3, 0.7],
    "class_weight": [None, "balanced_subsample"],
    "max_samples": [None, 0.7],
}


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, probability, threshold):
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "AP": average_precision_score(y, probability),
        "ROC_AUC": roc_auc_score(y, probability),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def evaluate(identifier, estimator, folds, y):
    """Cache by run signature and candidate parameters, with one file per completed candidate."""
    path = OUT / (identifier + ".joblib")
    params = estimator.get_params()
    if path.exists():
        saved = joblib.load(path)
        if saved["params"] != params:
            raise ValueError("Candidate parameters changed. Use a separate output directory.")
        print(identifier, "loaded completed candidate", flush=True)
        return saved
    started = time.monotonic()
    probability = np.empty(len(y))
    fold_scores = []
    for fold, (train, validation, x_train, x_validation) in enumerate(folds, 1):
        fitted = clone(estimator).fit(x_train, y.iloc[train])
        p = fitted.predict_proba(x_validation)[:, 1]
        probability[validation] = p
        fold_scores.append({
            "fold": fold, "AP": average_precision_score(y.iloc[validation], p),
            "ROC_AUC": roc_auc_score(y.iloc[validation], p),
        })
        print(identifier, "fold", fold, "AP", round(fold_scores[-1]["AP"], 6),
              "elapsed_s", round(time.monotonic() - started), flush=True)
    saved = {
        "id": identifier, "params": params, "probability": probability,
        "scores": metrics(y, probability, 0.15), "fold_scores": fold_scores,
        "seconds": time.monotonic() - started,
    }
    temporary = path.with_suffix(".partial")
    joblib.dump(saved, temporary)
    temporary.replace(path)
    return saved


def save_ranking(candidates):
    rows = []
    for item in candidates:
        rows.append({
            "candidate": item["id"], **item["scores"],
            "fold_AP_mean": np.mean([f["AP"] for f in item["fold_scores"]]),
            "fold_AP_std": np.std([f["AP"] for f in item["fold_scores"]]),
            "seconds": item["seconds"], "parameters": json.dumps(item["params"], sort_keys=True),
        })
    table = pd.DataFrame(rows).sort_values(["AP", "ROC_AUC"], ascending=False)
    table.to_csv(OUT / "cv_results_2024.csv", index=False)
    return table


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT / "data" / f"Brfss_{year}_clean.CSV" for year in [2024, 2025]]
    signature = {
        "inputs": {p.name: fingerprint(p) for p in inputs},
        "script_sha256": fingerprint(Path(__file__)),
        "sklearn": sklearn.__version__, "python": platform.python_version(),
    }
    signature_path = OUT / "run_signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Inputs, code or environment changed. Preserve this run and use a new output directory.")
    signature_path.write_text(json.dumps(signature, indent=2))

    a, b = [pd.read_csv(path) for path in inputs]
    states = set(a["_STATE"].dropna()) & set(b["_STATE"].dropna())
    a = a.loc[a["_STATE"].isin(states) & a["_MICHD"].isin([1, 2])]
    b = b.loc[b["_STATE"].isin(states) & b["_MICHD"].isin([1, 2])]
    x, y = a[FEATURES], a["_MICHD"].map({1: 1, 2: 0})
    print("Training rows", len(a), "test rows", len(b), "common areas", len(states), flush=True)

    # Fit preprocessing separately inside each fold, then reuse its matrices across candidates.
    folds = []
    for train, validation in StratifiedKFold(3, shuffle=True, random_state=42).split(x, y):
        preprocessing = clone(PREPROCESSOR)
        folds.append((train, validation, preprocessing.fit_transform(x.iloc[train]),
                      preprocessing.transform(x.iloc[validation])))

    lr = evaluate("logistic_baseline", LogisticRegression(max_iter=1000), folds, y)
    forest = RandomForestClassifier(random_state=42, n_jobs=JOBS)
    candidates = [evaluate("rf_original", clone(forest).set_params(**BASELINE), folds, y)]
    save_ranking(candidates)

    sampled = list(ParameterSampler(SPACE, n_iter=24, random_state=42))
    for number, params in enumerate(sampled, 1):
        estimator = clone(forest).set_params(n_estimators=120, **params)
        candidates.append(evaluate(f"rf_search_{number:02d}", estimator, folds, y))
        ranking = save_ranking(candidates)
        print("Completed", len(candidates), "best AP", ranking.iloc[0]["AP"], flush=True)

    top = sorted(candidates, key=lambda c: (c["scores"]["AP"], c["scores"]["ROC_AUC"]),
                 reverse=True)[:3]
    for number, item in enumerate(top, 1):
        estimator = RandomForestClassifier(**{**item["params"], "n_estimators": 400})
        candidates.append(evaluate(f"rf_refine_{number:02d}", estimator, folds, y))
        save_ranking(candidates)

    best = max(candidates, key=lambda c: (c["scores"]["AP"], c["scores"]["ROC_AUC"]))
    # Match L2 LR's 2024 OOF recall using the highest threshold that reaches it.
    target_recall = lr["scores"]["recall"]
    _, recalls, thresholds = precision_recall_curve(y, best["probability"])
    threshold = float(thresholds[recalls[:-1] >= target_recall].max())
    pd.DataFrame({
        "source_row": a.index, "target": y.to_numpy(),
        "logistic_probability": lr["probability"],
        "best_rf_probability": best["probability"],
    }).to_csv(OUT / "oof_predictions_2024.csv", index=False)
    threshold_rows = [
        {"model": "Logistic Regression", "rule": "L2 fixed 0.15", **lr["scores"]},
        {"model": best["id"], "rule": "fixed 0.15", **best["scores"]},
        {"model": best["id"], "rule": "match LR 2024 OOF recall",
         **metrics(y, best["probability"], threshold)},
    ]
    pd.DataFrame(threshold_rows).to_csv(OUT / "threshold_selection_2024.csv", index=False)
    pd.DataFrame([{"candidate": c["id"], **f} for c in [lr, *candidates]
                  for f in c["fold_scores"]]).to_csv(OUT / "fold_scores_2024.csv", index=False)

    selection = {
        "objective": "Maximize pooled 2024 OOF average precision, ROC-AUC breaks ties",
        "best_candidate": best["id"], "best_parameters": best["params"],
        "rf_candidates_evaluated": len(candidates), "matched_recall_threshold": threshold,
        "target_lr_oof_recall": target_recall, "cv_best_rf": best["scores"],
        "cv_lr": lr["scores"],
        "limitations": [
            "Best within the recorded search, not a global optimum.",
            "Search CV is used for selection; it is not a nested-CV performance estimate.",
            "2025 was already inspected in the original project and is not a pristine holdout.",
            "Unweighted sample evaluation, as in L2.",
        ],
    }
    # Freeze selection before evaluating any 2025 labels.
    (OUT / "selection_2024.json").write_text(json.dumps(selection, indent=2))
    fitted = Pipeline([
        ("preprocessing", clone(PREPROCESSOR)),
        ("model", RandomForestClassifier(**best["params"])),
    ]).fit(x, y)
    joblib.dump(fitted, OUT / "best_random_forest.joblib")
    logistic = Pipeline([
        ("preprocessing", clone(PREPROCESSOR)),
        ("model", LogisticRegression(max_iter=1000)),
    ]).fit(x, y)
    test_y = b["_MICHD"].eq(1)
    rf_probability = fitted.predict_proba(b[FEATURES])[:, 1]
    lr_probability = logistic.predict_proba(b[FEATURES])[:, 1]
    test_rows = [
        {"model": "Logistic Regression", "rule": "L2 fixed 0.15",
         **metrics(test_y, lr_probability, 0.15)},
        {"model": "Selected Random Forest", "rule": "fixed 0.15",
         **metrics(test_y, rf_probability, 0.15)},
        {"model": "Selected Random Forest", "rule": "match LR 2024 OOF recall",
         **metrics(test_y, rf_probability, threshold)},
    ]
    pd.DataFrame(test_rows).to_csv(OUT / "evaluation_2025.csv", index=False)
    pd.DataFrame({
        "source_row": b.index, "target": test_y.astype(int).to_numpy(),
        "logistic_probability": lr_probability, "best_rf_probability": rf_probability,
    }).to_csv(OUT / "test_predictions_2025.csv", index=False)
    print("COMPLETE", json.dumps(selection), flush=True)
    print(pd.DataFrame(test_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

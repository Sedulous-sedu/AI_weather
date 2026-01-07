import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve
)

# Optional but recommended
try:
    import psutil
except ImportError:
    psutil = None

try:
    import shap
except ImportError as e:
    raise SystemExit("Missing package: shap. Install with: pip install shap") from e

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit("Missing package: xgboost. Install with: pip install xgboost") from e


CSV_PATH = "C:/Users/abina/Downloads/AI_weather/AI_weather 2/AI_weather/rakta_trips_large.csv"   # <-- change if needed
OUT_DIR = "paper_assets"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42


def parse_time_to_minutes(x):
    # expects "HH:MM" or "HH:MM:SS"
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    parts = s.split(":")
    if len(parts) < 2:
        return np.nan
    h = int(parts[0])
    m = int(parts[1])
    return 60 * h + m


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # derive is_weekend and is_rush_hour from time fields
    dow = df["day_of_week"].astype(str).str.lower()
    df["is_weekend"] = dow.isin(["sat", "saturday", "sun", "sunday"]).astype(int)

    # try to use departure_time; fallback to time_of_day if already binned
    dep_min = df["departure_time"].apply(parse_time_to_minutes)
    df["departure_minutes"] = dep_min

    # Rush hour heuristic: 07:00–09:30 and 16:30–19:00
    df["is_rush_hour"] = (
        ((dep_min >= 7 * 60) & (dep_min <= 9 * 60 + 30)) |
        ((dep_min >= 16 * 60 + 30) & (dep_min <= 19 * 60))
    ).fillna(False).astype(int)

    return df


def get_feature_names(preprocessor: ColumnTransformer):
    # Works for sklearn >= 1.0
    return preprocessor.get_feature_names_out()


def plot_lr_coefficients(feature_names, coefs, top_k=20, out_path="LR_Coefficients.png"):
    # Sort by absolute value
    idx = np.argsort(np.abs(coefs))[::-1][:top_k]
    names = np.array(feature_names)[idx]
    vals = coefs[idx]

    plt.figure(figsize=(8, 6))
    y = np.arange(len(idx))
    plt.barh(y, vals)
    plt.yticks(y, names, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("Coefficient Weight")
    plt.title("Logistic Regression Feature Impact (Top |weights|)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pr_curve(y_true, y_score, title, out_path):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6.5, 5))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} (AUPRC={ap:.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    df = pd.read_csv(CSV_PATH)

    # Ensure expected columns exist
    expected_cols = [
        "date","day_of_week","time_of_day","route","direction","trip_no",
        "departure_time","destination","stop_distance_km","traffic_condition",
        "weather","special_event","temperature_celsius","delay_minutes","arrival_status"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = add_time_features(df)

    # Binary label: Late=1, On-Time=0
    y = (df["arrival_status"].astype(str).str.lower().str.contains("late")).astype(int)

    # Features (drop targets)
    X = df.drop(columns=["arrival_status", "delay_minutes"])

    # Pick categorical/numeric columns
    categorical_cols = [
        "day_of_week","time_of_day","route","direction","trip_no",
        "departure_time","destination","traffic_condition","weather","special_event","date"
    ]
    # You may drop "date" if you think it leaks; keep for now because you included it in schema
    numeric_cols = ["stop_distance_km","temperature_celsius","departure_minutes","is_weekend","is_rush_hour"]

    # Basic cleaning
    for c in categorical_cols:
        X[c] = X[c].astype(str).fillna("NA")

    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", Pipeline([("sc", StandardScaler())]), numeric_cols),
        ],
        remainder="drop"
    )

    # Class imbalance helper for XGBoost
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
        solver="lbfgs"
    )

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    lr_pipe = Pipeline([("pre", pre), ("model", lr)])
    xgb_pipe = Pipeline([("pre", pre), ("model", xgb)])

    # --- Train ---
    lr_pipe.fit(X_train, y_train)
    xgb_pipe.fit(X_train, y_train)

    # --- Predict probs ---
    lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
    xgb_proba = xgb_pipe.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    def summarize(name, proba):
        pred = (proba >= 0.5).astype(int)
        return {
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "auprc": float(average_precision_score(y_test, proba)),
        }

    metrics = [summarize("LogisticRegression", lr_proba), summarize("XGBoost", xgb_proba)]

    # --- Inference timing (per-sample) ---
    # Use transformed matrix once for fair timing
    X_test_tx = lr_pipe.named_steps["pre"].transform(X_test)

    def time_predict_proba(model, X_tx, repeats=5):
        # model is the estimator (already fitted)
        # returns ms/sample
        total = 0.0
        n = X_tx.shape[0]
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model.predict_proba(X_tx)
            total += (time.perf_counter() - t0)
        return (total / repeats) * 1000.0 / max(n, 1)

    lr_ms = time_predict_proba(lr_pipe.named_steps["model"], X_test_tx)
    xgb_ms = time_predict_proba(xgb_pipe.named_steps["model"], X_test_tx)

    footprint = {}
    if psutil is not None:
        p = psutil.Process(os.getpid())
        footprint["rss_mb"] = float(p.memory_info().rss / (1024**2))

    summary = {
        "metrics": metrics,
        "inference_ms_per_sample": {
            "LogisticRegression": float(lr_ms),
            "XGBoost": float(xgb_ms)
        },
        "process_footprint": footprint
    }

    with open(os.path.join(OUT_DIR, "model_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- PR Curves ---
    plot_pr_curve(y_test, lr_proba, "Precision–Recall (Late class) — Logistic Regression",
                  os.path.join(OUT_DIR, "PR_Curve_LR.png"))
    plot_pr_curve(y_test, xgb_proba, "Precision–Recall (Late class) — XGBoost",
                  os.path.join(OUT_DIR, "PR_Curve_XGB.png"))

    # --- LR Coefficients plot ---
    feat_names = get_feature_names(lr_pipe.named_steps["pre"])
    lr_coefs = lr_pipe.named_steps["model"].coef_[0]
    plot_lr_coefficients(
        feat_names, lr_coefs, top_k=25,
        out_path=os.path.join(OUT_DIR, "LR_Coefficients.png")
    )

    # --- SHAP Summary for XGBoost ---
    # SHAP works on the transformed feature matrix
    X_test_tx_dense = X_test_tx.toarray() if hasattr(X_test_tx, "toarray") else X_test_tx
    explainer = shap.TreeExplainer(xgb_pipe.named_steps["model"])
    shap_values = explainer.shap_values(X_test_tx_dense)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test_tx_dense,
        feature_names=feat_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "SHAP_Summary.png"), dpi=300)
    plt.close()

    print("Done. Assets written to:", OUT_DIR)
    print("Key numbers to paste into paper:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

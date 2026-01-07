import os
import time
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    PrecisionRecallDisplay,
    average_precision_score,
)

from xgboost import XGBClassifier
import shap

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = r"C:/Users/abina/Downloads/AI_weather/AI_weather 2/AI_weather/rakta_trips_large.csv"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
LATE_THRESHOLD_MINUTES = 5.0  # ✅ Late if delay_minutes > 5

# Split strategy (choose one):
#   "group" : avoids identical timetable templates in both train/test
#   "time"  : train on earlier dates, test on later dates (if 'date' exists)
#   "random": classic stratified random split
SPLIT_MODE = "group"
TEST_SIZE = 0.2

# ============================================================
# Helpers
# ============================================================

def clean_cat(series: pd.Series, missing_token: str) -> pd.Series:
    """Normalize categorical strings and replace missing-ish tokens."""
    s = series.copy()
    s = s.replace({np.nan: missing_token})
    s = s.astype(str).str.strip()
    s = s.replace({"nan": missing_token, "NaN": missing_token, "None": missing_token, "": missing_token})
    return s


def pick_existing(cols: List[str], df_cols) -> List[str]:
    return [c for c in cols if c in df_cols]


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# ============================================================
# Load
# ============================================================
df = pd.read_csv(CSV_PATH).copy()

# Core columns used for modeling (avoid leakage: do NOT include delay_minutes)
feature_cols = [
    "day_of_week",
    "time_of_day",
    "route",
    "direction",
    "trip_no",
    "destination",
    "stop_distance_km",
    "traffic_condition",
    "weather",
    "special_event",
    "temperature_celsius",
]

# Clean categorical columns (use NoEvent ONLY for special_event)
cat_cols = [
    "day_of_week",
    "time_of_day",
    "route",
    "direction",
    "trip_no",
    "destination",
    "traffic_condition",
    "weather",
    "special_event",
]
num_cols = ["stop_distance_km", "temperature_celsius"]

# --- label source (FIX) ---
# Prefer a deterministic label from delay_minutes > 5.
# If delay_minutes doesn't exist, fallback to arrival_status.
if "delay_minutes" in df.columns:
    df["delay_minutes"] = safe_numeric(df["delay_minutes"])
    df = df.dropna(subset=["delay_minutes"]).copy()
    y = (df["delay_minutes"] > LATE_THRESHOLD_MINUTES).astype(int)
    # For reporting only (optional):
    df["arrival_status"] = np.where(y == 1, "Late", "On-Time")
elif "arrival_status" in df.columns:
    df = df.dropna(subset=["arrival_status"]).copy()
    arrival = df["arrival_status"].astype(str).str.strip().str.lower()
    y = (arrival == "late").astype(int)
else:
    raise ValueError("CSV must contain either 'delay_minutes' or 'arrival_status' to define Late/On-Time labels.")

# Ensure required feature columns exist
missing_feats = [c for c in feature_cols if c not in df.columns]
if missing_feats:
    raise ValueError(f"Missing required feature columns: {missing_feats}")

# Clean / normalize features
for c in pick_existing(cat_cols, df.columns):
    if c == "special_event":
        df[c] = clean_cat(df[c], missing_token="NoEvent")
    else:
        df[c] = clean_cat(df[c], missing_token="Unknown")

X = df[feature_cols].copy()

# Numeric cleanup
for c in num_cols:
    X[c] = safe_numeric(X[c])
    X[c] = X[c].fillna(X[c].median())

# Categorical cleanup (keep consistent)
for c in cat_cols:
    if c == "special_event":
        X[c] = clean_cat(X[c], missing_token="NoEvent")
    else:
        X[c] = clean_cat(X[c], missing_token="Unknown")

# ============================================================
# IMPORTANT: Deduplicate to reduce leakage
# Drops exact duplicates of (features + label).
# ============================================================
before = len(X)
deduped = pd.concat([X, y.rename("y")], axis=1).drop_duplicates(subset=feature_cols + ["y"], keep="first")
X = deduped[feature_cols]
y = deduped["y"].astype(int)
print(f"Rows before dedup: {before:,} | after dedup: {len(deduped):,}")

# ============================================================
# Split
# ============================================================
if SPLIT_MODE == "time" and "date" in df.columns:
    # Time-based split using the original df date column aligned by index
    tmp = df.loc[X.index].copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp.dropna(subset=["date"]).sort_values("date")

    cutoff_idx = int((1 - TEST_SIZE) * len(tmp))
    train_idx = tmp.index[:cutoff_idx]
    test_idx = tmp.index[cutoff_idx:]

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

elif SPLIT_MODE == "group":
    # Group split to avoid the same timetable template in both train/test.
    # Use timetable-ish columns if present.
    tmp = df.loc[X.index].copy()
    group_cols = pick_existing(["route", "direction", "trip_no", "departure_time", "destination"], tmp.columns)

    if group_cols:
        groups = tmp[group_cols].astype(str).agg("|".join, axis=1)
    else:
        groups = X.astype(str).agg("|".join, axis=1)

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        train_idx, test_idx = next(sgkf.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    except Exception:
        from sklearn.model_selection import GroupShuffleSplit

        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

else:
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

print(f"Train Late rate: {float(y_train.mean()):.4f} | Test Late rate: {float(y_test.mean()):.4f}")

# ============================================================
# Preprocess
# ============================================================
pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        # with_mean=False supports sparse matrices
        ("num", StandardScaler(with_mean=False), num_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# ============================================================
# 1) Logistic Regression + Coef plot
# ============================================================
lr = LogisticRegression(max_iter=3000, class_weight="balanced")
lr_pipe = Pipeline([("pre", pre), ("clf", lr)])
lr_pipe.fit(X_train, y_train)

y_pred_lr = lr_pipe.predict(X_test)
print("\n[Logistic Regression]")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Macro-F1:", f1_score(y_test, y_pred_lr, average="macro"))
print("Balanced Acc:", balanced_accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, digits=4))

feat_names = lr_pipe.named_steps["pre"].get_feature_names_out()
coefs = lr_pipe.named_steps["clf"].coef_.ravel()

coef_df = (
    pd.DataFrame({"feature": feat_names, "coef": coefs})
    .assign(abs_coef=lambda d: d["coef"].abs())
    .sort_values("abs_coef", ascending=False)
)

TOP_N = 25
plot_df = coef_df.head(TOP_N).sort_values("coef")

plt.figure(figsize=(10, 6))
plt.barh(plot_df["feature"], plot_df["coef"])
plt.xlabel("LR coefficient (positive ⇒ higher Late risk)")
plt.title("Top Logistic Regression Coefficients (sorted by |coef|)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "LR_Coefficients.png"), dpi=300)
plt.close()

# ============================================================
# 2) XGBoost + SHAP Summary
# ============================================================
pos = float(y_train.sum())
neg = float((1 - y_train).sum())
scale_pos_weight = (neg / pos) if pos > 0 else 1.0

xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    tree_method="hist",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
)

xgb_pipe = Pipeline([("pre", pre), ("clf", xgb)])
xgb_pipe.fit(X_train, y_train)

X_test_proc = xgb_pipe.named_steps["pre"].transform(X_test)
xgb_model = xgb_pipe.named_steps["clf"]
xgb_feat_names = xgb_pipe.named_steps["pre"].get_feature_names_out()

# SHAP sample for speed
n_shap = min(2000, X_test_proc.shape[0])
idx = np.random.RandomState(RANDOM_STATE).choice(X_test_proc.shape[0], size=n_shap, replace=False)
X_shap = X_test_proc[idx]
X_shap_dense = X_shap.toarray() if hasattr(X_shap, "toarray") else X_shap

explainer = shap.TreeExplainer(xgb_model)
# Compatibility across SHAP versions
try:
    shap_values = explainer.shap_values(X_shap_dense)
except Exception:
    shap_values = explainer(X_shap_dense).values

plt.figure()
shap.summary_plot(
    shap_values,
    X_shap_dense,
    feature_names=xgb_feat_names,
    show=False,
    max_display=25,
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "SHAP_Summary.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 3) Precision–Recall curve (XGB) + no-skill baseline
# ============================================================
proba = xgb_pipe.predict_proba(X_test)[:, 1]
ap = average_precision_score(y_test, proba)
late_rate = float(y_test.mean())

plt.figure()
PrecisionRecallDisplay.from_predictions(
    y_test,
    proba,
    name=f"XGBoost (AUPRC={ap:.3f})",
)
plt.hlines(
    y=late_rate,
    xmin=0,
    xmax=1,
    linestyles="--",
    label=f"No-skill (Late rate={late_rate:.3f})",
)
plt.title("Precision–Recall Curve (Late = positive class)")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "PR_Curve_XGB.png"), dpi=300)
plt.close()

print("\nSaved figures in:", os.path.abspath(OUT_DIR))
print(" - LR_Coefficients.png")
print(" - SHAP_Summary.png")
print(" - PR_Curve_XGB.png")
print(f"PR AUPRC (XGBoost): {ap:.4f} | Late rate (test): {late_rate:.4f}")


def summarize(name, y_true, y_pred, y_proba):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "MacroF1": f1_score(y_true, y_pred, average="macro"),
        "F1_Late": f1_score(y_true, y_pred, pos_label=1),  # Late=1
        "Recall_Late": recall_score(y_true, y_pred, pos_label=1),
        "Precision_Late": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "AUPRC": average_precision_score(y_true, y_proba),
        "LateRate": float(np.mean(y_true)),
    }


# ----------------------------
# Majority baseline (predict majority class from TRAIN)
# ----------------------------
majority_class = int(pd.Series(y_train).mode()[0])
y_pred_base = np.full_like(y_test, fill_value=majority_class)
# constant probability score; AP baseline ≈ Late rate
p_train = float(np.mean(y_train))
y_proba_base = np.full(shape=len(y_test), fill_value=p_train)

# ----------------------------
# LR metrics
# ----------------------------
y_proba_lr = lr_pipe.predict_proba(X_test)[:, 1]

# ----------------------------
# XGB metrics
# ----------------------------
y_pred_xgb = xgb_pipe.predict(X_test)
y_proba_xgb = proba

results = [
    summarize("Majority Baseline", y_test, y_pred_base, y_proba_base),
    summarize("Logistic Regression", y_test, y_pred_lr, y_proba_lr),
    summarize("XGBoost", y_test, y_pred_xgb, y_proba_xgb),
]

res_df = pd.DataFrame(results)
print("\n=== TABLE METRICS (copy these into LaTeX) ===")
print(res_df[["Model", "Accuracy", "MacroF1", "BalancedAcc", "F1_Late", "AUPRC", "LateRate"]].to_string(index=False))

print("\n=== LaTeX rows ===")
for r in results:
    print(
        f"{r['Model']} & {r['Accuracy']:.3f} & {r['MacroF1']:.3f} & {r['BalancedAcc']:.3f} & {r['F1_Late']:.3f} & {r['AUPRC']:.3f} \\\\"  # noqa: W605
    )

# Optional: show label prevalence in the final, deduped dataset
print("\nLabel distribution (deduped):")
print(pd.Series(y).value_counts(normalize=True).rename({0: "On-Time", 1: "Late"}))

# Optional quick inference timing (useful to fill paper placeholders)
# NOTE: This is *inference only*, not training. Remove if not needed.
for name, model in [("LR", lr_pipe), ("XGB", xgb_pipe)]:
    X_small = X_test.iloc[:1000]
    # warmup
    _ = model.predict_proba(X_small)
    t0 = time.perf_counter()
    _ = model.predict_proba(X_small)
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"Inference timing ({name}): {dt_ms/len(X_small):.4f} ms / query (batch=1000)")

def single_query_latency_ms(pipe, X_test, n=200):
    # warmup
    for _ in range(10):
        pipe.predict_proba(X_test.iloc[:1])
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        pipe.predict_proba(X_test.iloc[:1])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.median(times)), float(np.percentile(times, 90))

lr_med, lr_p90 = single_query_latency_ms(lr_pipe, X_test)
xgb_med, xgb_p90 = single_query_latency_ms(xgb_pipe, X_test)
print("LR single-query median / p90 (ms):", lr_med, lr_p90)
print("XGB single-query median / p90 (ms):", xgb_med, xgb_p90)
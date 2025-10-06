import argparse, os
import numpy as np
import pandas as pd

# ---------- Settings ----------
DEFAULT_INPUT = "data/anomaly_detection_results.csv"
DEFAULT_OUT   = "data/company_risk.csv"
ENCODING      = "utf-8-sig"

# Thresholds (when a rule fires)
TH_REV_YOY  = -0.10   # Revenue YoY ≤ -10%
TH_TREND3Y  = -0.10   # 3-year average YoY ≤ -10%/yr
TH_EMP_YOY  = -0.10   # Employees YoY ≤ -10%

# Weights (importance; sum to 100)
W_REV_YOY = 40
W_TREND3Y = 35
W_EMP     = 15
W_BRANCH  = 10

# Risk bands
BAND_WATCH = 40.0
BAND_HIGH  = 70.0

# ---------- Helpers ----------
def read_csv_auto(path):
    for sep in (",", ";"):
        try:
            return pd.read_csv(path, encoding=ENCODING, sep=sep)
        except Exception:
            pass
    raise SystemExit(f"[!] Cannot read CSV: {path}")

# ---------- Features ----------
def build_features(df):
    df = df.sort_values(["الرقم_الضريبي", "السنة"]).copy()
    # choose revenue column (prefer inflation-adjusted if present)
    rev = "المبيعات_بعد_التضخم" if "المبيعات_بعد_التضخم" in df.columns else "الإيرادات_جنيه"

    # 1) Revenue YoY
    df["نمو_الإيراد_سنوياً"] = df.groupby("الرقم_الضريبي")[rev].pct_change()

    # 2) Direction over time = average of last up to 3 YoY values
    df["اتجاه_3_سنوات"] = (
        df.groupby("الرقم_الضريبي")["نمو_الإيراد_سنوياً"]
          .apply(lambda s: s.rolling(3, min_periods=2).mean())
          .reset_index(level=0, drop=True)
    )

    # 3) Employees YoY (if available)
    if "الموظفون" in df.columns:
        df["تغير_الموظفين_YoY"] = df.groupby("الرقم_الضريبي")["الموظفون"].pct_change()
    else:
        df["تغير_الموظفين_YoY"] = np.nan

    # 4) Branch reductions (if available)
    if "عدد الفروع" in df.columns:
        prev = df.groupby("الرقم_الضريبي")["عدد الفروع"].shift(1)
        df["branches_drop"] = (df["عدد الفروع"] < prev).astype(int)
    else:
        df["branches_drop"] = 0

    return df

# ---------- Scoring ----------
def score_companies(feats):
    rows = []
    for _, r in feats.iterrows():
        score = 0.0
        reasons = []

        # Rule 1: Revenue decline (YoY)
        if pd.notna(r["نمو_الإيراد_سنوياً"]) and r["نمو_الإيراد_سنوياً"] <= TH_REV_YOY:
            score += W_REV_YOY; reasons.append("Revenue YoY ≤ -10%")

        # Rule 2: Direction over time (3-year trend)
        if pd.notna(r["اتجاه_3_سنوات"]) and r["اتجاه_3_سنوات"] <= TH_TREND3Y:
            score += W_TREND3Y; reasons.append("3-year average ≤ -10%/yr")

        # Rule 3: Headcount change (YoY)
        if pd.notna(r["تغير_الموظفين_YoY"]) and r["تغير_الموظفين_YoY"] <= TH_EMP_YOY:
            score += W_EMP; reasons.append("Employees YoY ≤ -10%")

        # Rule 4: Branch reductions
        if int(r.get("branches_drop", 0)) == 1:
            score += W_BRANCH; reasons.append("Branches decreased vs last year")

        score = min(score, 100.0)

        # Banding
        band = "Stable"
        if score >= BAND_HIGH:
            band = "High"
        elif score >= BAND_WATCH:
            band = "Watchlist"

        rows.append({
            "firm_id": r["الرقم_الضريبي"],
            "year": int(r["السنة"]),
            "industry": r.get("القطاع", None),
            "risk_score": float(score),
            "band": band,
            "reason_codes": "; ".join(reasons)
        })

    return pd.DataFrame(rows)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Company risk scorer (4-rule, simple)")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"[!] Not found: {args.input}")

    df = read_csv_auto(args.input)
    feats = build_features(df)
    res = score_companies(feats)
    res.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {args.out} rows={len(res)}")

if __name__ == "__main__":
    main()
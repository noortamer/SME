# -*- coding: utf-8 -*-
"""
Company risk per firm×year using 4 rules (+simple forecast).
Outputs only: company_risk.csv

HARDENED:
- Arabic/English column auto-detect
- Rename lookup fields BEFORE merge to avoid _x/_y:
    sector      -> __sector__
    class_code  -> __class_code__
    class_name  -> __class_name__
- Keys normalized (firm_id, year), join coverage check, early fail if class looks empty
"""

import argparse, os
import numpy as np
import pandas as pd

# ==== Defaults ====
DEF_PANEL  = "data/anomaly_detection_results.csv"
DEF_LOOKUP = "data/الشركات_ar_enriched_fixed.csv"
DEF_OUT    = "data/new_company_risk.csv"
ENC        = "utf-8-sig"

# ==== Thresholds & Weights ====
TH_REV_YOY   = -0.10   # Revenue YoY ≤ -10%
TH_TREND3Y   = -0.10   # Avg of last ≤3 YoYs ≤ -10%/yr
TH_EMP_YOY   = -0.10   # Employees YoY ≤ -10%

W_REV_YOY = 40
W_TREND3Y = 35
W_EMP     = 15
W_BRANCH  = 10

BAND_WATCH = 40.0
BAND_HIGH  = 70.0
FORECAST_HIGH  = 70.0
FORECAST_WATCH = 55.0

# ---------- Utils ----------
def read_csv(path):
    for enc in ("utf-8-sig","utf-8","cp1256"):
        for sep in (",",";","|","\t"):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] >= 2:
                    return df
            except Exception:
                pass
    raise SystemExit(f"[!] Cannot read CSV: {path}")

def pick(cols, candidates):
    for c in candidates:
        if c in cols: return c
    return None

def require(name, val):
    if not val: raise SystemExit(f"[!] Required column not found: {name}")

def ensure_dirs(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Company risk (firm×year) only")
    ap.add_argument("--panel", default=DEF_PANEL)
    ap.add_argument("--lookup", default=DEF_LOOKUP)
    ap.add_argument("--out", default=DEF_OUT)

    # If your lookup headers differ slightly, you can force them:
    ap.add_argument("--lookup-class-code", default=None, help="Exact header for class code (e.g., الفرع_كود)")
    ap.add_argument("--lookup-class-name", default=None, help="Exact header for class name (e.g., الفرع)")

    # If IDs are zero-padded in one file but not the other:
    ap.add_argument("--id-zfill", type=int, default=None, help="Zero-pad firm_id to N digits in both panel & lookup")

    args = ap.parse_args()

    # ---- Load ----
    if not os.path.exists(args.panel):  raise SystemExit(f"[!] Not found: {args.panel}")
    if not os.path.exists(args.lookup): raise SystemExit(f"[!] Not found: {args.lookup}")

    panel  = read_csv(args.panel)
    lookup = read_csv(args.lookup)

    # ---- Column mapping ----
    pcols = set(panel.columns); lcols = set(lookup.columns)
    firm_id = pick(pcols, ["الرقم_الضريبي","firm_id","tax_id"])
    year    = pick(pcols, ["السنة","year"])
    revenue = pick(pcols, ["المبيعات_بعد_التضخم","الإيرادات_جنيه","revenue_real","revenue"])
    emp     = pick(pcols, ["الموظفون","employees"])
    branches= pick(pcols, ["عدد الفروع","branches"])

    lk_id   = pick(lcols, ["الرقم_الضريبي","firm_id","tax_id"])
    sector  = pick(lcols, ["القطاع","sector"])
    class_code_hdr = args.lookup_class_code or pick(lcols, ["الفرع_كود","class_code","كود_الفئة","رمز_الفئة","ISIC4_Class_Code","ISIC_Code","isic_code"])
    class_name_hdr = args.lookup_class_name or pick(lcols, ["الفرع","class_name","اسم_الفئة","ISIC4_Class_Name","isic_name"])

    for n,v in [("panel.firm_id",firm_id), ("panel.year",year), ("panel.revenue",revenue),
                ("lookup.firm_id",lk_id), ("lookup.sector",sector),
                ("lookup.class_code",class_code_hdr), ("lookup.class_name",class_name_hdr)]:
        require(n, v)

    print(f"[map] panel:  firm_id={firm_id}, year={year}, revenue={revenue}, emp={emp}, branches={branches}")
    print(f"[map] lookup: id={lk_id}, sector={sector}, class_code={class_code_hdr}, class_name={class_name_hdr}")

    # ---- Normalize join keys ----
    panel[firm_id] = panel[firm_id].astype(str).str.strip()
    lookup[lk_id]  = lookup[lk_id].astype(str).str.strip()
    if args.id_zfill:
        panel[firm_id] = panel[firm_id].str.zfill(args.id_zfill)
        lookup[lk_id]  = lookup[lk_id].str.zfill(args.id_zfill)
    panel[year] = pd.to_numeric(panel[year], errors="coerce").astype("Int64")

    # ---- Prepare lookup subset with SAFE internal names (avoid _x/_y) ----
    lk_sub = lookup[[lk_id, sector, class_code_hdr, class_name_hdr]].drop_duplicates().rename(
        columns={
            sector: "__sector__",                      # protect sector
            class_code_hdr: "__class_code__",         # protect class code
            class_name_hdr: "__class_name__",         # protect class name
        }
    )

    # ---- Merge ----
    df = panel.merge(lk_sub, left_on=firm_id, right_on=lk_id, how="left")

    coverage = 1 - df[lk_id].isna().mean()
    print(f"[join] panel→lookup coverage: {coverage:.1%}")
    if coverage < 0.95:
        bad = df.loc[df[lk_id].isna(), [firm_id]].drop_duplicates().head(10)
        print("[join] sample unmatched firm_ids:", bad[firm_id].tolist())
        raise SystemExit("[!] Too many firms in panel not found in lookup. Check IDs/whitespace/encoding or use --id-zfill.")

    # Verify class columns AFTER merge (using internal names)
    nn_code = float(1 - df["__class_code__"].isna().mean())
    nn_name = float(1 - df["__class_name__"].isna().mean())
    print(f"[class] After merge: code non-null={nn_code:.1%}, name non-null={nn_name:.1%}")
    if nn_code < 0.05 or nn_name < 0.05:
        print("[class] WARNING: Class columns look empty after merge. Sample:")
        print(df[[firm_id, "__class_code__", "__class_name__"]].head(10).to_string(index=False))
        raise SystemExit("[!] Class columns are empty after merge. Confirm lookup headers/values or pass --lookup-class-code/--lookup-class-name explicitly.")

    # ---- Features (4 rules) ----
    df = df.sort_values([firm_id, year]).copy()

    df["rev_num"] = pd.to_numeric(df[revenue], errors="coerce")
    df["rev_yoy"] = df.groupby(firm_id)["rev_num"].pct_change()

    df["trend3y"] = (
        df.groupby(firm_id)["rev_yoy"]
          .apply(lambda s: s.rolling(3, min_periods=2).mean())
          .reset_index(level=0, drop=True)
    )

    if emp in pcols:
        df["emp_num"] = pd.to_numeric(df[emp], errors="coerce")
        df["emp_yoy"] = df.groupby(firm_id)["emp_num"].pct_change()
    else:
        df["emp_yoy"] = np.nan

    if branches in pcols:
        prev_b = df.groupby(firm_id)[branches].shift(1)
        df["branch_drop"] = (pd.to_numeric(df[branches], errors="coerce") < pd.to_numeric(prev_b, errors="coerce")).astype(int)
    else:
        df["branch_drop"] = 0

    # ---- Score per firm×year ----
    rows=[]
    for _, r in df.iterrows():
        score=0; reasons=[]
        if pd.notna(r["rev_yoy"]) and r["rev_yoy"] <= TH_REV_YOY:
            score += W_REV_YOY; reasons.append("Revenue YoY ≤ -10%")
        if pd.notna(r["trend3y"]) and r["trend3y"] <= TH_TREND3Y:
            score += W_TREND3Y; reasons.append("3y direction ≤ -10%/yr")
        if pd.notna(r["emp_yoy"]) and r["emp_yoy"] <= TH_EMP_YOY:
            score += W_EMP; reasons.append("Employees YoY ≤ -10%")
        if int(r.get("branch_drop",0)) == 1:
            score += W_BRANCH; reasons.append("Branches decreased")
        score = float(min(score, 100.0))
        band = "High" if score>=BAND_HIGH else ("Watchlist" if score>=BAND_WATCH else "Stable")
        rows.append({
            "firm_id": r[firm_id],
            "year": int(r[year]) if pd.notna(r[year]) else None,
            "industry": r.get("__sector__", None),
            "class_code": r.get("__class_code__", None),
            "class_name": r.get("__class_name__", None),
            "risk_score": score,
            "band": band,
            "reason_codes": "; ".join(reasons)
        })
    risk = pd.DataFrame(rows).sort_values(["firm_id","year"])

    # ---- Simple forecast flag ----
    risk["forecast_close_risk"] = "Low"
    prev_b = risk.groupby("firm_id")["band"].shift(1)
    prev_s = risk.groupby("firm_id")["risk_score"].shift(1)
    now_high   = risk["risk_score"] >= FORECAST_HIGH
    keep_watch = (risk["band"].eq("Watchlist") & prev_b.eq("Watchlist") &
                  (risk["risk_score"]>=FORECAST_WATCH) & (prev_s>=FORECAST_WATCH))
    risk.loc[now_high | keep_watch, "forecast_close_risk"] = "High"

    # ---- Save ----
    ensure_dirs(args.out)
    risk.to_csv(args.out, index=False, encoding=ENC)
    print(f"[OK] Saved {args.out}  rows={len(risk)}")

if __name__ == "__main__":
    main()
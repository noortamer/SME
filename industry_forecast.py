import argparse
from dataclasses import dataclass
import pandas as pd
import numpy as np
import os

# ======= DEFAULTS (edit these) =======
DEFAULT_INPUT = "data/anomaly_detection_results.csv"
DEFAULT_OUT   = "data/industry_tags.csv"
DEFAULT_ENCODING = "utf-8-sig"

@dataclass
class IndustryCfgEGMin:
    w_years: int = 3
    boom_thr: float = 0.05
    shrink_thr: float = -0.05

def tag_industries_eg_min(df: pd.DataFrame, cfg: IndustryCfgEGMin = IndustryCfgEGMin()) -> pd.DataFrame:
    df = df.sort_values(["القطاع","الرقم_الضريبي","السنة"]).copy()
    rev_col = "المبيعات_بعد_التضخم" if "المبيعات_بعد_التضخم" in df.columns else "الإيرادات_جنيه"

    df["نمو_الإيراد_سنوياً"] = df.groupby("الرقم_الضريبي")[rev_col].pct_change(1)

    ind = (df.groupby(["القطاع","السنة"])["نمو_الإيراد_سنوياً"]
             .median()
             .rename("نمو_القطاع")
             .reset_index())

    ind["ma"] = ind.groupby("القطاع")["نمو_القطاع"].transform(
        lambda s: s.rolling(cfg.w_years, min_periods=2).mean()
    )

    rows = []
    for sec, sub in ind.groupby("القطاع", sort=False):
        sub = sub.sort_values("السنة")
        for _, r in sub.iterrows():
            tag = "Neutral"
            if pd.notna(r["ma"]) and r["ma"] >= cfg.boom_thr:
                tag = "Booming"
            elif pd.notna(r["ma"]) and r["ma"] <= cfg.shrink_thr:
                tag = "Shrinking"
            rows.append({
                "industry": sec,
                "year": int(r["السنة"]),
                "tag": tag,
                "ma": float(r["ma"]) if pd.notna(r["ma"]) else np.nan
            })
    return pd.DataFrame(rows)

def _read_csv_auto(path: str) -> pd.DataFrame:
    encodings = [DEFAULT_ENCODING, "utf-8"]
    seps = [",",";"]
    ex = None
    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(path, encoding=enc, sep=sep)
            except Exception as e:
                ex = e
    raise ex if ex else RuntimeError("Failed to read CSV.")

def main():
    parser = argparse.ArgumentParser(description="Standalone industry tagger (reads CSV itself).")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV path (default: anomaly_detection_results.csv)")
    parser.add_argument("--out",   default=DEFAULT_OUT,   help="Output CSV (default: industry_tags.csv)")
    parser.add_argument("--boom-thr", type=float, default=0.05)
    parser.add_argument("--shrink-thr", type=float, default=-0.05)
    parser.add_argument("--window-years", type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] Input not found: {args.input}")
        return

    df = _read_csv_auto(args.input)
    cfg = IndustryCfgEGMin(w_years=args.window_years, boom_thr=args.boom_thr, shrink_thr=args.shrink_thr)
    res = tag_industries_eg_min(df, cfg)
    res.to_csv(args.out, index=False)
    print(f"[OK] Saved {args.out}  rows={len(res)}")

if __name__ == "__main__":
    main()
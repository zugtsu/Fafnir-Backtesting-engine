"""
Convert raw HistData downloads into the cache format this project reads.

HistData hands you one zip per pair per year, containing a semicolon-delimited
file with no header:

    20150101 170000;0.81875;0.81875;0.81856;0.81863;0

Columns are YYYYMMDD HHMMSS;open;high;low;close;volume, and the clock is
Eastern Standard WITHOUT daylight saving - a fixed UTC-5 all year. That is why
the FX week opens at 22:00 UTC in this project's data: 17:00 EST + 5.

Point this at a folder of zips (or loose CSVs) and it writes
cache/histdata/<PAIR>_<YEAR>.csv in the exact shape fetch_histdata_cache()
expects, then you rebuild the store and every backtest reruns over the longer
history with no other changes.

    python scripts/import_histdata.py ~/Downloads
    python scripts/import_histdata.py ~/Downloads --check      # dry run

Afterwards:
    python -m saty_bt.data.build
"""
from __future__ import annotations
import argparse, io, re, sys, zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cache" / "histdata"
COLS = ["ts", "open", "high", "low", "close", "volume"]
# HistData's M1 ASCII clock is EST with no DST, i.e. a fixed UTC-5
EST_FIXED = pd.Timedelta(hours=5)


def parse_one(text: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep=";", header=None,
                     names=["stamp", "open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(df["stamp"], format="%Y%m%d %H%M%S", errors="coerce")
    bad = ts.isna().sum()
    if bad:
        print(f"    {name}: dropped {bad} unparseable rows")
    df = df[ts.notna()].copy()
    df["ts"] = (ts[ts.notna()] + EST_FIXED).dt.tz_localize("UTC")
    return df[COLS].sort_values("ts").drop_duplicates("ts")


def find_pair_year(name: str) -> tuple[str, str] | None:
    m = re.search(r"([A-Z]{6})[_ ]?M1[_ ]?(\d{4})", name.upper())
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"([A-Z]{6}).*?(\d{4})", name.upper())
    return (m.group(1), m.group(2)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="where the HistData zips or csvs are")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    a = ap.parse_args()
    src = Path(a.folder).expanduser()
    if not src.exists():
        sys.exit(f"no such folder: {src}")

    files = sorted([p for p in src.iterdir()
                    if p.suffix.lower() in (".zip", ".csv") and "HISTDATA" in p.name.upper()
                    or (p.suffix.lower() == ".zip" and re.search(r"[A-Z]{6}", p.name.upper()))])
    if not files:
        files = sorted([p for p in src.iterdir() if p.suffix.lower() in (".zip", ".csv")])
    if not files:
        sys.exit(f"nothing to import in {src}")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} candidate file(s) in {src}\n")
    done = 0
    for p in files:
        key = find_pair_year(p.name)
        if not key:
            print(f"  skip {p.name} - cannot read pair/year from the name")
            continue
        pair, year = key
        try:
            if p.suffix.lower() == ".zip":
                with zipfile.ZipFile(p) as z:
                    inner = [n for n in z.namelist() if n.lower().endswith(".csv")]
                    if not inner:
                        print(f"  skip {p.name} - no csv inside"); continue
                    text = z.read(inner[0]).decode("utf-8", "ignore")
            else:
                text = p.read_text(encoding="utf-8", errors="ignore")
            df = parse_one(text, p.name)
        except Exception as e:
            print(f"  FAIL {p.name}: {type(e).__name__}: {e}"); continue

        dest = OUT / f"{pair}_{year}.csv"
        note = "  (exists - would overwrite)" if dest.exists() else ""
        print(f"  {p.name}\n    -> {dest.name}  {len(df):,} rows  "
              f"{df.ts.min()} .. {df.ts.max()}{note}")
        if not a.check:
            df.to_csv(dest, index=False)
            done += 1

    if a.check:
        print("\ndry run - nothing written. Drop --check to import.")
        return
    print(f"\nwrote {done} file(s) into {OUT}")

    # show what the store will see once rebuilt
    have = {}
    for f in sorted(OUT.glob("*.csv")):
        m = re.match(r"([A-Z]{6})_(\d{4})", f.name)
        if m:
            have.setdefault(m.group(1), []).append(m.group(2))
    print("\ncoverage now on disk:")
    for pair, yrs in sorted(have.items()):
        ys = sorted(set(yrs))
        print(f"  {pair:<8} {len(ys):>2} periods  {ys[0]} .. {ys[-1]}")
    print("\nnext:  python -m saty_bt.data.build")


if __name__ == "__main__":
    main()

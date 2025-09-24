"""
CLI CSV Analyzer (v1.2)
-----------------------
A simple command-line tool for analyzing CSV files inside Docker.

Features:
- Auto-detect delimiter (unless --delimiter is provided).
- Summary (rows, columns, approx memory, headers).
- Per-column profiles:
    * numeric  -> mean, std, min, quartiles, max
    * datetime -> min/max (auto-detected)
    * text     -> unique count + top-N frequent values
- Reads from a file path or from stdin ("-").
- Optional run history (SQLite) in /data/history.db.

Docker examples:
  docker run --rm -v ${PWD}:/mnt csv-analyzer:1.2 analyze /mnt/iris.csv
  docker run --rm -v ${PWD}:/mnt -v csvan_data:/data csv-analyzer:1.2 analyze /mnt/your.csv --save-history
  docker run --rm -v csvan_data:/data csv-analyzer:1.2 history
"""

from __future__ import annotations

import argparse
import sys
import os
import json
import sqlite3
import datetime as dt
import io
import warnings

import pandas as pd
import numpy as np
from tabulate import tabulate

# ============================================================================
# Config & History DB
# ============================================================================
DATA_DIR = os.environ.get("CSVAN_DATA_DIR", "/data")
os.makedirs(DATA_DIR, exist_ok=True)
HIST_DB = os.path.join(DATA_DIR, "history.db")


def _connect():
    """Connect to the SQLite history database."""
    con = sqlite3.connect(HIST_DB)
    con.row_factory = sqlite3.Row
    return con


def init_history():
    """Create history table if it doesn't exist."""
    with _connect() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS runs(
              id INTEGER PRIMARY KEY,
              ts TEXT NOT NULL,
              filename TEXT,
              rows INTEGER,
              cols INTEGER,
              read_seconds REAL,
              notes TEXT
            )
            """
        )
        con.commit()


# ============================================================================
# Helpers
# ============================================================================
def human_bytes(n: int | None) -> str:
    """Pretty-print a byte count."""
    if n is None:
        return "–"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


def sniff_delimiter(_source, explicit: str | None):
    """Return user-provided delimiter, or None to let pandas sniff."""
    return explicit or None


def now_utc() -> dt.datetime:
    """Timezone-aware UTC now (avoids deprecated utcnow())."""
    return dt.datetime.now(dt.timezone.utc)


def iso_utc(ts: dt.datetime | int | float | None = None) -> str:
    """Return ISO8601 string (UTC) from timestamp/datetime/None."""
    if ts is None:
        ts = now_utc()
    elif isinstance(ts, (int, float)):
        ts = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
    elif ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.replace(microsecond=0).isoformat()  # includes +00:00


def try_parse_datetime(series: pd.Series) -> tuple[bool, pd.Series | None]:
    """
    Try to detect a datetime-like column by probing conversion.
    Suppresses Pandas 'Could not infer format' warnings, since we're only
    testing if values are plausibly datetimes.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        parsed = pd.to_datetime(series, errors="coerce", utc=True)

    valid_ratio = parsed.notna().mean()
    return (valid_ratio >= 0.8), (parsed if valid_ratio >= 0.8 else None)


def try_float(x):
    """Safe float conversion for display; leaves None for NaN."""
    try:
        return None if pd.isna(x) else float(x)
    except Exception:
        return None


def fmt(x):
    """Format numeric nicely for text output."""
    return "—" if x is None else f"{x:.6g}"


# ============================================================================
# Core CSV Analysis
# ============================================================================
def analyze_csv(
    source: str,
    columns: list[str] | None = None,
    sep: str | None = None,
    no_header: bool = False,
    topn: int = 5,
    nrows: int | None = None,
) -> tuple[dict, list[dict]]:
    """
    Read and analyze a CSV file or stdin.
    Returns (summary dict, list of per-column profiles).
    """
    read_kwargs: dict = {
        "sep": sniff_delimiter(source, sep),
        "engine": "python",  # needed when sep=None
        "nrows": nrows,
    }
    if no_header:
        read_kwargs["header"] = None

    # --- Load CSV (file path or stdin) --------------------------------------
    try:
        if source == "-":
            data = sys.stdin.read()
            df = pd.read_csv(io.StringIO(data), **read_kwargs)
            filename = None
        else:
            df = pd.read_csv(source, **read_kwargs)
            filename = os.path.basename(source)
    except Exception as e:
        raise SystemExit(f"Failed to read CSV: {e}")

    # Assign default names if no header row
    if no_header:
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

    # Keep only requested columns (if provided)
    if columns:
        keep = [c for c in columns if c in df.columns]
        missing = sorted(set(columns) - set(keep))
        df = df[keep]
        if missing:
            print(
                f"Warning: columns not found and ignored: {', '.join(missing)}",
                file=sys.stderr,
            )

    # Approx memory usage
    try:
        mem_bytes = int(df.memory_usage(deep=True).sum())
    except Exception:
        mem_bytes = None

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "approx_memory": mem_bytes,
        "filename": filename,
    }

    # --- Per-column profiling ----------------------------------------------
    profiles: list[dict] = []
    for col in df.columns:
        s = df[col]
        miss = int(s.isna().sum())
        base = {"name": col, "dtype": str(s.dtype), "count": int(s.size), "missing": miss}

        if pd.api.types.is_numeric_dtype(s):
            # Numeric summary
            desc = s.describe(percentiles=[0.25, 0.5, 0.75])
            prof = {
                **base,
                "kind": "numeric",
                "mean": try_float(desc.get("mean")),
                "std": try_float(desc.get("std")),
                "min": try_float(desc.get("min")),
                "p25": try_float(desc.get("25%")),
                "median": try_float(desc.get("50%")),
                "p75": try_float(desc.get("75%")),
                "max": try_float(desc.get("max")),
            }

        else:
            # Datetime detection (on string view)
            is_dt, parsed = try_parse_datetime(s.astype("string"))
            if is_dt:
                non_null = parsed.dropna()
                prof = {
                    **base,
                    "kind": "datetime",
                    "min": (non_null.min().isoformat() if not non_null.empty else None),
                    "max": (non_null.max().isoformat() if not non_null.empty else None),
                }
            else:
                # Text stats: unique count + top-N frequent values
                vc = s.astype("string").str.strip().replace("", np.nan)
                uniq = int(vc.nunique(dropna=True))
                top_counts = vc.value_counts(dropna=True).head(topn)
                prof = {
                    **base,
                    "kind": "text",
                    "unique": uniq,
                    "top": [(str(val), int(cnt)) for val, cnt in top_counts.items()],
                }

        profiles.append(prof)

    return summary, profiles


# ============================================================================
# Output Renderers
# ============================================================================
def render_text(summary: dict, profiles: list[dict]) -> str:
    """Pretty text output."""
    out: list[str] = []
    out.append("=== CSV Summary ===")
    out.append(f"Rows     : {summary['rows']}")
    out.append(f"Columns  : {summary['columns']}")
    out.append(f"Memory   : {human_bytes(summary['approx_memory'])}")
    headers = summary.get("column_names") or []
    out.append(f"Headers  : {', '.join(headers) if headers else '—'}")
    out.append("")

    for p in profiles:
        out.append(f"[{p['name']}] ({p['kind']}, dtype={p['dtype']})")
        out.append(f"  count={p['count']}  missing={p['missing']}")
        if p["kind"] == "numeric":
            tbl = [
                ["mean", fmt(p["mean"])],
                ["std", fmt(p["std"])],
                ["min", fmt(p["min"])],
                ["25%", fmt(p["p25"])],
                ["median", fmt(p["median"])],
                ["75%", fmt(p["p75"])],
                ["max", fmt(p["max"])],
            ]
            out.append(tabulate(tbl, tablefmt="plain"))
        elif p["kind"] == "datetime":
            out.append(f"  min={p['min'] or '—'}")
            out.append(f"  max={p['max'] or '—'}")
        else:  # text
            out.append(f"  unique={p.get('unique','—')}")
            tops = p.get("top") or []
            if tops:
                out.append("  top values:")
                for val, cnt in tops:
                    out.append(f"    {val[:60]} — {cnt}")
        out.append("")
    return "\n".join(out)


# ============================================================================
# CLI
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="CSV Analyzer — summaries & per-column stats"
    )
    sub = parser.add_subparsers(dest="cmd")

    # analyze
    p_run = sub.add_parser("analyze", help="Analyze a CSV file or stdin ('-')")
    p_run.add_argument("file", nargs="?", default="-", help="Path or '-' for stdin")
    p_run.add_argument("--columns", help="Comma-separated subset of columns")
    p_run.add_argument("--delimiter", help="Delimiter (auto if omitted)")
    p_run.add_argument("--no-header", action="store_true", help="Treat first row as data")
    p_run.add_argument("--topn", type=int, default=5, help="Top N for text columns")
    p_run.add_argument("--nrows", type=int, help="Read only first N rows")
    p_run.add_argument("--output", choices=["text", "json"], default="text")
    p_run.add_argument("--save-history", action="store_true", help="Persist run info")

    # history
    sub.add_parser("history", help="Show last 10 runs from history")

    args = parser.parse_args()

    # history command
    if args.cmd == "history":
        init_history()
        with _connect() as con:
            rows = con.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT 10"
            ).fetchall()
        if not rows:
            print("No history yet.")
            return 0
        tbl = [
            [
                r["id"],
                r["ts"],
                r["filename"] or "-",
                r["rows"],
                r["cols"],
                f"{r['read_seconds']:.3f}s",
                r["notes"] or "-",
            ]
            for r in rows
        ]
        print(
            tabulate(
                tbl,
                headers=["id", "timestamp", "file", "rows", "cols", "read", "notes"],
                tablefmt="github",
            )
        )
        return 0

    # default to analyze
    if args.cmd != "analyze":
        parser.print_help()
        return 0

    columns = [c.strip() for c in args.columns.split(",")] if args.columns else None
    src = args.file
    sep = args.delimiter

    # measure read time (timezone-aware)
    t0 = now_utc()
    summary, profiles = analyze_csv(
        src, columns=columns, sep=sep, no_header=args.no_header, topn=args.topn, nrows=args.nrows
    )
    dt_read = (now_utc() - t0).total_seconds()

    # output
    if args.output == "json":
        print(json.dumps({"summary": summary, "profiles": profiles}, indent=2))
    else:
        print(render_text(summary, profiles))

    # optional history
    if args.save_history:
        init_history()
        with _connect() as con:
            con.execute(
                "INSERT INTO runs(ts, filename, rows, cols, read_seconds, notes) "
                "VALUES(?,?,?,?,?,?)",
                (
                    iso_utc(),
                    summary.get("filename"),
                    summary["rows"],
                    summary["columns"],
                    dt_read,
                    f"topn={args.topn}; nrows={args.nrows or 'all'}",
                ),
            )
            con.commit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

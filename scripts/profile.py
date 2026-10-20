#!/usr/bin/env python
"""Quick profiler for an unknown CSV.

Usage:
    python scripts/profile.py data/raw/<file>.csv [out.md]

Goal: within seconds, know shape, dtypes, missingness, cardinality,
date coverage, numeric spread, duplicates and candidate key columns.
Everything printed here is derived from the provided data only.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


def read_any(path: Path) -> pd.DataFrame:
    """Read a CSV trying common encodings and separators."""
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1"):
        for sep in (None, ",", ";", "\t", "|"):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                if df.shape[1] >= 2 or sep is None:
                    print(f"[read] encoding={enc} sep={sep!r} -> {df.shape}")
                    return df
            except Exception as e:  # noqa: BLE001
                last_err = e
    raise RuntimeError(f"cannot read {path}: {last_err}")


def profile(df: pd.DataFrame, name: str) -> str:
    out = io.StringIO()

    def w(line: str = "") -> None:
        out.write(line + "\n")

    w(f"# Data profile: `{name}`")
    w()
    w(f"- rows: **{len(df):,}**")
    w(f"- columns: **{df.shape[1]}**")
    w(f"- duplicated rows: **{int(df.duplicated().sum()):,}**")
    w()

    w("## Columns")
    w()
    w("| # | column | dtype | non-null | null% | unique | sample |")
    w("|---|--------|-------|----------|-------|--------|--------|")
    for i, col in enumerate(df.columns):
        s = df[col]
        null_pct = s.isna().mean() * 100
        try:
            nunique = s.nunique(dropna=True)
        except TypeError:
            nunique = -1
        sample = s.dropna().astype(str).head(3).tolist()
        sample_s = ", ".join(x[:24] for x in sample)
        w(f"| {i} | `{col}` | {s.dtype} | {s.notna().sum():,} | {null_pct:.1f}% | {nunique:,} | {sample_s} |")
    w()

    # candidate keys
    w("## Candidate keys")
    w()
    n = len(df)
    for col in df.columns:
        try:
            u = df[col].nunique(dropna=True)
        except TypeError:
            continue
        if n and u == n:
            w(f"- `{col}` is unique (n={n:,}) -> possible primary key")
        elif n and 0.95 * n <= u < n:
            w(f"- `{col}` nearly unique ({u:,}/{n:,}) -> possible entity id")
    w()

    # numeric
    num = df.select_dtypes("number")
    if not num.empty:
        w("## Numeric summary")
        w()
        w(num.describe().T.to_markdown(floatfmt=".3f"))
        w()

    # datetime-ish
    dt_cols = df.select_dtypes("datetime").columns.tolist()
    for col in df.columns:
        if col in dt_cols or is_string_dtype(df[col]) or is_object_dtype(df[col]):
            parsed = None
            try:
                parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            except Exception:  # noqa: BLE001
                parsed = None
            if parsed is not None and parsed.notna().mean() > 0.9:
                w(f"## Time column: `{col}`")
                w()
                w(f"- range: {parsed.min()} .. {parsed.max()}")
                w(f"- span: {(parsed.max() - parsed.min())}")
                w()

    # categorical
    for col in df.columns:
        if is_string_dtype(df[col]) or is_object_dtype(df[col]):
            try:
                u = df[col].nunique(dropna=True)
            except TypeError:
                continue
            if 0 < u <= 20:
                vc = df[col].value_counts(dropna=False).head(20)
                w(f"## Category: `{col}` ({u} distinct)")
                w()
                w(vc.to_markdown())
                w()

    return out.getvalue()


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("analysis/00_profile.md")
    df = read_any(path)
    text = profile(df, path.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(text[:4000])
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()

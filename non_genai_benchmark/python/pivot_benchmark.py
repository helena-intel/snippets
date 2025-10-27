"""
This script reads a CSV file and produces a pivot table containing the median value of a chosen numeric column per
model. Optionally you can filter the input by device and/or by a substring match on the model name. The pivot shows
devices as columns when no device filter is supplied; when a device is specified the output is a single-column table of
medians per model.

Usage:
python pivot_benchmark.py CSV [--device DEVICE] [--modelfilter MODEL_FILTER] [--col COLUMN] [--out OUT_CSV]

Positional argument:
    csv: Path to the input CSV file.

Optional arguments:
    --device, -d: Filter rows to a single device (exact match). If provided, the pivot will not include device columns — it will return the median of the chosen column for each model restricted to that device.
    --modelfilter, -m: Filter models by substring. Uses Python's str.contains semantics; provide a string and any model containing that substring will be kept.
    --col, -c: Name of the numeric column to aggregate. Default: "throughput".
    --out, -o: Optional path to write the resulting pivot table as CSV.

Data requirements:
- The CSV must contain at least the following columns: model, device, and the column specified via --col (default "throughput").
- The script is intended for CSV files that were run on the same device, with the same config. The script prints the unique device names and configs to manually verify this.

Python requirements: `pip install pandas`
"""

import argparse
from pathlib import Path

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def pivot_median(df: pd.DataFrame, col: str = "throughput", device: str | None = None, modelfilter: str | None = None) -> pd.DataFrame:
    required = {"model", col, "device"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    # It is sometimes useful to consolidate all AUTO columns but only uncomment if you know you need this
    # df["device"] = df["device"].apply(lambda x: x.split(":")[0] if "AUTO" in x else x)
    if modelfilter:
        df = df.query(f"model.str.contains('{modelfilter}')")
    include_ov_version = df["openvino"].nunique() > 1
    group_cols = ["model", "openvino"] if include_ov_version else ["model"]
    if device:
        # filter on device and group by model and if needed OpenVINO version
        df = df[df["device"] == device]
        # Pivot: index=model (or framework+model if multiple frameworks), single column of median
        result = df.groupby(group_cols, dropna=False)[col].median().to_frame(name=f"median_{col}")
    else:
        index_cols = ["model", "framework"] if include_ov_version else ["model"]
        result = df.pivot_table(index=group_cols, columns="device", values=col, aggfunc="median")
        result = result.reindex(sorted(result.columns), axis=1)
    return result.round(2)


def main():
    p = argparse.ArgumentParser(description="Create pivot table of median performance per model (optionally filter by device or model).")
    p.add_argument("csv", type=Path, help="Path to input CSV")
    p.add_argument("--device", "-d", type=str, default=None, help="Device to filter on (optional).")
    p.add_argument("--modelfilter", "-m", type=str, default=None, help="Model string to filter on (optional).")
    p.add_argument("--col", "-c", type=str, default="throughput", help="Column to aggregate (default: throughput).")
    p.add_argument("--out", "-o", type=Path, default=None, help="Optional output CSV path for the pivot table.")
    args = p.parse_args()

    df = load_csv(args.csv)
    pivot = pivot_median(df, col=args.col, device=args.device, modelfilter=args.modelfilter)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(pivot)
    print()
    print("Devices:", {sub.strip() for dev in df.device_name.unique() for sub in dev.split(",")})
    print("Configs:", df.config.unique().tolist())
    print("OpenVINO versions:", df.openvino.unique().tolist())

    if args.out:
        pivot.to_csv(args.out)
        print(f"\nSaved pivot to: {args.out}")


if __name__ == "__main__":
    main()

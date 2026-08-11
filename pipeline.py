"""
Runs the full Factory Reallocation & Shipping Optimization pipeline end to end:
  1. data_prep.py  -- clean, engineer, encode
  2. modeling.py   -- train Lead Time & Margin predictors
  3. simulate.py   -- score Factory x Region x Ship Mode scenarios
  4. recommend.py  -- rank scenarios, flag reallocation candidates

Usage:
    python pipeline.py --input "../Nassau Candy Distributor.csv" --outdir out
"""
import argparse
import subprocess
import sys


def run(args, extra):
    cmd = [sys.executable] + args + extra
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../Nassau Candy Distributor.csv")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--w-leadtime", type=float, default=0.5)
    ap.add_argument("--w-margin", type=float, default=0.5)
    args = ap.parse_args()

    run(["data_prep.py", "--input", args.input, "--outdir", args.outdir], [])
    run(["modeling.py", "--outdir", args.outdir], [])
    run(["simulate.py", "--outdir", args.outdir], [])
    run(
        ["recommend.py", "--outdir", args.outdir, "--w-leadtime", str(args.w_leadtime), "--w-margin", str(args.w_margin)],
        [],
    )


if __name__ == "__main__":
    main()

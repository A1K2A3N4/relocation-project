"""
Runs the full Factory Reallocation & Shipping Optimization pipeline end to end:
  1. data_prep.py             -- clean, engineer, encode
  2. modeling.py              -- train & evaluate Lead Time / Margin predictors
  3. clustering.py            -- cluster routes & products by performance
  4. simulate.py              -- score Factory x Region x Ship Mode scenarios
  5. simulate_reassignment.py -- score Product x Region x candidate-Factory scenarios
  6. recommend.py             -- rank ship-mode scenarios, flag reallocation candidates
  7. recommend_reassignment.py-- rank & top-N candidate factories per Product x Region
  8. kpis.py                  -- summarize Lead Time Reduction, Profit Impact
                                  Stability, and Scenario Confidence Score

Usage:
    python pipeline.py --input "Nassau Candy Distributor.csv" --outdir out
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
    ap.add_argument("--input", default="Nassau Candy Distributor.csv")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--w-leadtime", type=float, default=0.5)
    ap.add_argument("--w-margin", type=float, default=0.5)
    ap.add_argument("--top-n", type=int, default=2)
    args = ap.parse_args()

    run(["data_prep.py", "--input", args.input, "--outdir", args.outdir], [])
    run(["modeling.py", "--outdir", args.outdir], [])
    run(["clustering.py", "--outdir", args.outdir], [])
    run(["simulate.py", "--outdir", args.outdir], [])
    run(["simulate_reassignment.py", "--outdir", args.outdir], [])
    run(
        ["recommend.py", "--outdir", args.outdir, "--w-leadtime", str(args.w_leadtime), "--w-margin", str(args.w_margin)],
        [],
    )
    run(["recommend_reassignment.py", "--outdir", args.outdir, "--top-n", str(args.top_n)], [])
    run(["kpis.py", "--outdir", args.outdir], [])


if __name__ == "__main__":
    main()

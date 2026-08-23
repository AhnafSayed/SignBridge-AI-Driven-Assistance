"""Kaggle's per-file download endpoint for asl-signs turns out to only
serve a subset of signs (confirmed: retries and delays don't help, it's
not rate limiting -- some signs 404 on every sample, every time). This
probes many candidate signs cheaply (one file each) to find which ones
are actually servable, so the real core vocabulary can be built from
what's available rather than a wishlist.
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/asl-signs")
    p.add_argument("--rank-start", type=int, default=0)
    p.add_argument("--rank-end", type=int, default=250)
    p.add_argument("--exclude", nargs="*", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    full = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    ranked = full["sign"].value_counts().index.tolist()
    candidates = [s for s in ranked[args.rank_start : args.rank_end] if s not in args.exclude]

    api = KaggleApi()
    api.authenticate()

    accessible = []
    probe_dir = os.path.join(args.data_dir, "_probe")
    os.makedirs(probe_dir, exist_ok=True)

    for i, sign in enumerate(candidates):
        row = full[full["sign"] == sign].sample(1, random_state=7).iloc[0]
        try:
            api.competition_download_file(
                "asl-signs", row.path, path=probe_dir, quiet=True
            )
            accessible.append(sign)
            print(f"[{i+1}/{len(candidates)}] OK    {sign}")
        except Exception:
            print(f"[{i+1}/{len(candidates)}] FAIL  {sign}")
        # small delay to stay well clear of any rate limit
        time.sleep(0.3)

    print("\nAccessible signs:", accessible)
    print(f"\n{len(accessible)}/{len(candidates)} accessible")
    with open(os.path.join(args.data_dir, "accessible_signs.txt"), "a") as f:
        for s in accessible:
            f.write(s + "\n")


if __name__ == "__main__":
    main()

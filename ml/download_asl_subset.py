"""Download only the slice of Kaggle's asl-signs dataset needed for Core
mode training -- the full competition dataset is ~94k landmark files
(tens of GB); a 40-sign x 40-sample core vocabulary is ~1-2k files and
downloads in minutes instead of hours.

Requires train.csv already downloaded (see ml/README.md step 1) and a
valid ~/.kaggle/kaggle.json.

Usage:
    python download_asl_subset.py --data-dir data/asl-signs \
        --core-size 40 --samples-per-class 40
"""

from __future__ import annotations

import argparse
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

COMPETITION = "asl-signs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/asl-signs")
    p.add_argument("--core-size", type=int, default=40)
    p.add_argument("--samples-per-class", type=int, default=40)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def download_one(api: KaggleApi, data_dir: str, rel_path: str) -> bool:
    dest_dir = os.path.join(data_dir, os.path.dirname(rel_path))
    os.makedirs(dest_dir, exist_ok=True)
    final_path = os.path.join(data_dir, rel_path)
    if os.path.exists(final_path):
        return True
    try:
        api.competition_download_file(COMPETITION, rel_path, path=dest_dir, quiet=True)
        zip_path = os.path.join(dest_dir, os.path.basename(rel_path) + ".zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(dest_dir)
            os.remove(zip_path)
        return os.path.exists(final_path)
    except Exception as e:
        print(f"FAILED {rel_path}: {e}")
        return False


def main():
    args = parse_args()
    full_csv = os.path.join(args.data_dir, "train.csv")
    full_df = pd.read_csv(full_csv)

    top_signs = full_df["sign"].value_counts().head(args.core_size).index.tolist()
    parts = []
    for sign in top_signs:
        group = full_df[full_df["sign"] == sign]
        parts.append(group.sample(min(len(group), args.samples_per_class), random_state=args.seed))
    subset = pd.concat(parts, ignore_index=True)
    assert "sign" in subset.columns
    print(
        f"Core vocabulary: {len(top_signs)} signs, "
        f"{len(subset)} sequences across {subset['participant_id'].nunique()} signers"
    )

    api = KaggleApi()
    api.authenticate()

    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, api, args.data_dir, row.path): row.path
            for row in subset.itertuples()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="downloading"):
            if fut.result():
                ok += 1

    print(f"Downloaded {ok}/{len(subset)} files")

    subset_csv = os.path.join(args.data_dir, "train_subset.csv")
    subset.to_csv(subset_csv, index=False)
    print(f"Wrote subset index to {subset_csv}")


if __name__ == "__main__":
    main()

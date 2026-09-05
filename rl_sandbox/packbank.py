"""
Real opening packs, extracted from 17lands data.

Modelling booster collation (rarity slots, bonus sheets, the-list slots, per-set
oddities) is work with nothing to show for it: the raw data already contains every pack
anyone opened. Rows with ``pick_number == 0`` are packs before anybody picked from them,
so they are exactly the distribution a simulated draft should deal from.

Packs are stored as card indices rather than count vectors - a FIN booster is 14
distinct cards, so 14 int16s beats a 359-wide row.

Basic lands are kept, mapped to a single pad index at ``num_cards``, matching the
convention in ``CollectionWinrateDataset``: a basic is a real pick that occupies a pick
slot but is not part of the collection. Dropping them instead would leave some boosters
with 13 cards and break the fixed 14-picks-per-booster structure.

Run from the repo root:  python rl_sandbox/packbank.py FIN
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASIC_NAMES = ["Forest", "Island", "Mountain", "Plains", "Swamp"]
PACK_FOLDER = "data/packs/"


def packbank_path(set_abbreviation, draft_mode, folder=PACK_FOLDER):
    return os.path.join(folder, f"{set_abbreviation}_{draft_mode}_packs.npz")


def build_packbank(
    set_abbreviation,
    draft_mode="Premier",
    data_folder="data/17lands/",
    folder=PACK_FOLDER,
    chunksize=200_000,
    max_packs_per_booster=100_000,
    overwrite=False,
):
    """Cache every observed opening pack for a set, split by booster number."""
    out_path = packbank_path(set_abbreviation, draft_mode, folder)
    if os.path.exists(out_path) and not overwrite:
        return out_path

    csv_path = os.path.join(
        data_folder, f"draft_data_public.{set_abbreviation}.{draft_mode}Draft.csv.gz"
    )
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    pack_columns = sorted(c for c in columns if c.startswith("pack_card_"))
    names = [c[len("pack_card_"):] for c in pack_columns]

    # Cardnames, and the index each pack column maps to. Basics all collapse onto the
    # pad slot at num_cards, so the simulator never has to special-case which basic.
    cardnames = [n for n in names if n not in BASIC_NAMES]
    num_cards = len(cardnames)
    card_index = {name: i for i, name in enumerate(cardnames)}
    column_to_index = np.array(
        [card_index.get(name, num_cards) for name in names], dtype=np.int64
    )

    collected = {0: [], 1: [], 2: []}
    reader = pd.read_csv(
        csv_path,
        usecols=["pack_number", "pick_number"] + pack_columns,
        chunksize=chunksize,
    )
    for chunk in reader:
        opens = chunk[chunk["pick_number"] == 0]
        if opens.empty:
            continue
        counts = opens[pack_columns].to_numpy(dtype=np.int16)
        booster = opens["pack_number"].to_numpy()
        for number in collected:
            rows = counts[booster == number]
            if len(rows) == 0 or sum(len(r) for r in collected[number]) >= max_packs_per_booster:
                continue
            # Every observed pack holds 14 distinct cards, so nonzero positions in row
            # order give the pack directly.
            pack_rows, pack_columns_hit = np.nonzero(rows)
            per_pack = np.bincount(pack_rows, minlength=len(rows))
            if per_pack.min() != per_pack.max():
                keep = per_pack == np.bincount(per_pack).argmax()
                rows = rows[keep]
                pack_rows, pack_columns_hit = np.nonzero(rows)
                per_pack = np.bincount(pack_rows, minlength=len(rows))
            collected[number].append(
                column_to_index[pack_columns_hit].reshape(len(rows), per_pack[0]).astype(np.int16)
            )

    os.makedirs(folder, exist_ok=True)
    payload = {"cardnames": np.array(cardnames)}
    for number, parts in collected.items():
        stacked = np.concatenate(parts)[:max_packs_per_booster]
        payload[f"booster_{number}"] = stacked
        print(f"  booster {number}: {len(stacked)} packs of {stacked.shape[1]} cards")
    np.savez_compressed(out_path, **payload)
    print(f"Wrote {out_path}")
    return out_path


class PackBank:
    """Samples real opening packs, by booster number."""

    def __init__(self, set_abbreviation, draft_mode="Premier", folder=PACK_FOLDER):
        path = packbank_path(set_abbreviation, draft_mode, folder)
        if not os.path.exists(path):
            build_packbank(set_abbreviation, draft_mode, folder=folder)
        payload = np.load(path, allow_pickle=True)
        self.cardnames = list(payload["cardnames"])
        self.num_cards = len(self.cardnames)
        self.boosters = [payload[f"booster_{i}"] for i in range(3)]
        self.pack_size = self.boosters[0].shape[1]
        self.n_boosters = len(self.boosters)

    def sample(self, rng, n, booster):
        """(n, pack_size) int64 card indices; ``num_cards`` is the basic-land pad."""
        source = self.boosters[booster]
        return source[rng.integers(0, len(source), size=n)].astype(np.int64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FIN")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_packbank(args.set_abbreviation, args.draft_mode, overwrite=args.overwrite)

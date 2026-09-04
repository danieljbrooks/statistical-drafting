"""
Predict the winrate of a draft pool.

Edit POOL below and run it from the repo root:

    python winrate_sandbox/predict_pool.py

Options:
    --model NAME     model to load from data/winrate_models/ (default: the FIN one)
    --per-card       show what each card is worth, by leaving it out one at a time
    --cards-file F   read the pool from a file (one cardname per line) instead of POOL

The model reads a *pool*, not a finished deck: it estimates the per-match win
probability of the deck this pool becomes. Partial pools work too - pass fewer cards and
the prediction regresses toward the population average, which is correct, since an
8-card pool genuinely says less than a finished one.
"""
import argparse
import difflib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statisticaldrafting as sd
from architectures import build_network

DEFAULT_MODEL = "FIN_Premier_diamond_dn_all_features"
MODEL_FOLDER = "data/winrate_models/"

# ---------------------------------------------------------------------------
# Edit this list. Duplicates are counted, basic lands are ignored.
# ---------------------------------------------------------------------------
POOL = [
    "Sephiroth, Fabled SOLDIER",
    "Dion, Bahamut's Dominant",
    "Traveling Chocobo",
    "Summon: Primal Odin",
    "Nibelheim Aflame",
    "Ardyn, the Usurper",
    "Adeline, Resplendent Cathar",
    "Sazh Katzroy",
    "Smuggler's Copter",
    "Cornered by Black Mages",
    "Cornered by Black Mages",
    "Black Mage's Rod",
    "Black Mage's Rod",
    "Dark Knight's Greatsword",
    "Red Mage's Rapier",
    "Vayne's Treachery",
    "Garland, Knight of Cornelia",
    "Choco-Comet",
    "Blazing Bomb",
    "Malboro",
    "Tonberry",
    "Sabotender",
    "Loporrit Scout",
]

# --- Sample pools from held-out FIN drafts, for calibration against real data. ---
#
# STRONG - the highest-rated pool in the FIN test split.
# Model says 0.724; this player actually went 6-3.
# POOL = [
#     "Adventurer's Airship", "Bard's Bow", "Black Mage's Rod", "Black Mage's Rod",
#     "Black Mage's Rod", "Blazing Bomb", "Call the Mountain Chocobo",
#     "Captain Lannery Storm", "Choco-Comet", "Circle of Power", "Circle of Power",
#     "Coral Sword", "Cornered by Black Mages", "Cornered by Black Mages",
#     "Dark Knight's Greatsword", "Dark Knight's Greatsword", "Fatal Push",
#     "Garland, Knight of Cornelia", "Garland, Knight of Cornelia", "Gladiolus Amicitia",
#     "Hecteyes", "Lion Heart", "Loporrit Scout", "Malboro", "Mysidian Elder",
#     "Namazu Trader", "Namazu Trader", "Nibelheim Aflame", "Qutrub Forayer",
#     "Red Mage's Rapier", "Red Mage's Rapier", "Reno and Rude",
#     "Resentful Revelation", "Rook Turret", "Sabotender", "Sephiroth's Intervention",
#     "Sidequest: Hunt the Mark", "Sorceress's Schemes", "Tonberry",
#     "Vayne's Treachery", "Vayne's Treachery",
# ]
#
# TYPICAL - a median pool. Model says 0.528; this player went 1-3, which is the
# point of the noise ceiling: one event of five matches says very little.
# POOL = [
#     "Adventurer's Airship", "Black Mage's Rod", "Black Mage's Rod", "Black Mage's Rod",
#     "Blitzball", "Blitzball Shot", "Brainstorm", "Choco-Comet", "Combat Tutorial",
#     "Combat Tutorial", "Cornered by Black Mages", "Dreams of Laguna",
#     "Evil Reawakened", "Gaius van Baelsar", "Galuf's Final Act",
#     "Garland, Knight of Cornelia", "Hill Gigas", "Ice Magic", "Iron Giant",
#     "Light of Judgment", "Loporrit Scout", "Magic Damper", "Malboro",
#     "Opera Love Song", "Opera Love Song", "Random Encounter", "Resentful Revelation",
#     "Sabotender", "Sahagin", "Scorpion Sentinel", "Sephiroth's Intervention",
#     "Shambling Cie'th", "Shantotto, Tactician Magician", "Summon: Anima", "Syncopate",
#     "The Lunar Whale", "Thunder Magic", "Treno, Dark City", "Undercity Dire Rat",
#     "Vector, Imperial Capital", "Vincent's Limit Break",
# ]
#
# WEAK - the lowest-rated pool in the test split: five colours, no direction.
# Model says 0.332; this player went 4-3 anyway.
# POOL = [
#     "Adventurer's Airship", "Airship Crash", "Ashe, Princess of Dalmasca",
#     "Battle Menu", "Black Waltz No. 3", "Blitzball Shot", "Captain Lannery Storm",
#     "Chocobo Racetrack", "Cid, Timeless Artificer", "Coeurl", "Crossroads Village",
#     "Fynn, the Fangbearer", "G'raha Tia", "Gaelicat", "Gigantoad",
#     "Goobbue Gardener", "Gysahl Greens", "Ice Flan", "Light of Judgment",
#     "Locke Cole", "Magic Damper", "Monk's Fist", "Noctis, Prince of Lucis",
#     "Paladin's Arms", "Paladin's Arms", "Rinoa Heartilly", "Sage's Nouliths",
#     "Serah Farron", "Shambling Cie'th", "Stiltzkin, Moogle Merchant",
#     "Summon: Fat Chocobo", "Summon: G.F. Cerberus", "Summon: G.F. Ifrit",
#     "Summon: Shiva", "Suplex", "The Fire Crystal", "Thief's Knife",
#     "Torgal, A Fine Hound", "Ultima, Origin of Oblivion", "You're Not Alone",
#     "Zack Fair",
# ]


def load_model(model_name, device):
    """Rebuild a trained winrate model from its checkpoint and JSON sidecar."""
    base = os.path.join(MODEL_FOLDER, model_name)
    if not os.path.exists(base + ".json"):
        available = sorted(
            f[:-5] for f in os.listdir(MODEL_FOLDER) if f.endswith(".json")
        )
        raise SystemExit(
            f"No model sidecar at {base}.json\n"
            f"Available: {', '.join(available) if available else '(none)'}\n"
            "Train one with: python winrate_sandbox/train_final_model.py FIN dn_all_features"
        )
    with open(base + ".json") as f:
        sidecar = json.load(f)
    network = build_network(
        sidecar["architecture"], sidecar["cardnames"], sidecar["picks_per_draft"],
        sidecar["base_winrate"], sidecar["dropout"], sidecar["input_dropout"],
        sidecar["set"], sidecar["draft_mode"],
    ).to(device)
    network.load_state_dict(torch.load(base + ".pt", map_location=device))
    network.eval()
    return network, sidecar


def build_collection(cards, cardnames):
    """Card list -> count vector. Reports unknown names instead of dropping them silently."""
    card_to_index = {name: i for i, name in enumerate(cardnames)}
    counts = np.zeros(len(cardnames), dtype=np.float32)
    n_basics, unknown = 0, []
    for card in cards:
        card = card.strip()
        if not card:
            continue
        if card in sd.BASIC_NAMES:
            n_basics += 1
            continue
        if card not in card_to_index:
            unknown.append(card)
            continue
        counts[card_to_index[card]] += 1
    if unknown:
        print(f"{len(unknown)} cardname(s) not in this set:")
        for card in unknown:
            close = difflib.get_close_matches(card, cardnames, n=2, cutoff=0.6)
            hint = f"  did you mean: {', '.join(close)}" if close else ""
            print(f"  {card!r}{hint}")
        raise SystemExit("Fix the cardnames above and rerun.")
    return counts, n_basics


@torch.no_grad()
def predict_winrate(network, counts, n_picks, device):
    """Predicted per-match win probability for one collection."""
    collection = torch.from_numpy(counts).unsqueeze(0).to(device)
    n_picks_tensor = torch.tensor([n_picks], dtype=torch.long, device=device)
    return float(torch.sigmoid(network(collection, n_picks_tensor))[0])


def percentile_of(value, percentiles):
    """Where this prediction sits among held-out pools, 0-100."""
    keys = sorted(int(k) for k in percentiles)
    values = [percentiles[str(k)] for k in keys]
    if value <= values[0]:
        return float(keys[0])
    if value >= values[-1]:
        return float(keys[-1])
    return float(np.interp(value, values, keys))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cards-file", default=None)
    parser.add_argument("--per-card", action="store_true")
    parser.add_argument(
        "--picks", type=int, default=None,
        help="Override how many picks the pool represents (default: one per card).",
    )
    args = parser.parse_args()

    cards = POOL
    if args.cards_file:
        with open(args.cards_file) as f:
            cards = [line for line in f.read().splitlines() if line.strip()]

    device = sd.get_device()
    network, sidecar = load_model(args.model, device)
    counts, n_basics = build_collection(cards, sidecar["cardnames"])

    n_cards = int(counts.sum())
    if n_cards == 0:
        raise SystemExit("Pool is empty.")
    picks_per_draft = sidecar["picks_per_draft"]
    n_picks = min(args.picks or (n_cards + n_basics), picks_per_draft)
    # Basics wheel to the last pick of a pack, so a completed draft often ends with a
    # few fewer real cards than picks. Treat those as complete rather than partial.
    is_complete = n_picks >= picks_per_draft - 3

    winrate = predict_winrate(network, counts, n_picks, device)
    expected_wins = float(sd.expected_match_wins(torch.tensor(winrate)))

    print(f"\nModel: {sidecar['name']}")
    print(
        f"  trained on {sidecar['set']} {sidecar['draft_mode']} "
        f"[{sidecar['rank_tag']}], population winrate "
        f"{sidecar['base_winrate']:.4f}"
    )
    print(f"Pool:  {n_cards} cards" + (f" (+{n_basics} basics ignored)" if n_basics else ""))
    if not is_complete:
        print(
            f"       partial pool - scored as {n_picks} of {picks_per_draft} picks, so "
            "the estimate is pulled toward the average"
        )

    print(f"\n  Predicted match winrate:  {winrate:.4f}")
    print(f"  Expected match wins:      {expected_wins:.2f}  (7-win / 3-loss event)")
    reference = sidecar.get("prediction_percentiles")
    if reference and is_complete:
        print(
            f"  Percentile vs held-out pools: {percentile_of(winrate, reference):.0f}"
            f"   (median pool {reference['50']:.4f})"
        )

    spread = sidecar.get("test_metrics", {}).get("skill_spread")
    if spread:
        print(
            f"\n  For scale: predictions have sd {spread:.3f}, and a single event of "
            f"~5.5 matches\n  has a binomial sd near 0.21, so any one result tells you "
            "little about the deck."
        )

    if args.per_card:
        print("\nWhat each card is worth (winrate change from removing it):")
        rows = []
        for index in np.nonzero(counts)[0]:
            reduced = counts.copy()
            reduced[index] -= 1
            without = predict_winrate(network, reduced, max(n_picks - 1, 1), device)
            rows.append((winrate - without, sidecar["cardnames"][index], int(counts[index])))
        for delta, name, count in sorted(rows, reverse=True):
            label = f"{name} x{count}" if count > 1 else name
            print(f"  {delta:+.4f}  {label}")


if __name__ == "__main__":
    main()

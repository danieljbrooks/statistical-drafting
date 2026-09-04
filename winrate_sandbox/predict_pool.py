"""
Predict the winrate of a draft pool.

Edit POOL below and run it from the repo root:

    python winrate_sandbox/predict_pool.py

Set the SET constant below (or pass --set) to choose which set's model to score with.
The cardnames in POOL have to belong to that set.

Options:
    --set ABBREV     three letter set code, e.g. FDN or FIN (default: the SET constant)
    --rank-tag TAG   which rank filter the model was trained with (default: diamond)
    --architecture A which architecture to load (default: dn_all_features)
    --model NAME     load an exact model name, overriding the three options above
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

MODEL_FOLDER = "data/winrate_models/"

# Which set to score against. Models are per set, so this has to match the cards in POOL.
SET = "FDN"
DRAFT_MODE = "Premier"
RANK_TAG = "diamond"
ARCHITECTURE = "dn_all_features"

# ---------------------------------------------------------------------------
# Edit this list. Duplicates are counted, basic lands are ignored.
# The cards must belong to the set named in SET above.
#
# Default is a typical completed FDN pool from the held-out split: all 42 picks, no
# basics taken, white-black. Model says 0.514 — dead on the median pool — and this
# player went 2-3.
# ---------------------------------------------------------------------------
POOL = [
    "Bake into a Pie", "Bigfin Bouncer",
    "Blasphemous Edict", "Burglar Rat",
    "Burglar Rat", "Campus Guide",
    "Cat Collector", "Claws Out",
    "Crypt Feaster", "Crypt Feaster",
    "Crypt Feaster", "Dazzling Angel",
    "Dazzling Angel", "Diregraf Ghoul",
    "Dismal Backwater", "Eaten Alive",
    "Evolving Wilds", "Fake Your Own Death",
    "Fanatical Firebrand", "Fiendish Panda",
    "Fleeting Flight", "Helpful Hunter",
    "Herald of Eternal Dawn", "Hungry Ghoul",
    "Hungry Ghoul", "Hungry Ghoul",
    "Inspiring Paladin", "Joust Through",
    "Luminous Rebuke", "Macabre Waltz",
    "Make Your Move", "Pilfer",
    "Prideful Parent", "Rogue's Passage",
    "Sanguine Syphoner", "Squad Rallier",
    "Sun-Blessed Healer", "Tranquil Cove",
    "Valkyrie's Call", "Vampire Soulcaller",
    "Wind-Scarred Crag", "Youthful Valkyrie",
]

# ===========================================================================
# More real pools from held-out drafts, for calibration against known records.
# FDN predictions span roughly 0.42 (5th percentile) to 0.60 (95th).
# ===========================================================================
#
# FDN STRONG — a focused blue-black tempo pool. Model says 0.617 (97th percentile);
# this player went 1-0 and stopped, which is why the record says so little.
# POOL = [
#     "Aegis Turtle", "Aegis Turtle", "Aegis Turtle",
#     "Arcane Epiphany", "Arcane Epiphany", "Archmage of Runes",
#     "Bake into a Pie", "Bake into a Pie", "Bigfin Bouncer",
#     "Bigfin Bouncer", "Broken Wings", "Dreadwing Scavenger",
#     "Essence Scatter", "Evolving Wilds", "Fake Your Own Death",
#     "Fleeting Distraction", "High Fae Trickster", "Hungry Ghoul",
#     "Hungry Ghoul", "Infernal Vessel", "Inspiration from Beyond",
#     "Inspiration from Beyond", "Involuntary Employment", "Micromancer",
#     "Perforating Artist", "Ravenous Amulet", "Refute",
#     "Refute", "Rune-Sealed Wall", "Searslicer Goblin",
#     "Secluded Courtyard", "Solemn Simulacrum", "Spectral Sailor",
#     "Spectral Sailor", "Spectral Sailor", "Stromkirk Bloodthief",
#     "Swiftwater Cliffs", "Tinybones, Bauble Burglar", "Tinybones, Bauble Burglar",
#     "Tranquil Cove", "Vanguard Seraph", "Witness Protection",
# ]
#
# FDN WEAK — four colours, no direction. Model says 0.412 (3rd percentile);
# this player went 1-3.
# POOL = [
#     "Bake into a Pie", "Banishing Light", "Battlesong Berserker",
#     "Battlesong Berserker", "Boltwave", "Burglar Rat",
#     "Burst Lightning", "Courageous Goblin", "Crypt Feaster",
#     "Crypt Feaster", "Dragon Trainer", "Eaten Alive",
#     "Eaten Alive", "Fanatical Firebrand", "Fanatical Firebrand",
#     "Felidar Savior", "Fiery Annihilation", "Firebrand Archer",
#     "Goblin Boarders", "Goblin Negotiation", "Goblin Surprise",
#     "Goblin Surprise", "Gorehorn Raider", "Gorehorn Raider",
#     "Gutless Plunderer", "Healer's Hawk", "Helpful Hunter",
#     "Infestation Sage", "Inspiring Paladin", "Inspiring Paladin",
#     "Involuntary Employment", "Juggernaut", "Kellan, Planar Trailblazer",
#     "Macabre Waltz", "Mocking Sprite", "Quick-Draw Katana",
#     "Slumbering Cerberus", "Sower of Chaos", "Stab",
#     "Think Twice", "Time Stop", "Youthful Valkyrie",
# ]
#
# FDN PARTIAL — a mid-draft collection. Fewer cards means the estimate is pulled
# toward the population average, which is correct.
# POOL = [
#     "Bake into a Pie", "Crypt Feaster", "Crypt Feaster", "Hungry Ghoul",
#     "Hungry Ghoul", "Dazzling Angel", "Inspiring Paladin", "Prideful Parent",
#     "Eaten Alive", "Pilfer", "Burglar Rat", "Diregraf Ghoul",
# ]
#
# ---------------------------------------------------------------------------
# FIN pools — set SET = "FIN" above, or run with --set FIN.
# These are 41 cards, not 42: in each, one pick was a basic land, which is a real
# pick but not part of the collection.
# ---------------------------------------------------------------------------
#
# FIN TYPICAL — model says 0.528; this player went 1-3.
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
# FIN STRONG — the highest-rated pool in the FIN test split. Model says 0.724;
# this player went 6-3.
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


def build_collection(cards, cardnames, set_abbreviation=None):
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
        recognized = len(cards) - len(unknown) - n_basics
        if recognized == 0 and len(unknown) > 3:
            # Every card missed: almost certainly the pool belongs to a different set.
            raise SystemExit(
                f"None of these {len(unknown)} cards are in "
                f"{set_abbreviation or 'this set'}. The pool is probably from another "
                "set — set SET at the top of this file, or pass --set."
            )
        print(f"{len(unknown)} cardname(s) not in this set:")
        for card in unknown[:10]:
            close = difflib.get_close_matches(card, cardnames, n=2, cutoff=0.6)
            hint = f"  did you mean: {', '.join(close)}" if close else ""
            print(f"  {card!r}{hint}")
        if len(unknown) > 10:
            print(f"  ... and {len(unknown) - 10} more")
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
    parser.add_argument("--set", dest="set_abbreviation", default=SET)
    parser.add_argument("--draft-mode", default=DRAFT_MODE)
    parser.add_argument("--rank-tag", default=RANK_TAG)
    parser.add_argument("--architecture", default=ARCHITECTURE)
    parser.add_argument(
        "--model", default=None,
        help="Exact model name, overriding --set / --draft-mode / --rank-tag / --architecture.",
    )
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

    model_name = args.model or (
        f"{args.set_abbreviation}_{args.draft_mode}_{args.rank_tag}_{args.architecture}"
    )
    device = sd.get_device()
    network, sidecar = load_model(model_name, device)
    counts, n_basics = build_collection(
        cards, sidecar["cardnames"], sidecar["set"]
    )

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

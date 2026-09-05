"""
Print actual decks the policy drafts, built from its pools by IterativeDeckBuilder.

Samples are taken in draft order, not selected by score, so they are representative
rather than flattering; each deck reports where it falls in the distribution of the
run. A DraftNet deck from the same pod is printed alongside for contrast - it saw the
same packs from a different seat.

Run from the repo root:
    python rl_sandbox/sample_decks.py FDN --policy <path> --reward dn_all_features
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "winrate_sandbox"))

import statisticaldeckbuild as sdb
import statisticaldrafting as sd
from colour_distribution import Adapter
from draft_env import N_SEATS, DraftEnv, DraftNetPolicy, run_drafts
from packbank import PackBank
from run_env_check import load_winrate_model
from train_policy import PickPolicy, score_pools

MONO = ["W", "U", "B", "R", "G"]
ORDER = MONO + ["Multicolor", "Colorless"]


def pool_list(counts, cardnames):
    out = []
    for index in np.nonzero(counts)[0]:
        out.extend([cardnames[index]] * int(counts[index]))
    return out


def show_deck(title, deck_counts, colour_of, rarity_of, note=""):
    by_colour = defaultdict(list)
    for name, n in sorted(deck_counts.items()):
        by_colour[colour_of.get(name, "Colorless")].append((n, name))
    mono_counts = {c: sum(n for n, _ in by_colour.get(c, [])) for c in MONO}
    total = sum(sum(n for n, _ in v) for v in by_colour.values())
    colours = [c for c in MONO if mono_counts[c] >= 2]
    print(f"\n  {title}   {total} spells   colours: {''.join(colours) if colours else '-'} "
          f"({len(colours)}c)   {note}")
    print(f"    " + "  ".join(f"{c}:{mono_counts[c]}" for c in MONO)
          + f"  gold:{sum(n for n, _ in by_colour.get('Multicolor', []))}"
          + f"  colourless:{sum(n for n, _ in by_colour.get('Colorless', []))}")
    for colour in ORDER:
        entries = by_colour.get(colour)
        if not entries:
            continue
        print(f"    {colour:<11}" + ", ".join(
            f"{n}x {name}" if n > 1 else name for n, name in sorted(entries, key=lambda e: e[1])
        ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FDN")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument("--rank-tag", default="diamond")
    parser.add_argument("--reward", default="dn_all_features")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--n-drafts", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    bank = PackBank(args.set_abbreviation, args.draft_mode)
    num_cards, picks = bank.num_cards, bank.pack_size * bank.n_boosters
    rewards = {
        name: load_winrate_model(args.set_abbreviation, args.draft_mode, args.rank_tag, name)[0]
        for name in ["mlp_400x2", "dn_all_features"]
    }
    draftnet = sd.load_draftnet(
        bank.cardnames, args.set_abbreviation, args.draft_mode, "data/models/"
    )
    policy = PickPolicy(bank.cardnames, dropout=0.1)
    policy.load_state_dict(torch.load(args.policy, weights_only=True))
    policy.eval()

    cards = pd.read_csv(f"data/cards/{args.set_abbreviation}.csv").set_index("name")
    colour_of = cards["color_identity"].fillna("Colorless").to_dict()
    rarity_of = cards["rarity"].fillna("common").to_dict()

    env = DraftEnv(bank, n_drafts=args.n_drafts, seed=args.seed)
    run_drafts(env, [Adapter(policy)] + [DraftNetPolicy(draftnet, 0.5)] * (N_SEATS - 1))
    agent_pools = env.collections()[:, 0].numpy()
    rival_pools = env.collections()[:, 1].numpy()

    agent_scores = score_pools(rewards[args.reward], torch.from_numpy(agent_pools), picks).numpy()
    builder = sdb.IterativeDeckBuilder(
        args.set_abbreviation, args.draft_mode,
        model_folder="data/models/", cards_folder="data/cards/",
    )

    print(f"{args.set_abbreviation}: decks drafted by {os.path.basename(args.policy)}")
    print(f"seat 0 vs 7 DraftNet bots, {args.n_drafts} drafts, showing the first {args.n_samples}")
    print(f"reward={args.reward}   run mean={agent_scores.mean():.4f}  sd={agent_scores.std():.4f}")

    for i in range(args.n_samples):
        percentile = 100 * (agent_scores < agent_scores[i]).mean()
        built = builder.build_deck(pool_list(agent_pools[i], bank.cardnames))
        deck_vector = torch.zeros(1, num_cards)
        for name, n in built["deck_counts"].items():
            deck_vector[0, bank.cardnames.index(name)] = n
        scores = {
            k: score_pools(m, deck_vector, picks).item() for k, m in rewards.items()
        }
        note = (f"pool {agent_scores[i]:.3f} (p{percentile:.0f})   "
                f"deck: mlp {scores['mlp_400x2']:.3f} / dn_feat {scores['dn_all_features']:.3f}")
        show_deck(f"RL DECK {i + 1}", built["deck_counts"], colour_of, rarity_of, note)

    rival = builder.build_deck(pool_list(rival_pools[0], bank.cardnames))
    show_deck("DRAFTNET DECK (seat 1, same pod as RL deck 1)",
              rival["deck_counts"], colour_of, rarity_of)


if __name__ == "__main__":
    main()

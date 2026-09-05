"""
What did the RL policy actually learn?

Five views, each chosen because it separates real drafting skill from reward hacking:

  p1p1 order      what the policy takes from a full pack with an empty pool, and how
                  that ranking compares to DraftNet's. Agreement means it rediscovered
                  human card evaluation; disagreement is where to look.
  colour          how concentrated its pools are in two colours. This is the first thing
                  a drafter learns and the easiest to check.
  curve of skill  reward-model winrate of its pools, in self-play and seated against a
                  DraftNet field.
  over-taken      cards it takes far more often than any real pool holds them - the axis
                  the greedy-value probe hacked the reward on.
  cross-model     whether a winrate model that was not the reward agrees the pools are good.

Run from the repo root:
    python rl_sandbox/inspect_policy.py FDN --policy data/rl_models/FDN_Premier_tabularasa_selfplay.pt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "winrate_sandbox"))

import statisticaldrafting as sd
from draft_env import N_SEATS, DraftEnv, DraftNetPolicy, run_drafts
from packbank import PackBank
from run_env_check import load_winrate_model, score
from train_policy import PickPolicy, play, score_pools

MONO = ["W", "U", "B", "R", "G"]


class PolicyAdapter:
    """Wraps a PickPolicy so run_drafts can seat it alongside DraftNetPolicy."""

    def __init__(self, policy, temperature=0.0):
        self.policy = policy.eval()
        self.temperature = temperature

    @torch.no_grad()
    def pick(self, pools, packs):
        logits = self.policy.logits(pools, packs)
        if self.temperature <= 0:
            return logits.argmax(dim=1)
        return torch.distributions.Categorical(logits=logits / self.temperature).sample()


@torch.no_grad()
def pick_order(network, cardnames, is_draftnet=False):
    """Score every card from an empty pool with everything available: the p1p1 order."""
    pool = torch.zeros((1, len(cardnames)))
    pack = torch.ones((1, len(cardnames)))
    if is_draftnet:
        network.eval()
        import torch.nn.functional as F
        x = pool
        for layer, norm in zip(network.hidden_layers, network.norms):
            x = norm(network.dropout_layer(F.gelu(layer(x))))
        return network.output_layer(x).squeeze(0).numpy()
    return network.logits(pool, pack).squeeze(0).numpy()


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def colour_profile(collections, colour):
    counts = collections.numpy()
    mass = np.stack([counts[:, colour == c].sum(1) for c in MONO], axis=1)
    top2 = np.sort(mass, axis=1)[:, -2:].sum(1)
    return {
        "top2_colours": top2.mean(),
        "off_colour": np.sort(mass, axis=1)[:, :3].sum(1).mean(),
        "multicolour": counts[:, colour == "Multicolor"].sum(1).mean(),
        "colourless": counts[:, colour == "Colorless"].sum(1).mean(),
        "distinct": (counts > 0).sum(1).mean(),
        "max_copies": counts.max(1).mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FDN")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument("--rank-tag", default="diamond")
    parser.add_argument("--reward", default="mlp_400x2")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--n-drafts", type=int, default=400)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    bank = PackBank(args.set_abbreviation, args.draft_mode)
    num_cards, picks = bank.num_cards, bank.pack_size * bank.n_boosters
    reward, _ = load_winrate_model(
        args.set_abbreviation, args.draft_mode, args.rank_tag, args.reward
    )
    draftnet = sd.load_draftnet(
        bank.cardnames, args.set_abbreviation, args.draft_mode, "data/models/"
    )
    policy_path = args.policy or (
        f"data/rl_models/{args.set_abbreviation}_{args.draft_mode}_tabularasa_selfplay.pt"
    )
    policy = PickPolicy(bank.cardnames, dropout=args.dropout)
    policy.load_state_dict(torch.load(policy_path, weights_only=True))
    policy.eval()

    cards = pd.read_csv(f"data/cards/{args.set_abbreviation}.csv").set_index("name")
    cards = cards.reindex(bank.cardnames)
    colour = cards["color_identity"].fillna("Colorless").to_numpy()
    rarity = cards["rarity"].fillna("common").to_numpy()

    print(f"{args.set_abbreviation} {args.draft_mode}   policy={os.path.basename(policy_path)}")
    print(f"reward={args.reward}   {num_cards} cards\n")

    # --- 1. p1p1 order -----------------------------------------------------
    rl_scores = pick_order(policy, bank.cardnames)
    dn_scores = pick_order(draftnet, bank.cardnames, is_draftnet=True)
    print(f"p1p1 rank correlation with DraftNet: {spearman(rl_scores, dn_scores):+.3f}")
    print("\nRL policy's top 15 first picks        | DraftNet's top 15")
    rl_top = np.argsort(-rl_scores)[:15]
    dn_top = np.argsort(-dn_scores)[:15]
    for i in range(15):
        left = f"{bank.cardnames[rl_top[i]][:28]:<28} [{rarity[rl_top[i]][:4]},{colour[rl_top[i]][:4]}]"
        right = f"{bank.cardnames[dn_top[i]][:28]:<28} [{rarity[dn_top[i]][:4]}]"
        print(f"  {i + 1:>2}. {left}  | {right}")

    # --- 2. pools it produces ----------------------------------------------
    runs = {}
    dataset = torch.load(
        sd.winrate_dataset_path(args.set_abbreviation, args.draft_mode, "test", args.rank_tag),
        weights_only=False,
    )
    sequences = torch.from_numpy(dataset.pick_sequences).long()
    runs["real_pools"] = sd.build_collections(
        sequences, torch.full((len(sequences),), picks), num_cards
    )

    env = DraftEnv(bank, n_drafts=args.n_drafts, seed=4242)
    run_drafts(env, [DraftNetPolicy(draftnet, 0.5)] * N_SEATS)
    runs["draftnet_x8"] = env.collections().reshape(-1, num_cards)

    env = DraftEnv(bank, n_drafts=args.n_drafts, seed=4242)
    play(env, policy, temperature=0.0)
    runs["rl_x8"] = env.collections().reshape(-1, num_cards)

    env = DraftEnv(bank, n_drafts=args.n_drafts, seed=4242)
    opponent = DraftNetPolicy(draftnet, 0.5)
    run_drafts(env, [PolicyAdapter(policy)] + [opponent] * (N_SEATS - 1))
    runs["rl_vs_draftnet"] = env.collections()[:, 0]

    print(f"\n{'pool source':<18}{'reward_wr':>11}{'top2col':>9}{'offcol':>8}"
          f"{'multi':>7}{'clrless':>9}{'distinct':>10}{'maxcopy':>9}")
    print("-" * 81)
    for name, collections in runs.items():
        values = score_pools(reward, collections, picks).numpy()
        profile = colour_profile(collections, colour)
        print(f"{name:<18}{values.mean():>11.4f}{profile['top2_colours']:>9.1f}"
              f"{profile['off_colour']:>8.1f}{profile['multicolour']:>7.1f}"
              f"{profile['colourless']:>9.1f}{profile['distinct']:>10.1f}"
              f"{profile['max_copies']:>9.2f}")

    # --- 3. what it over-takes ---------------------------------------------
    real_frequency = runs["real_pools"].numpy().mean(0)
    for label in ["rl_x8", "rl_vs_draftnet"]:
        frequency = runs[label].numpy().mean(0)
        excess = frequency - real_frequency
        print(f"\n{label}: cards taken most in excess of real pools")
        for i in np.argsort(-excess)[:10]:
            print(f"    {frequency[i]:5.2f} vs {real_frequency[i]:5.2f} real   "
                  f"{bank.cardnames[i][:34]:<34} [{rarity[i]}, {colour[i]}]")

    # --- 4. cross-model check ----------------------------------------------
    others = {}
    for architecture in ["linear", "dn_all_features"]:
        try:
            others[architecture] = load_winrate_model(
                args.set_abbreviation, args.draft_mode, args.rank_tag, architecture
            )[0]
        except FileNotFoundError:
            pass
    if others:
        print(f"\nCross-model check (mean predicted winrate):")
        print(f"{'pool source':<18}{args.reward:>14}" + "".join(f"{a:>18}" for a in others))
        for name, collections in runs.items():
            line = f"{name:<18}{score(reward, collections, picks).mean():>14.4f}"
            for other in others.values():
                line += f"{score(other, collections, picks).mean():>18.4f}"
            print(line)


if __name__ == "__main__":
    main()

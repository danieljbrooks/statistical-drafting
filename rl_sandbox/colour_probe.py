"""
Why did the tabula-rasa policy never learn colour discipline?

Two hypotheses:
  (a) The reward does not value colour coherence, so there was nothing to learn.
  (b) The reward does value it, but a single-action deviation cannot reveal it. Staying
      on colour only pays if the *rest* of the draft also stays on colour, and the
      rollouts that estimate Q are run under the current, colour-blind policy. The
      benefit is invisible to the advantage estimate, so it is never learned.

The test: take the trained policy and force it onto two colours after the first few
picks, changing nothing else. If the reward goes up, the reward valued colour discipline
all along and (b) is the explanation.

Run from the repo root:
    python rl_sandbox/colour_probe.py FDN --policy data/rl_models/FDN_Premier_tabularasa_selfplay.pt
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
from run_env_check import load_winrate_model
from train_policy import PickPolicy, play, score_pools

MONO = ["W", "U", "B", "R", "G"]
NEG_INF = float("-inf")


class TwoColourPolicy:
    """
    The given policy, restricted to its pool's two strongest colours after ``commit_at``
    picks. Colourless and multicolour cards are always allowed; if nothing on-colour is
    in the pack, the restriction is dropped for that pick.
    """

    def __init__(self, policy, colour_matrix, always_ok, commit_at=8):
        self.policy = policy.eval()
        self.colour_matrix = colour_matrix  # (5, num_cards) float
        self.always_ok = always_ok  # (num_cards,) bool
        self.commit_at = commit_at

    @torch.no_grad()
    def pick(self, pools, packs):
        logits = self.policy.logits(pools, packs)
        n_picks = pools.sum(dim=1)
        # Colour mass currently in each pool, and its top two colours.
        mass = pools @ self.colour_matrix.T  # (batch, 5)
        top2 = mass.topk(2, dim=1).indices
        allowed = self.always_ok.unsqueeze(0).expand_as(logits).clone()
        for slot in range(2):
            allowed |= self.colour_matrix[top2[:, slot]] > 0
        # Only bind once the pool actually says something about colour, and never bind
        # a seat into a pack that holds nothing it is allowed to take.
        restrict = (n_picks >= self.commit_at).unsqueeze(1)
        restrict = restrict & ((packs > 0) & allowed).any(dim=1, keepdim=True)
        return logits.masked_fill(restrict & ~allowed, NEG_INF).argmax(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FDN")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument("--rank-tag", default="diamond")
    parser.add_argument("--reward", default="mlp_400x2")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--n-drafts", type=int, default=400)
    parser.add_argument("--commit-at", type=int, default=8)
    args = parser.parse_args()

    bank = PackBank(args.set_abbreviation, args.draft_mode)
    num_cards, picks = bank.num_cards, bank.pack_size * bank.n_boosters
    reward, _ = load_winrate_model(
        args.set_abbreviation, args.draft_mode, args.rank_tag, args.reward
    )
    others = {}
    for architecture in ["linear", "dn_all_features"]:
        try:
            others[architecture] = load_winrate_model(
                args.set_abbreviation, args.draft_mode, args.rank_tag, architecture
            )[0]
        except FileNotFoundError:
            pass
    draftnet = sd.load_draftnet(
        bank.cardnames, args.set_abbreviation, args.draft_mode, "data/models/"
    )
    policy = PickPolicy(bank.cardnames, dropout=0.1)
    policy.load_state_dict(torch.load(
        args.policy or f"data/rl_models/{args.set_abbreviation}_{args.draft_mode}_tabularasa_selfplay.pt",
        weights_only=True,
    ))
    policy.eval()

    cards = pd.read_csv(f"data/cards/{args.set_abbreviation}.csv").set_index("name")
    cards = cards.reindex(bank.cardnames)
    colour = cards["color_identity"].fillna("Colorless").to_numpy()
    colour_matrix = torch.tensor(
        np.stack([(colour == c).astype(np.float32) for c in MONO])
    )
    always_ok = torch.tensor(np.isin(colour, ["Colorless", "Multicolor"]))

    def profile(collections):
        counts = collections.numpy()
        mass = np.stack([counts[:, colour == c].sum(1) for c in MONO], axis=1)
        return np.sort(mass, axis=1)[:, -2:].sum(1).mean()

    results = {}

    env = DraftEnv(bank, n_drafts=args.n_drafts, seed=7000)
    play(env, policy, temperature=0.0)
    results["rl_x8 (as trained)"] = env.collections().reshape(-1, num_cards)

    forced = TwoColourPolicy(policy, colour_matrix, always_ok, args.commit_at)
    env = DraftEnv(bank, n_drafts=args.n_drafts, seed=7000)
    run_drafts(env, [forced] * N_SEATS)
    results["rl_x8 + forced 2c"] = env.collections().reshape(-1, num_cards)

    opponent = DraftNetPolicy(draftnet, 0.5)
    for label, seat0 in [("rl_vs_draftnet", policy), ("rl_vs_draftnet + forced 2c", forced)]:
        env = DraftEnv(bank, n_drafts=args.n_drafts, seed=7000)
        agent = seat0 if isinstance(seat0, TwoColourPolicy) else _Adapter(seat0)
        run_drafts(env, [agent] + [opponent] * (N_SEATS - 1))
        results[label] = env.collections()[:, 0]

    print(f"{args.set_abbreviation}  policy={os.path.basename(args.policy or 'default')}  "
          f"commit_at=pick {args.commit_at}\n")
    header = f"{'configuration':<28}{args.reward:>12}{'top2col':>9}"
    header += "".join(f"{a:>18}" for a in others)
    print(header)
    print("-" * len(header))
    for name, collections in results.items():
        line = f"{name:<28}{score_pools(reward, collections, picks).mean():>12.4f}{profile(collections):>9.1f}"
        for other in others.values():
            line += f"{score_pools(other, collections, picks).mean():>18.4f}"
        print(line)


class _Adapter:
    def __init__(self, policy):
        self.policy = policy.eval()

    @torch.no_grad()
    def pick(self, pools, packs):
        return self.policy.logits(pools, packs).argmax(dim=1)


if __name__ == "__main__":
    main()

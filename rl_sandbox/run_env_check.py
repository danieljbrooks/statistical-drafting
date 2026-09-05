"""
Validate the simulator, then probe whether the reward survives being optimized in it.

Two questions, in order:

1. Is the environment right? Drafts played by DraftNet - the model fit to the humans who
   generated the data - should produce pools whose predicted winrate distribution looks
   like real held-out pools. If simulated pools score much higher, either the packs, the
   wheel, or the opponents are wrong, and every result downstream is fiction.

2. Does pack availability constrain reward hacking? ``tabula_rasa_check.py`` shows the
   reward can be pushed to 0.91 by a pool with a legal rarity mix, but that pool holds
   4 copies of each of 8 commons, which no drafter could ever open. GreedyValuePolicy
   maximizes the same reward while restricted to cards actually in its packs. Where it
   lands is the answer: near real pools means the environment is the regularizer that
   model ensembling was not; far above them means the reward is hacked even here.

Run from the repo root:  python rl_sandbox/run_env_check.py FIN
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "winrate_sandbox"))

import statisticaldrafting as sd
from architectures import build_network
from draft_env import (
    N_SEATS, DraftEnv, DraftNetPolicy, GreedyValuePolicy, RandomPolicy, run_drafts,
)
from packbank import PackBank

MODEL_FOLDER = "data/winrate_models/"


def load_winrate_model(set_abbreviation, draft_mode, rank_tag, architecture):
    name = f"{set_abbreviation}_{draft_mode}_{rank_tag}_{architecture}"
    with open(os.path.join(MODEL_FOLDER, name + ".json")) as f:
        meta = json.load(f)
    network = build_network(
        architecture,
        cardnames=meta["cardnames"],
        picks_per_draft=meta["picks_per_draft"],
        base_winrate=meta["base_winrate"],
        dropout=meta["dropout"],
        input_dropout=meta["input_dropout"],
        set_abbreviation=meta["set"],
        draft_mode=meta["draft_mode"],
    )
    network.load_state_dict(
        torch.load(os.path.join(MODEL_FOLDER, name + ".pt"), weights_only=True)
    )
    return network.eval(), meta


@torch.no_grad()
def score(network, collections, picks_per_draft=42, batch=2048):
    out = []
    for i in range(0, collections.shape[0], batch):
        chunk = collections[i : i + batch].float()
        n_picks = torch.full((chunk.shape[0],), picks_per_draft, dtype=torch.long)
        out.append(torch.sigmoid(network(chunk, n_picks)))
    return torch.cat(out).numpy()


def describe(collections):
    """Concentration stats - the axis the hill-climbed exploit pools ran off."""
    counts = collections.numpy()
    return {
        "distinct": (counts > 0).sum(1).mean(),
        "max_copies": counts.max(1).mean(),
        "top8_mass": np.sort(counts, axis=1)[:, -8:].sum(1).mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FIN")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument("--rank-tag", default="diamond")
    parser.add_argument("--reward", default="mlp_400x2")
    parser.add_argument("--n-drafts", type=int, default=1000)
    parser.add_argument("--n-drafts-greedy", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()

    bank = PackBank(args.set_abbreviation, args.draft_mode)
    reward, meta = load_winrate_model(
        args.set_abbreviation, args.draft_mode, args.rank_tag, args.reward
    )
    assert list(meta["cardnames"]) == bank.cardnames, "pack bank and reward model disagree on cards"

    draftnet = sd.load_draftnet(
        bank.cardnames, args.set_abbreviation, args.draft_mode, "data/models/"
    )

    # Held-out models, never used as the reward, for the cross-check.
    others = {}
    for architecture in ["linear", "dn_all_features"]:
        try:
            others[architecture] = load_winrate_model(
                args.set_abbreviation, args.draft_mode, args.rank_tag, architecture
            )[0]
        except FileNotFoundError:
            pass

    runs = {}

    # Real held-out pools, for reference.
    dataset = torch.load(
        sd.winrate_dataset_path(args.set_abbreviation, args.draft_mode, "test", args.rank_tag),
        weights_only=False,
    )
    sequences = torch.from_numpy(dataset.pick_sequences).long()
    runs["real_pools"] = sd.build_collections(
        sequences, torch.full((len(sequences),), 42), bank.num_cards
    )

    configurations = {
        "random_x8": (args.n_drafts, [RandomPolicy(s) for s in range(N_SEATS)], None),
        "draftnet_x8": (
            args.n_drafts,
            [DraftNetPolicy(draftnet, args.temperature)] * N_SEATS,
            None,
        ),
        "greedyV_vs_draftnet": (
            args.n_drafts_greedy,
            None,  # built below, needs the shared opponent object for batching
            "mixed",
        ),
        "greedyV_x8": (
            args.n_drafts_greedy,
            [GreedyValuePolicy(reward, bank.pack_size * bank.n_boosters)] * N_SEATS,
            None,
        ),
    }

    for name, (n_drafts, policies, mode) in configurations.items():
        if mode == "mixed":
            opponent = DraftNetPolicy(draftnet, args.temperature)
            policies = [GreedyValuePolicy(reward, bank.pack_size * bank.n_boosters)]
            policies += [opponent] * (N_SEATS - 1)
        env = DraftEnv(bank, n_drafts=n_drafts, seed=11)
        run_drafts(env, policies)
        collections = env.collections()
        # For mixed configurations only seat 0 is the agent; elsewhere pool every seat.
        runs[name] = collections[:, 0] if mode == "mixed" else collections.reshape(-1, bank.num_cards)
        print(f"  ran {name}: {n_drafts} drafts", flush=True)

    print(f"\n{args.set_abbreviation} {args.draft_mode} [{args.rank_tag}]  reward={args.reward}")
    print(f"opponent DraftNet temperature={args.temperature}\n")
    header = f"{'pool source':<22}{'n':>6}{'mean':>8}{'sd':>7}{'p5':>7}{'p95':>7}{'max':>7}"
    header += f"{'distinct':>10}{'maxcopy':>9}{'top8':>7}"
    print(header)
    print("-" * len(header))
    for name, collections in runs.items():
        values = score(reward, collections)
        stats = describe(collections)
        print(
            f"{name:<22}{len(collections):>6}{values.mean():>8.4f}{values.std():>7.4f}"
            f"{np.percentile(values, 5):>7.4f}{np.percentile(values, 95):>7.4f}{values.max():>7.4f}"
            f"{stats['distinct']:>10.1f}{stats['max_copies']:>9.2f}{stats['top8_mass']:>7.1f}"
        )

    if others:
        print(f"\nCross-model check (mean predicted winrate):")
        print(f"{'pool source':<22}{args.reward:>14}" + "".join(f"{a:>18}" for a in others))
        for name, collections in runs.items():
            line = f"{name:<22}{score(reward, collections).mean():>14.4f}"
            for architecture, other in others.items():
                line += f"{score(other, collections).mean():>18.4f}"
            print(line)


if __name__ == "__main__":
    main()

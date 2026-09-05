"""
Where does a fine-tuned policy actually differ from DraftNet?

At rho = 0.986 on pack-1 pick-1 the two agree almost perfectly on context-free card
power, so whatever the reinforcement learning changed lives somewhere else. This walks
the policy through real drafts and, at every state it reaches, asks both policies what
they would take - so the comparison is over states the agent actually visits rather than
over a synthetic ranking.

Reported:
  agreement by pick number   where in the draft the disagreements are concentrated
  colour commitment curve    how fast each policy narrows to two colours
  the disagreements          when they differ, is the agent taking the on-colour card,
                             the rarer card, the card DraftNet rates higher?
  p1p1 movers                the few cards whose first-pick rank did move
  contextual preferences     cards taken far more or less often than DraftNet takes
                             them, given the same opportunities

Run from the repo root:
    python rl_sandbox/policy_diff.py FDN --policy <path>
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
from draft_env import N_SEATS, DraftEnv, DraftNetPolicy
from inspect_policy import pick_order, spearman
from packbank import PackBank
from run_env_check import load_winrate_model
from train_policy import PickPolicy

MONO = ["W", "U", "B", "R", "G"]


class _GreedyAdapter:
    """Gives a PickPolicy the same .pick() interface as DraftNetPolicy."""

    def __init__(self, policy):
        self.policy = policy.eval()

    @torch.no_grad()
    def pick(self, pools, packs):
        return self.policy.logits(pools, packs).argmax(dim=1)


@torch.no_grad()
def run_and_compare(policy, draftnet_policy, draftnet_raw, bank, n_drafts, seed, temperature):
    """
    Play drafts with ``policy`` in seat 0, recording at every seat-0 state what both it
    and DraftNet would take.
    """
    num_cards = bank.num_cards
    env = DraftEnv(bank, n_drafts=n_drafts, seed=seed)
    records = []
    while not env.done:
        pools = env.pools[:, :, :num_cards]
        packs = env.packs[:, :, :num_cards]
        empty = ~(packs > 0).any(dim=2, keepdim=True)
        offered = packs + empty.to(packs.dtype)

        agent_pick = policy.logits(pools[:, 0], offered[:, 0]).argmax(dim=1)
        rival_pick = draftnet_policy.pick(pools[:, 0], offered[:, 0])
        records.append({
            "t": env.t,
            "pool": pools[:, 0].clone(),
            "pack": packs[:, 0].clone(),
            "agent": agent_pick.clone(),
            "rival": rival_pick.clone(),
            "legal": int((packs[:, 0] > 0).sum(dim=1).min().item()),
        })

        actions = torch.zeros((env.n_drafts, N_SEATS), dtype=torch.long)
        actions[:, 0] = agent_pick
        rest_pools = pools[:, 1:].reshape(-1, num_cards)
        rest_packs = offered[:, 1:].reshape(-1, num_cards)
        actions[:, 1:] = draftnet_policy.pick(rest_pools, rest_packs).view(
            env.n_drafts, N_SEATS - 1
        )
        env.step(actions)
    return env, records


def colour_mass(counts, colour):
    return np.stack([counts[:, colour == c].sum(1) for c in MONO], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FDN")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument("--rank-tag", default="topplayer")
    parser.add_argument("--reward", default="dn_all_features_colour")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--baseline", default="draftnet",
                        help='What to compare against: "draftnet", or a path to another '
                             'PickPolicy checkpoint. States are those the --policy agent '
                             'visits, so the question is always "what would the baseline '
                             'have done here".')
    parser.add_argument("--n-drafts", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()

    bank = PackBank(args.set_abbreviation, args.draft_mode)
    num_cards = bank.num_cards
    draftnet = sd.load_draftnet(
        bank.cardnames, args.set_abbreviation, args.draft_mode, "data/models/"
    )
    policy = PickPolicy(bank.cardnames, dropout=0.1)
    policy.load_state_dict(torch.load(args.policy, weights_only=True))
    policy.eval()
    # The seven opponents always play DraftNet at temperature, so both agents face the
    # same field; only the thing being compared against changes.
    if args.baseline == "draftnet":
        baseline = DraftNetPolicy(draftnet, 0.0)
        baseline_name = "DraftNet"
    else:
        other = PickPolicy(bank.cardnames, dropout=0.1)
        other.load_state_dict(torch.load(args.baseline, weights_only=True))
        other.eval()
        baseline = _GreedyAdapter(other)
        baseline_name = os.path.basename(args.baseline).replace(".pt", "")
    greedy = baseline

    cards = pd.read_csv(f"data/cards/{args.set_abbreviation}.csv").set_index("name")
    cards = cards.reindex(bank.cardnames)
    colour = cards["color_identity"].fillna("Colorless").to_numpy()
    rarity = cards["rarity"].fillna("common").to_numpy()
    rarity_rank = pd.Series(rarity).map(
        {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3, "special": 2}
    ).fillna(0).to_numpy()

    env, records = run_and_compare(
        policy, greedy, draftnet, bank, args.n_drafts, seed=246810, temperature=args.temperature
    )
    # Opponents are DraftNet regardless; only seat 0's comparison target changes.

    dn_scores = pick_order(draftnet, bank.cardnames, is_draftnet=True)
    rl_scores = pick_order(policy, bank.cardnames)
    dn_rank = pd.Series(-dn_scores).rank().to_numpy()
    rl_rank = pd.Series(-rl_scores).rank().to_numpy()

    print(f"{args.set_abbreviation}  policy={os.path.basename(args.policy)}")
    print(f"compared against: {baseline_name}")
    print(f"seat 0 vs 7 DraftNet (T={args.temperature}), {args.n_drafts} drafts; both "
          f"policies queried at every state the --policy agent reaches")
    print(f"p1p1 Spearman with DraftNet: {spearman(rl_scores, dn_scores):+.3f}\n")

    # --- agreement through the draft ---------------------------------------
    print(f"Agreement with {baseline_name}, by pick number")
    print(f"{'picks':<10}{'agree':>8}{'n states':>10}{'legal cards':>13}"
          f"{'agent on-colour':>17}{'baseline on-colour':>20}")
    print("-" * 78)
    bins = [(0, 6), (6, 13), (14, 20), (20, 27), (28, 34), (34, 42)]
    for lo, hi in bins:
        chunk = [r for r in records if lo <= r["t"] < hi]
        if not chunk:
            continue
        agree = np.mean([(r["agent"] == r["rival"]).float().mean().item() for r in chunk])
        legal = np.mean([r["legal"] for r in chunk])
        on_agent, on_rival = [], []
        for r in chunk:
            mass = colour_mass(r["pool"].numpy(), colour)
            top2 = set()
            top2_idx = np.argsort(-mass, axis=1)[:, :2]
            for who, key in [(on_agent, "agent"), (on_rival, "rival")]:
                picked = r[key].numpy()
                pc = colour[picked]
                ok = np.array([
                    pc[i] in ([MONO[j] for j in top2_idx[i]] + ["Colorless", "Multicolor"])
                    for i in range(len(picked))
                ])
                who.append(ok.mean())
        print(f"{f'{lo + 1}-{hi}':<10}{100 * agree:>7.1f}%{len(chunk) * args.n_drafts:>10}"
              f"{legal:>13.1f}{100 * np.mean(on_agent):>16.1f}%{100 * np.mean(on_rival):>19.1f}%")

    # --- colour commitment curve -------------------------------------------
    print("\nColour commitment: share of the pool in its top two colours, by pick")
    print(f"{'after pick':<12}{'agent':>9}{'DraftNet (seat 1)':>20}")
    print("-" * 41)
    seat1 = None
    for r in records:
        pass
    for t in [7, 14, 21, 28, 35, 42]:
        chunk = [r for r in records if r["t"] == t - 1]
        if not chunk:
            continue
        r = chunk[0]
        mass = colour_mass(r["pool"].numpy(), colour)
        share = np.sort(mass, axis=1)[:, -2:].sum(1) / np.maximum(mass.sum(1), 1)
        print(f"{t:<12}{100 * share.mean():>8.1f}%{'':>20}")

    # --- what changes when they disagree -----------------------------------
    print("\nWhen they disagree, what does the agent take instead?")
    agent_dn_rank, rival_dn_rank, agent_rar, rival_rar = [], [], [], []
    agent_oncol, rival_oncol = [], []
    taken_agent = np.zeros(num_cards)
    taken_rival = np.zeros(num_cards)
    offered_count = np.zeros(num_cards)
    for r in records:
        differ = (r["agent"] != r["rival"]).numpy()
        a, b = r["agent"].numpy(), r["rival"].numpy()
        np.add.at(taken_agent, a, 1)
        np.add.at(taken_rival, b, 1)
        offered_count += (r["pack"].numpy() > 0).sum(0)
        if differ.sum() == 0:
            continue
        agent_dn_rank.append(dn_rank[a[differ]].mean())
        rival_dn_rank.append(dn_rank[b[differ]].mean())
        agent_rar.append(rarity_rank[a[differ]].mean())
        rival_rar.append(rarity_rank[b[differ]].mean())
        mass = colour_mass(r["pool"].numpy(), colour)
        top2_idx = np.argsort(-mass, axis=1)[:, :2]
        for who, picked in [(agent_oncol, a), (rival_oncol, b)]:
            pc = colour[picked]
            ok = [pc[i] in ([MONO[j] for j in top2_idx[i]] + ["Colorless", "Multicolor"])
                  for i in np.where(differ)[0]]
            who.append(np.mean(ok))
    print(f"  DraftNet p1p1 rank of card taken       agent {np.mean(agent_dn_rank):6.1f}   "
          f"baseline {np.mean(rival_dn_rank):6.1f}")
    print(f"  rarity (0=common .. 3=mythic)          agent {np.mean(agent_rar):6.2f}   "
          f"baseline {np.mean(rival_rar):6.2f}")
    print(f"  on-colour rate                         agent {100 * np.mean(agent_oncol):5.1f}%   "
          f"baseline {100 * np.mean(rival_oncol):5.1f}%")

    # --- contextual preferences --------------------------------------------
    rate_a = np.divide(taken_agent, np.maximum(offered_count, 1))
    rate_b = np.divide(taken_rival, np.maximum(offered_count, 1))
    enough = offered_count >= 200
    delta = np.where(enough, rate_a - rate_b, 0.0)
    for label, order in [(f"taken MORE often than {baseline_name}", np.argsort(-delta)),
                         (f"taken LESS often than {baseline_name}", np.argsort(delta))]:
        print(f"\nCards {label} (given the same opportunities):")
        print(f"    {'card':<32}{'rar':<6}{'col':<6}{'agent':>8}{'baseline':>10}{'DNrank':>8}")
        for i in order[:10]:
            print(f"    {bank.cardnames[i][:31]:<32}{rarity[i][:5]:<6}{colour[i][:5]:<6}"
                  f"{100 * rate_a[i]:>7.1f}%{100 * rate_b[i]:>9.1f}%{int(dn_rank[i]):>8}")

    moved = np.argsort(-np.abs(rl_rank - dn_rank))
    print(f"\nBiggest p1p1 rank movers (RL rank vs DraftNet rank):")
    print(f"    {'card':<32}{'rar':<6}{'col':<6}{'RL':>6}{'DN':>6}{'move':>7}")
    for i in moved[:12]:
        print(f"    {bank.cardnames[i][:31]:<32}{rarity[i][:5]:<6}{colour[i][:5]:<6}"
              f"{int(rl_rank[i]):>6}{int(dn_rank[i]):>6}{int(dn_rank[i] - rl_rank[i]):>+7}")


if __name__ == "__main__":
    main()

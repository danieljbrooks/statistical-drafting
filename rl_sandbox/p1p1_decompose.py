"""
The trained agent's pack-1 pick-1 order, decomposed into where it came from.

There is no separate value network in this setup (see METHOD.md) - the critic is the
winrate model, applied only to finished pools. So a p1p1 score has three distinct
readings, and comparing them says what the policy actually learned:

  policy      the trained network's own logit from an empty pool with every card
              available. What it would do.
  rollout A   the measured advantage of first-picking that card: fork the draft at pick
              one, force the card, play the remaining 41 picks under the trained policy
              against seven copies of itself, score the finished pool, and centre within
              the pack. This is the training signal - the thing the policy is fit to.
              Centring within the pack matters: packs differ in overall quality, and
              without it a card that happens to appear in strong packs looks good.
  static V    the winrate model's opinion of a one-card pool holding just that card.
              What the reward says about the card in isolation, with no draft around it.

policy vs rollout says whether the network absorbed its own training signal.
rollout vs static says what the 41 picks of context added on top of the card alone.

Run from the repo root:
    python rl_sandbox/p1p1_decompose.py FDN --n-drafts 1500
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
from draft_env import N_SEATS, DraftEnv
from packbank import PackBank
from run_env_check import load_winrate_model
from train_policy import PickPolicy, play, score_pools


def spearman(a, b, mask=None):
    a, b = np.asarray(a), np.asarray(b)
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


@torch.no_grad()
def policy_p1p1(policy, num_cards):
    """Logits from an empty pool with every card available."""
    return policy.logits(torch.zeros(1, num_cards), torch.ones(1, num_cards)).squeeze(0).numpy()


@torch.no_grad()
def draftnet_p1p1(network, num_cards):
    import torch.nn.functional as F
    x = torch.zeros(1, num_cards)
    for layer, norm in zip(network.hidden_layers, network.norms):
        x = norm(network.dropout_layer(F.gelu(layer(x))))
    return network.output_layer(x).squeeze(0).numpy()


@torch.no_grad()
def static_value(reward, num_cards):
    """Winrate the reward model assigns a pool holding one copy of each card."""
    pools = torch.eye(num_cards)
    n_picks = torch.ones(num_cards, dtype=torch.long)
    out = []
    for i in range(0, num_cards, 2048):
        out.append(torch.sigmoid(reward(pools[i : i + 2048], n_picks[i : i + 2048])))
    return torch.cat(out).numpy()


@torch.no_grad()
def rollout_advantage(policy, bank, reward, n_drafts, seed=31337, chunk=250):
    """
    Measured advantage of each possible first pick, under the trained policy.

    Returns (mean advantage per card, number of observations per card).
    """
    num_cards = bank.num_cards
    picks = bank.pack_size * bank.n_boosters
    total = np.zeros(num_cards)
    count = np.zeros(num_cards)

    for start in range(0, n_drafts, chunk):
        size = min(chunk, n_drafts - start)
        env = DraftEnv(bank, n_drafts=size, seed=seed + start)
        packs = env.packs[:, 0, :num_cards].clone()
        n_actions = int((packs > 0).sum(dim=1).min().item())
        candidates = packs.topk(n_actions, dim=1).indices

        forked = env.fork(n_actions)
        pools_all = forked.pools[:, :, :num_cards].reshape(-1, num_cards)
        packs_all = forked.packs[:, :, :num_cards].reshape(-1, num_cards)
        empty = ~(packs_all > 0).any(dim=1, keepdim=True)
        actions = policy.logits(pools_all, packs_all + empty.to(packs_all.dtype)).argmax(1)
        actions = actions.view(forked.n_drafts, N_SEATS)
        actions[:, 0] = candidates.reshape(-1)
        forked.step(actions)
        play(forked, policy, temperature=0.0, noise_group=n_actions)

        q = score_pools(reward, forked.collections()[:, 0], picks).view(size, n_actions)
        # Centre within the pack: the question is which card in THIS pack is best.
        advantage = (q - q.mean(dim=1, keepdim=True)).numpy()
        index = candidates.numpy()
        np.add.at(total, index.reshape(-1), advantage.reshape(-1))
        np.add.at(count, index.reshape(-1), 1)
        print(f"  rolled out {start + size}/{n_drafts} packs", flush=True)

    with np.errstate(invalid="ignore"):
        return total / np.maximum(count, 1), count


def zscore(values, mask):
    z = np.full_like(values, np.nan, dtype=float)
    z[mask] = (values[mask] - values[mask].mean()) / values[mask].std()
    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FDN")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument("--rank-tag", default="diamond")
    parser.add_argument("--reward", default="mlp_400x2")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--n-drafts", type=int, default=1500)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    bank = PackBank(args.set_abbreviation, args.draft_mode)
    num_cards = bank.num_cards
    reward, _ = load_winrate_model(
        args.set_abbreviation, args.draft_mode, args.rank_tag, args.reward
    )
    draftnet = sd.load_draftnet(
        bank.cardnames, args.set_abbreviation, args.draft_mode, "data/models/"
    )
    path = args.policy or (
        f"data/rl_models/{args.set_abbreviation}_{args.draft_mode}_tabularasa_selfplay.pt"
    )
    policy = PickPolicy(bank.cardnames, dropout=0.1)
    policy.load_state_dict(torch.load(path, weights_only=True))
    policy.eval()

    cards = pd.read_csv(f"data/cards/{args.set_abbreviation}.csv").set_index("name")
    cards = cards.reindex(bank.cardnames)

    print(f"{args.set_abbreviation} p1p1 decomposition  policy={os.path.basename(path)}")
    print(f"rolling out {args.n_drafts} packs x every card in them...\n")

    table = pd.DataFrame({
        "card": bank.cardnames,
        "rarity": cards["rarity"].fillna("?").to_numpy(),
        "colour": cards["color_identity"].fillna("Colorless").to_numpy(),
        "policy": policy_p1p1(policy, num_cards),
        "static_v": static_value(reward, num_cards),
        "draftnet": draftnet_p1p1(draftnet, num_cards),
    })
    advantage, support = rollout_advantage(policy, bank, reward, args.n_drafts)
    table["rollout_a"] = advantage
    table["support"] = support.astype(int)

    supported = table["support"].to_numpy() >= args.min_support
    print(f"\n{supported.sum()} of {num_cards} cards seen in at least "
          f"{args.min_support} packs; correlations use those.\n")

    print("Spearman rank correlations (well-supported cards):")
    pairs = [("policy", "rollout_a"), ("policy", "static_v"), ("rollout_a", "static_v"),
             ("policy", "draftnet"), ("rollout_a", "draftnet"), ("static_v", "draftnet")]
    for a, b in pairs:
        print(f"  {a:<10} vs {b:<10} {spearman(table[a], table[b], supported):+.3f}")

    print(f"\nspread: policy logit sd={table['policy'][supported].std():.3f}   "
          f"rollout advantage sd={table['rollout_a'][supported].std():.4f} winrate   "
          f"static V sd={table['static_v'][supported].std():.4f} winrate")

    # Residual: what the policy likes beyond what the card alone is worth.
    table["z_policy"] = zscore(table["policy"].to_numpy(), supported)
    table["z_static"] = zscore(table["static_v"].to_numpy(), supported)
    table["z_rollout"] = zscore(table["rollout_a"].to_numpy(), supported)
    table["residual"] = table["z_policy"] - table["z_static"]

    view = table[supported].copy()
    view["dn_rank"] = table["draftnet"].rank(ascending=False).astype(int)[supported]

    print(f"\nTop {args.top} first picks by the trained policy")
    print(f"{'#':>3} {'card':<32}{'rar':<5}{'col':<5}{'z_pol':>7}{'z_roll':>8}"
          f"{'z_stat':>8}{'DNrank':>8}{'obs':>6}")
    print("-" * 85)
    for rank, (_, row) in enumerate(view.sort_values("policy", ascending=False).head(args.top).iterrows(), 1):
        print(f"{rank:>3} {row['card'][:31]:<32}{row['rarity'][:4]:<5}{row['colour'][:4]:<5}"
              f"{row['z_policy']:>7.2f}{row['z_rollout']:>8.2f}{row['z_static']:>8.2f}"
              f"{row['dn_rank']:>8}{row['support']:>6}")

    for label, ascending in [("most over-rated by the policy vs the card alone", False),
                             ("most under-rated by the policy vs the card alone", True)]:
        print(f"\nCards {label}:")
        subset = view.sort_values("residual", ascending=ascending).head(10)
        print(f"    {'card':<32}{'rar':<5}{'col':<5}{'z_pol':>7}{'z_roll':>8}{'z_stat':>8}{'resid':>8}")
        for _, row in subset.iterrows():
            print(f"    {row['card'][:31]:<32}{row['rarity'][:4]:<5}{row['colour'][:4]:<5}"
                  f"{row['z_policy']:>7.2f}{row['z_rollout']:>8.2f}{row['z_static']:>8.2f}"
                  f"{row['residual']:>8.2f}")

    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()

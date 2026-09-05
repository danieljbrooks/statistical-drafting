"""
Train a draft policy by rollout-based policy iteration (see METHOD.md).

Each iteration:
  1. Play a batch of self-play drafts with the current policy, sampling for exploration.
  2. Freeze the state at several pick numbers. At each, enumerate every card in the pack.
  3. Fork the draft once per candidate card, force that card, and roll the rest out to
     completion. Forked worlds share the pack schedule and the opponents' sampling noise,
     so the only difference between them is the card under test.
  4. Score the finished pools with the winrate model, form advantages, and fit the policy
     to pi* proportional to pi(a|s) exp(A(s,a)/beta).

Run from the repo root:
    python rl_sandbox/train_policy.py FDN --iterations 20
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "winrate_sandbox"))

import statisticaldrafting as sd
from draft_env import N_SEATS, DraftEnv
from packbank import PackBank
from run_env_check import load_winrate_model

NEG_INF = float("-inf")


class PickPolicy(sd.DraftNet):
    """
    DraftNet's architecture, with two changes for reinforcement learning.

    Dropout is configurable (DraftNet's 0.6 is tuned for fitting millions of human picks
    and is far too strong here), and masking is additive rather than multiplicative -
    DraftNet multiplies its logits by the pack vector, which lets a card that is not in
    the pack score 0 and beat an in-pack card with a negative logit.

    Parameter names match DraftNet exactly, so a trained DraftNet can be loaded straight
    into this class for the bootstrap experiments.
    """

    def __init__(self, cardnames, dropout=0.1):
        super().__init__(cardnames)
        self.dropout_layer = nn.Dropout(dropout)

    def logits(self, pools, packs):
        x = pools
        for layer, norm in zip(self.hidden_layers, self.norms):
            x = norm(self.dropout_layer(F.gelu(layer(x))))
        return self.output_layer(x).masked_fill(packs <= 0, NEG_INF)


@torch.no_grad()
def play(env, policy, temperature, noise_group=1, generator=None):
    """
    Play ``env`` to completion under ``policy``.

    ``noise_group`` is the number of forked replicas per underlying draft. Gumbel noise
    is drawn once per underlying draft and tiled across its replicas, so replicas that
    have not yet diverged make identical picks - the common random numbers.
    """
    policy.eval()
    num_cards = env.num_cards
    n_base = env.n_drafts // noise_group
    while not env.done:
        pools = env.pools[:, :, :num_cards].reshape(-1, num_cards)
        packs = env.packs[:, :, :num_cards].reshape(-1, num_cards)
        # A seat holding only basics has no legal card; step() forces the pad slot, so
        # the action computed here is discarded. Offer a full pack to keep it finite.
        empty = ~(packs > 0).any(dim=1, keepdim=True)
        scores = policy.logits(pools, packs + empty.to(packs.dtype))
        if temperature > 0:
            uniform = torch.rand(
                (n_base, N_SEATS, num_cards), device=env.device, generator=generator
            ).clamp_min(1e-20)
            gumbel = -torch.log(-torch.log(uniform))
            scores = scores / temperature + gumbel.repeat_interleave(
                noise_group, dim=0
            ).reshape(-1, num_cards)
        env.step(scores.argmax(dim=1).view(env.n_drafts, N_SEATS))
    return env


@torch.no_grad()
def score_pools(reward, collections, picks_per_draft, batch=4096):
    out = []
    for i in range(0, collections.shape[0], batch):
        chunk = collections[i : i + batch].float()
        n_picks = torch.full((chunk.shape[0],), picks_per_draft, dtype=torch.long)
        out.append(torch.sigmoid(reward(chunk, n_picks)))
    return torch.cat(out)


@torch.no_grad()
def collect(policy, bank, reward, n_drafts, timesteps, seed,
            explore_temperature=1.0, rollout_temperature=0.0, beta_scale=1.0,
            beta_absolute=None, kl_reference=None, kl_lambda=0.0):
    """
    One batch of training data: states, and the advantage-weighted target at each.

    Returns (pools, packs, targets, diagnostics). ``targets`` is a full-width probability
    vector per state - the same shape ``train.py`` already trains against, with the
    one-hot human pick replaced by pi*.

    With ``kl_reference`` set, the improvement step carries a second KL constraint, to a
    frozen reference policy rather than to the previous iteration:

        max  sum_a pi'(a) A(s,a) - beta KL(pi'||pi_old) - lambda KL(pi'||pi_ref)

    whose solution is a geometric mixture of the two anchors,

        pi*(a) ~ pi_old(a)^(beta/(beta+lambda)) pi_ref(a)^(lambda/(beta+lambda))
                 exp(A(s,a) / (beta+lambda))

    ``kl_lambda`` is lambda/beta, so it is dimensionless: 0 recovers the unanchored
    update exactly, and large values drive pi* to the reference regardless of advantage.

    Two ways to set beta, and they behave very differently when the reward is weak:

    ``beta_scale``   beta = beta_scale * sd(A), recomputed per batch. The step is then a
                     fixed distance in *advantage standard deviations*, so it is the same
                     whether the advantage is real signal or pure noise - sd(A) divides
                     the noise back out. This is the per-group-std problem that Dr. GRPO
                     identifies, one level up. Convenient, but it does not degrade.
    ``beta_absolute`` beta is a constant in winrate units. When the reward gets noisier
                     the advantage shrinks, A/beta shrinks with it, and the update
                     genuinely collapses toward pi* = pi. This is the version the
                     "graceful degradation" argument in METHOD.md actually requires.

    Reference points from this repo: advantage sd runs about 0.022 with a Diamond+Mythic
    transfer reward and about 0.009 with the top-player reward, so a beta_absolute near
    0.02 is roughly one Diamond+ standard deviation and about two top-player ones.
    """
    policy.eval()
    num_cards = bank.num_cards
    picks = bank.pack_size * bank.n_boosters
    generator = torch.Generator().manual_seed(seed)

    env = DraftEnv(bank, n_drafts=n_drafts, seed=seed)
    snapshots = {}
    while not env.done:
        if env.t in timesteps:
            snapshots[env.t] = env.fork(1)  # fork(1) is a deep copy
        pools = env.pools[:, :, :num_cards].reshape(-1, num_cards)
        packs = env.packs[:, :, :num_cards].reshape(-1, num_cards)
        empty = ~(packs > 0).any(dim=1, keepdim=True)
        scores = policy.logits(pools, packs + empty.to(packs.dtype))
        uniform = torch.rand((n_drafts, N_SEATS, num_cards), generator=generator).clamp_min(1e-20)
        scores = scores / explore_temperature + (-torch.log(-torch.log(uniform))).reshape(-1, num_cards)
        env.step(scores.argmax(dim=1).view(n_drafts, N_SEATS))

    all_pools, all_packs, all_targets = [], [], []
    advantage_spread, kl_to_prior, kl_from_reference, betas = [], [], [], []

    for t, snapshot in snapshots.items():
        state_pools = snapshot.pools[:, 0, :num_cards].clone()
        state_packs = snapshot.packs[:, 0, :num_cards].clone()
        # Every draftable card in the pack is a candidate. The count is not simply
        # pack_size - t: a pack can also hold a basic land, which is not draftable, so
        # take the smallest number of real options any draft has at this t and enumerate
        # that many. topk on a 0/1 pack vector returns legal cards first.
        n_actions = int((state_packs > 0).sum(dim=1).min().item())
        if n_actions < 2:
            # A single legal card is not a decision and carries no training signal.
            continue
        candidates = state_packs.topk(n_actions, dim=1).indices  # (n_drafts, n_actions)

        forked = snapshot.fork(n_actions)
        # Opponents act normally; seat 0 is forced to each candidate in turn.
        pools = forked.pools[:, :, :num_cards].reshape(-1, num_cards)
        packs = forked.packs[:, :, :num_cards].reshape(-1, num_cards)
        empty = ~(packs > 0).any(dim=1, keepdim=True)
        scores = policy.logits(pools, packs + empty.to(packs.dtype))
        actions = scores.argmax(dim=1).view(forked.n_drafts, N_SEATS)
        actions[:, 0] = candidates.reshape(-1)
        forked.step(actions)
        play(forked, policy, rollout_temperature, noise_group=n_actions, generator=generator)

        q = score_pools(reward, forked.collections()[:, 0], picks).view(n_drafts, n_actions)

        prior = torch.softmax(
            policy.logits(state_pools, state_packs).gather(1, candidates), dim=1
        )
        advantage = q - (prior * q).sum(dim=1, keepdim=True)
        if beta_absolute is not None and beta_absolute > 0:
            beta = beta_absolute
        else:
            beta = max(advantage.std().item(), 1e-6) * beta_scale
        betas.append(beta)
        # Geometric mixture of the two anchors; done in log space because prior_ref can
        # be arbitrarily small under a confident reference.
        scale = 1.0 + kl_lambda
        log_weights = torch.log(prior.clamp_min(1e-12)) / scale + advantage / (beta * scale)
        if kl_reference is not None and kl_lambda > 0:
            prior_ref = torch.softmax(
                kl_reference.logits(state_pools, state_packs).gather(1, candidates), dim=1
            )
            log_weights = log_weights + (kl_lambda / scale) * torch.log(
                prior_ref.clamp_min(1e-12)
            )
        weights = torch.softmax(log_weights, dim=1)

        assert torch.isfinite(weights).all(), f"non-finite pi* at t={t}"
        target = torch.zeros_like(state_pools)
        target.scatter_(1, candidates, weights)
        all_pools.append(state_pools)
        all_packs.append(state_packs)
        all_targets.append(target)
        advantage_spread.append(advantage.std().item())
        kl_to_prior.append(
            (weights * (torch.log(weights + 1e-12) - torch.log(prior + 1e-12))).sum(1).mean().item()
        )
        if kl_reference is not None:
            # How far the *policy* now sits from the reference - the quantity the
            # penalty is meant to control, as opposed to the per-step KL above.
            reference_prior = torch.softmax(
                kl_reference.logits(state_pools, state_packs).gather(1, candidates), dim=1
            )
            kl_from_reference.append(
                (prior * (torch.log(prior + 1e-12)
                          - torch.log(reference_prior + 1e-12))).sum(1).mean().item()
            )

    diagnostics = {
        "advantage_sd": float(np.mean(advantage_spread)),
        "kl": float(np.mean(kl_to_prior)),
        "kl_ref": float(np.mean(kl_from_reference)) if kl_from_reference else float("nan"),
        "beta": float(np.mean(betas)),
    }
    return (
        torch.cat(all_pools), torch.cat(all_packs), torch.cat(all_targets), diagnostics,
    )


def fit(policy, optimizer, pools, packs, targets, epochs=4, batch=256):
    """Cross-entropy of the policy against pi*. The projection step."""
    policy.train()
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(pools))
        for i in range(0, len(order), batch):
            index = order[i : i + batch]
            optimizer.zero_grad()
            logits = policy.logits(pools[index], packs[index])
            log_probs = torch.log_softmax(logits, dim=1)
            # Masked cards carry -inf log-probability and zero target weight; the
            # product is NaN, so zero those terms rather than multiplying them.
            target = targets[index]
            loss = -(target * torch.where(target > 0, log_probs, torch.zeros_like(log_probs))).sum(1).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(policy, bank, reward, n_drafts=256, seed=999):
    """Self-play winrate of the policy's own pools, under the reward model."""
    env = DraftEnv(bank, n_drafts=n_drafts, seed=seed)
    play(env, policy, temperature=0.0)
    collections = env.collections().reshape(-1, bank.num_cards)
    values = score_pools(reward, collections, bank.pack_size * bank.n_boosters)
    return values.mean().item(), collections


@torch.no_grad()
def evaluate_vs_reference(policy, reference, bank, reward, n_drafts=256, seed=999):
    """
    Seat 0 plays the policy, the other seven play a frozen reference.

    Self-play winrate is close to useless as a progress metric here: when all eight
    seats improve together they compete for the same cards, and the gains cancel. Held
    against a fixed opponent the improvement is visible.
    """
    num_cards = bank.num_cards
    env = DraftEnv(bank, n_drafts=n_drafts, seed=seed)
    policy.eval()
    reference.eval()
    while not env.done:
        pools = env.pools[:, :, :num_cards]
        packs = env.packs[:, :, :num_cards]
        empty = ~(packs > 0).any(dim=2, keepdim=True)
        offered = packs + empty.to(packs.dtype)
        actions = torch.zeros((env.n_drafts, N_SEATS), dtype=torch.long)
        actions[:, 0] = policy.logits(pools[:, 0], offered[:, 0]).argmax(dim=1)
        rest_pools = pools[:, 1:].reshape(-1, num_cards)
        rest_packs = offered[:, 1:].reshape(-1, num_cards)
        actions[:, 1:] = reference.logits(rest_pools, rest_packs).argmax(dim=1).view(
            env.n_drafts, N_SEATS - 1
        )
        env.step(actions)
    picks = bank.pack_size * bank.n_boosters
    agent = score_pools(reward, env.collections()[:, 0], picks).mean().item()
    field = score_pools(reward, env.collections()[:, 1:].reshape(-1, num_cards), picks).mean().item()
    return agent, field


def save(policy, args, history, suffix=""):
    os.makedirs(args.out, exist_ok=True)
    tag = "draftnetinit" if args.init_draftnet else "tabularasa"
    # Reward architecture AND the population it was fit to are both part of the
    # experiment; leaving the rank tag out silently overwrites a different run.
    anchor = "" if args.kl_lambda == 0 else f"_kl{args.kl_reference}{args.kl_lambda:g}"
    if args.beta_absolute:
        anchor += f"_beta{args.beta_absolute:g}"
    name = (f"{args.set_abbreviation}_{args.draft_mode}_{args.rank_tag}_{tag}_"
            f"{args.reward}{anchor}_selfplay{suffix}")
    torch.save(policy.state_dict(), os.path.join(args.out, name + ".pt"))
    with open(os.path.join(args.out, name + ".json"), "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set_abbreviation", nargs="?", default="FDN")
    parser.add_argument("--draft-mode", default="Premier")
    parser.add_argument("--rank-tag", default="diamond")
    parser.add_argument("--reward", default="mlp_400x2")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--n-drafts", type=int, default=96)
    parser.add_argument("--n-timesteps", type=int, default=6)
    parser.add_argument("--beta-scale", type=float, default=1.0,
                        help="beta = beta_scale * sd(advantage), recomputed per batch. "
                             "Steps are measured in advantage SDs, so they do not shrink "
                             "when the reward is noisy. Ignored if --beta-absolute is set.")
    parser.add_argument("--beta-absolute", type=float, default=0.02,
                        help="Fixed beta in winrate units, overriding --beta-scale. On by "
                             "default: the update then degrades toward a no-op as the "
                             "advantage shrinks, which beta_scale does not. Pass 0 to "
                             "fall back to --beta-scale.")
    parser.add_argument("--kl-reference", default="draftnet", choices=["none", "init", "draftnet"],
                        help='Frozen policy the KL penalty anchors to. "init" is the '
                             'starting policy (DraftNet when --init-draftnet, else random).')
    parser.add_argument("--kl-lambda", type=float, default=1.0,
                        help="lambda/beta. 0 disables the penalty; large values pin the "
                             "policy to the reference. On by default: without it the "
                             "policy drifts arbitrarily far from the reference even "
                             "with a working per-step trust region.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--explore-temperature", type=float, default=1.0)
    parser.add_argument("--init-draftnet", action="store_true",
                        help="start from the trained DraftNet instead of random weights")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/rl_models/")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    bank = PackBank(args.set_abbreviation, args.draft_mode)
    reward, meta = load_winrate_model(
        args.set_abbreviation, args.draft_mode, args.rank_tag, args.reward
    )
    assert list(meta["cardnames"]) == bank.cardnames

    policy = PickPolicy(bank.cardnames, dropout=args.dropout)
    if args.init_draftnet:
        policy.load_state_dict(
            torch.load(
                f"data/models/{args.set_abbreviation}_{args.draft_mode}.pt",
                weights_only=True,
            )
        )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    picks = bank.pack_size * bank.n_boosters
    # Spread the sampled decisions over the part of the draft where picks still matter;
    # the last few picks of the third booster change nothing.
    timesteps = sorted(set(np.linspace(1, picks - 12, args.n_timesteps).astype(int).tolist()))
    # The last pick of a booster is forced, so it is never a decision worth sampling.
    timesteps = [t for t in timesteps if t % bank.pack_size <= bank.pack_size - 2]

    print(f"{args.set_abbreviation} {args.draft_mode}  {bank.num_cards} cards  "
          f"{picks} picks  reward={args.reward}")
    print(f"init={'DraftNet' if args.init_draftnet else 'random'}  "
          f"drafts/iter={args.n_drafts}  timesteps={timesteps}  "
          + (f"beta={args.beta_absolute} (absolute)" if args.beta_absolute
             else f"beta_scale={args.beta_scale} (x advantage sd)")
          + (f"  kl={args.kl_reference}/{args.kl_lambda:g}" if args.kl_lambda else "") + "\n")

    history = []
    # A frozen copy of the starting policy: the fixed yardstick progress is measured on.
    reference = PickPolicy(bank.cardnames, dropout=args.dropout)
    reference.load_state_dict({k: v.clone() for k, v in policy.state_dict().items()})
    reference.eval()

    kl_reference = None
    if args.kl_reference == "init":
        kl_reference = reference
    elif args.kl_reference == "draftnet":
        kl_reference = PickPolicy(bank.cardnames, dropout=args.dropout)
        kl_reference.load_state_dict(torch.load(
            f"data/models/{args.set_abbreviation}_{args.draft_mode}.pt", weights_only=True
        ))
    if kl_reference is not None:
        kl_reference.eval()
        for parameter in kl_reference.parameters():
            parameter.requires_grad = False

    baseline, _ = evaluate(policy, bank, reward)
    agent, field = evaluate_vs_reference(policy, reference, bank, reward)
    print(f"{'iter':>5}{'selfplay':>10}{'vs_frozen':>11}{'field':>8}{'edge':>8}"
          f"{'adv_sd':>9}{'KL':>7}{'KLref':>8}{'loss':>7}{'sec':>6}")
    print(f"{0:>5}{baseline:>10.4f}{agent:>11.4f}{field:>8.4f}{agent - field:>+8.4f}"
          f"{'-':>9}{'-':>7}{'-':>8}{'-':>7}{'-':>6}")

    for iteration in range(1, args.iterations + 1):
        t0 = time.time()
        pools, packs, targets, diagnostics = collect(
            policy, bank, reward, args.n_drafts, timesteps,
            seed=args.seed + iteration, explore_temperature=args.explore_temperature,
            beta_scale=args.beta_scale, beta_absolute=args.beta_absolute,
            kl_reference=kl_reference, kl_lambda=args.kl_lambda,
        )
        loss = fit(policy, optimizer, pools, packs, targets)
        winrate, _ = evaluate(policy, bank, reward)
        agent, field = evaluate_vs_reference(policy, reference, bank, reward)
        history.append(dict(iteration=iteration, winrate=winrate, vs_frozen=agent,
                            field=field, loss=loss, **diagnostics))
        print(f"{iteration:>5}{winrate:>10.4f}{agent:>11.4f}{field:>8.4f}{agent - field:>+8.4f}"
              f"{diagnostics['advantage_sd']:>9.4f}{diagnostics['kl']:>7.3f}"
              f"{diagnostics['kl_ref']:>8.3f}{loss:>7.3f}{time.time() - t0:>6.1f}", flush=True)
        if iteration % 10 == 0:
            save(policy, args, history, suffix=f"_iter{iteration}")

    save(policy, args, history)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()

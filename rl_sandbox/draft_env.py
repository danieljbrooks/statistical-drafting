"""
An exact 8-seat booster draft, vectorized over many drafts at once.

Every mechanic that changes what a drafter sees is reproduced: eight seats, three
boosters, alternating pass direction, and the wheel (the pack you open reaches you again
eight picks later, minus the eight cards taken from it). Packs are dealt from real
observed boosters rather than a collation model, so pack contents are exactly the
17lands distribution.

Basic lands occupy the pad index at ``num_cards``. They sit in packs like any other card
but no policy can choose one, so they wheel to the end of a booster and are taken only
when a seat has nothing else - which is what human drafters do, and what makes a
simulated 42-pick sequence directly comparable to a row of ``CollectionWinrateDataset``.

Shapes throughout: ``(n_drafts, n_seats, ...)``. Seat 0 is the agent's seat by
convention, though ``run_drafts`` will place any policy in any seat.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packbank import PackBank  # noqa: E402

N_SEATS = 8
# Booster 0 and 2 pass left, booster 1 passes right.
PASS_DIRECTION = [1, -1, 1]


class DraftEnv:
    """
    Vectorized booster draft.

    ``packs`` counts cards currently in the pack each seat is holding; ``pools`` counts
    cards each seat has taken. Both are width ``num_cards + 1`` so the basic-land pad
    slot has somewhere to go, and both are sliced to ``num_cards`` before any model
    sees them.
    """

    def __init__(self, pack_bank: PackBank, n_drafts: int, device=None, seed=0):
        self.bank = pack_bank
        self.n_drafts = n_drafts
        self.num_cards = pack_bank.num_cards
        self.pack_size = pack_bank.pack_size
        self.n_boosters = pack_bank.n_boosters
        self.picks_per_draft = self.pack_size * self.n_boosters
        self.device = device or torch.device("cpu")
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        width = self.num_cards + 1
        self.pools = torch.zeros((self.n_drafts, N_SEATS, width), device=self.device)
        self.packs = torch.zeros((self.n_drafts, N_SEATS, width), device=self.device)
        self.pick_sequences = torch.zeros(
            (self.n_drafts, N_SEATS, self.picks_per_draft), dtype=torch.long, device=self.device
        )
        # Deal every booster up front rather than at the start of each pack. A forked
        # rollout then inherits the whole pack schedule, so sibling worlds testing
        # different actions see identical future packs - the common random numbers that
        # make small advantage differences measurable.
        self.schedule = torch.stack(
            [
                torch.from_numpy(
                    self.bank.sample(self.rng, self.n_drafts * N_SEATS, booster)
                ).view(self.n_drafts, N_SEATS, -1)
                for booster in range(self.n_boosters)
            ],
            dim=1,
        ).to(self.device)
        self.t = 0
        self._deal(booster=0)
        return self

    def _deal(self, booster):
        """Give every seat its scheduled opening pack for this booster."""
        indices = self.schedule[:, booster]
        self.packs.zero_()
        self.packs.scatter_add_(2, indices, torch.ones_like(indices, dtype=self.packs.dtype))

    def fork(self, k):
        """
        ``k`` independent copies of every draft, sharing this draft's pack schedule.

        Used to evaluate several candidate actions from one state: replica ``i * k + j``
        continues draft ``i``. Because the schedule is copied rather than redrawn, the
        replicas differ only by the actions taken in them.
        """
        forked = object.__new__(DraftEnv)
        forked.bank = self.bank
        forked.n_drafts = self.n_drafts * k
        forked.num_cards = self.num_cards
        forked.pack_size = self.pack_size
        forked.n_boosters = self.n_boosters
        forked.picks_per_draft = self.picks_per_draft
        forked.device = self.device
        forked.rng = self.rng
        forked.t = self.t
        for name in ("pools", "packs", "pick_sequences", "schedule"):
            setattr(forked, name, getattr(self, name).repeat_interleave(k, dim=0))
        return forked

    @property
    def booster(self):
        return self.t // self.pack_size

    @property
    def done(self):
        return self.t >= self.picks_per_draft

    def legal_mask(self):
        """(n_drafts, n_seats, num_cards) bool over draftable (non-basic) cards."""
        return self.packs[:, :, : self.num_cards] > 0

    def step(self, actions):
        """
        Apply one pick for every seat of every draft, then pass the packs.

        Args:
            actions: (n_drafts, n_seats) long card indices. A seat whose pack holds only
                basics is overridden to take the pad slot.
        """
        if self.done:
            raise RuntimeError("draft is over")

        # A seat with no draftable card left must take the basic.
        forced = ~self.legal_mask().any(dim=2)
        actions = torch.where(
            forced, torch.full_like(actions, self.num_cards), actions
        )

        taken = F.one_hot(actions, num_classes=self.num_cards + 1).to(self.packs.dtype)
        self.packs -= taken
        self.pools += taken
        self.pick_sequences[:, :, self.t] = actions
        self.t += 1

        if self.t % self.pack_size == 0:
            if not self.done:
                self._deal(booster=self.booster)
        else:
            # Pass: the pack held by seat i moves to seat i + direction.
            self.packs = torch.roll(self.packs, shifts=PASS_DIRECTION[self.booster], dims=1)
        return self

    def collections(self):
        """(n_drafts, n_seats, num_cards) finished pools, pad slot dropped."""
        return self.pools[:, :, : self.num_cards]


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class Policy:
    """Maps a batch of (pool, pack) to a batch of chosen card indices."""

    def pick(self, pools, packs):
        raise NotImplementedError


class DraftNetPolicy(Policy):
    """
    A trained DraftNet, optionally sampled at a temperature.

    DraftNet masks by *multiplying* its logits by the pack vector, which lets a card
    that is not in the pack (score 0) beat an in-pack card with a negative score. That
    is fine for the website's usual case and wrong for a simulator, so masking here is
    additive.
    """

    def __init__(self, network, temperature=0.0):
        self.network = network.eval()
        self.temperature = temperature

    @torch.no_grad()
    def pick(self, pools, packs):
        logits = self.network.output_layer(_draftnet_hidden(self.network, pools))
        logits = logits.masked_fill(packs <= 0, float("-inf"))
        if self.temperature <= 0:
            return logits.argmax(dim=1)
        return torch.distributions.Categorical(logits=logits / self.temperature).sample()


class RandomPolicy(Policy):
    """Uniform over draftable cards in the pack. The tabula-rasa starting point."""

    def __init__(self, seed=0):
        self.generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def pick(self, pools, packs):
        weights = (packs > 0).float()
        return torch.multinomial(weights, 1, generator=self.generator).squeeze(1)


class GreedyValuePolicy(Policy):
    """
    Takes whichever card in the pack most raises the winrate model's estimate.

    No rollouts and no training: this is one-step lookahead on the reward, and it is the
    cheapest possible probe for whether the reward survives being optimized against
    inside the real pack constraints.
    """

    def __init__(self, winrate_network, picks_per_draft=42):
        self.network = winrate_network.eval()
        self.picks_per_draft = picks_per_draft

    @torch.no_grad()
    def pick(self, pools, packs):
        batch, num_cards = pools.shape
        available = packs > 0
        # Score pool + card for every legal card at once. Illegal cards are scored too
        # and then masked; branching would cost more than the wasted rows.
        candidates = pools.unsqueeze(1).expand(batch, num_cards, num_cards).clone()
        candidates += torch.eye(num_cards, device=pools.device).unsqueeze(0)
        n_picks = pools.sum(dim=1).long().clamp(max=self.picks_per_draft - 1) + 1
        n_picks = n_picks.unsqueeze(1).expand(batch, num_cards).reshape(-1)
        values = self.network(candidates.reshape(-1, num_cards), n_picks).view(batch, num_cards)
        return values.masked_fill(~available, float("-inf")).argmax(dim=1)


def _draftnet_hidden(network, collection):
    """Penultimate DraftNet activation. Mirrors winratetransfer.draftnet_hidden."""
    x = collection
    for layer, norm in zip(network.hidden_layers, network.norms):
        x = norm(network.dropout_layer(F.gelu(layer(x))))
    return x


# ---------------------------------------------------------------------------
# Driving the environment
# ---------------------------------------------------------------------------


def run_drafts(env, policies):
    """
    Play every draft in ``env`` to completion.

    Args:
        policies: one Policy for every seat. Seats sharing a policy object are batched
            into a single forward pass, so eight DraftNet seats cost one call per pick.

    Returns:
        The env, finished. Read ``collections()`` and ``pick_sequences``.
    """
    assert len(policies) == N_SEATS
    groups = {}
    for seat, policy in enumerate(policies):
        groups.setdefault(id(policy), (policy, []))[1].append(seat)

    while not env.done:
        actions = torch.zeros((env.n_drafts, N_SEATS), dtype=torch.long, device=env.device)
        for policy, seats in groups.values():
            index = torch.tensor(seats, device=env.device)
            pools = env.pools[:, index, : env.num_cards].reshape(-1, env.num_cards)
            packs = env.packs[:, index, : env.num_cards].reshape(-1, env.num_cards)
            # A seat down to nothing but basics has no legal card, which would leave a
            # policy sampling from an all-zero distribution. Hand it a full pack; the
            # action is discarded by ``step``, which forces the pad slot.
            empty = ~(packs > 0).any(dim=1, keepdim=True)
            chosen = policy.pick(pools, packs + empty.to(packs.dtype)).view(env.n_drafts, len(seats))
            actions[:, index] = chosen
        env.step(actions)
    return env

"""
Invariants of the draft environment. Run from the repo root:

    python rl_sandbox/test_draft_env.py

The wheel is the mechanic worth testing hardest: if the pass direction or the roll
offset is wrong the drafts still look plausible - packs still empty, pools still fill -
but every signal-reading decision in the format is being learned against fiction.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from draft_env import N_SEATS, PASS_DIRECTION, DraftEnv, RandomPolicy, run_drafts
from packbank import PackBank


def test_wheel_returns_the_opened_pack(bank):
    """
    The pack a seat opens comes back to it eight picks later, holding the six cards
    nobody took. Every card in the wheeled pack must have been in the opened one.
    """
    env = DraftEnv(bank, n_drafts=32, seed=3)
    policies = [RandomPolicy(seed) for seed in range(N_SEATS)]
    opened = env.packs.clone()

    for _ in range(N_SEATS):
        actions = torch.stack(
            [
                policies[seat].pick(
                    env.pools[:, seat, : env.num_cards],
                    env.packs[:, seat, : env.num_cards] + (~(env.packs[:, seat, : env.num_cards] > 0).any(1, keepdim=True)),
                )
                for seat in range(N_SEATS)
            ],
            dim=1,
        )
        env.step(actions)

    wheeled = env.packs
    assert wheeled.sum(dim=2).unique().tolist() == [bank.pack_size - N_SEATS], (
        f"wheeled packs should hold {bank.pack_size - N_SEATS} cards, "
        f"got {wheeled.sum(dim=2).unique().tolist()}"
    )
    assert (wheeled <= opened).all(), "wheeled pack holds a card the opened pack did not"
    print(f"  wheel: seat sees its own pack again at pick {N_SEATS}, "
          f"{bank.pack_size - N_SEATS} of {bank.pack_size} cards left  OK")


def test_pass_direction_alternates(bank):
    """
    Booster 0 and 2 pass one way, booster 1 the other.

    Comparing pack *sizes* cannot tell the directions apart - every pack holds the same
    count either way - so this compares contents: after a pick, the pack now at a seat
    must be the pack its neighbour held, minus exactly the one card that neighbour took.
    """
    assert PASS_DIRECTION == [1, -1, 1]
    env = DraftEnv(bank, n_drafts=8, seed=4)
    for booster, expected in enumerate(PASS_DIRECTION):
        env.t = booster * bank.pack_size
        env._deal(booster)
        before = env.packs.clone()
        # Each seat takes the lowest-indexed card actually in its pack.
        actions = before[:, :, : env.num_cards].argmax(dim=2)
        env.step(actions)

        matches = []
        for shift in (-1, 1):
            removed = torch.roll(before, shifts=shift, dims=1) - env.packs
            # A correct shift leaves exactly one card removed per pack, never negative.
            if (removed >= 0).all() and (removed.sum(dim=2) == 1).all():
                matches.append(shift)
        assert matches == [expected], (
            f"booster {booster}: consistent with shifts {matches}, expected [{expected}]"
        )
    print(f"  pass direction: {PASS_DIRECTION} by booster, verified on contents  OK")


def test_cards_are_conserved(bank):
    """Every dealt card ends up in exactly one pool, and no pack has leftovers."""
    env = DraftEnv(bank, n_drafts=64, seed=5)
    run_drafts(env, [RandomPolicy(seed) for seed in range(N_SEATS)])
    assert env.packs.sum().item() == 0, "packs not empty at the end of the draft"
    total = env.pools.sum().item()
    expected = 64 * N_SEATS * env.picks_per_draft
    assert total == expected, f"{total} cards in pools, expected {expected}"
    assert (env.pools >= 0).all(), "a pool went negative - a card was taken twice"
    print(f"  conservation: {expected} cards dealt, {int(total)} collected, packs empty  OK")


def test_no_seat_takes_a_card_it_was_not_offered(bank):
    """A pick must come out of the pack the seat was holding at that moment."""
    env = DraftEnv(bank, n_drafts=16, seed=6)
    policies = [RandomPolicy(seed) for seed in range(N_SEATS)]
    for _ in range(env.picks_per_draft):
        held = env.packs.clone()
        actions = torch.zeros((16, N_SEATS), dtype=torch.long)
        for seat in range(N_SEATS):
            packs = env.packs[:, seat, : env.num_cards]
            empty = ~(packs > 0).any(dim=1, keepdim=True)
            actions[:, seat] = policies[seat].pick(
                env.pools[:, seat, : env.num_cards], packs + empty.float()
            )
        env.step(actions)
        taken = env.pick_sequences[:, :, env.t - 1]
        offered = torch.gather(held, 2, taken.unsqueeze(2)).squeeze(2)
        assert (offered > 0).all(), "a seat took a card that was not in its pack"
    print("  legality: every pick came from the pack that seat held  OK")


def test_basics_are_taken_last(bank):
    """
    Basics sit in packs but no policy picks them, so they should only ever be taken at
    the very end of a booster, when a seat has nothing else.
    """
    env = DraftEnv(bank, n_drafts=64, seed=7)
    run_drafts(env, [RandomPolicy(seed) for seed in range(N_SEATS)])
    pad_positions = (env.pick_sequences == bank.num_cards).nonzero()[:, 2]
    within_booster = pad_positions % bank.pack_size
    assert within_booster.min().item() >= bank.pack_size - 4, (
        f"a basic was taken at position {within_booster.min().item()} of a booster"
    )
    print(f"  basics: taken only in the last {bank.pack_size - within_booster.min().item()} "
          f"picks of a booster, {len(pad_positions) / (64 * N_SEATS):.2f} per pool  OK")


if __name__ == "__main__":
    bank = PackBank("FIN")
    print(f"FIN: {bank.num_cards} cards, {bank.pack_size}-card packs, "
          f"{bank.n_boosters} boosters, {N_SEATS} seats")
    test_wheel_returns_the_opened_pack(bank)
    test_pass_direction_alternates(bank)
    test_cards_are_conserved(bank)
    test_no_seat_takes_a_card_it_was_not_offered(bank)
    test_basics_are_taken_last(bank)
    print("all invariants hold")

"""Architecture registry shared by the winrate sandbox scripts."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statisticaldrafting as sd

# name -> (network class, fixed kwargs, learning rate, train_winrate_model overrides)
ARCHITECTURES = {
    "linear": (sd.WinrateLinear, {}, 3e-3, {}),
    "embed_32": (sd.WinrateEmbeddingNet, dict(embedding_dim=32, hidden_dims=[64]), 1e-3, {}),
    "embed_128": (sd.WinrateEmbeddingNet, dict(embedding_dim=128, hidden_dims=[256]), 1e-3, {}),
    "mlp_256x2": (sd.WinrateNet, dict(hidden_dims=[256, 256]), 5e-4, {}),
    "mlp_400x2": (sd.WinrateNet, dict(hidden_dims=[400, 400]), 5e-4, {}),
    "mlp_1024x2": (sd.WinrateNet, dict(hidden_dims=[1024, 1024]), 5e-4, {}),
    "mlp_1024x3_res": (
        sd.WinrateNet, dict(hidden_dims=[1024, 1024, 1024], residual=True), 5e-4, {},
    ),
    "mlp_2048x2": (sd.WinrateNet, dict(hidden_dims=[2048, 2048]), 3e-4, {}),
    # Ablations, all at DraftNet's 400x400 size.
    "mlp_400x2_no_pick_cond": (
        sd.WinrateNet, dict(hidden_dims=[400, 400], use_pick_conditioning=False), 5e-4, {},
    ),
    "mlp_400x2_mse_loss": (
        sd.WinrateNet, dict(hidden_dims=[400, 400]), 5e-4, {"loss_name": "mse"},
    ),
    "mlp_400x2_full_pool_only": (
        sd.WinrateNet, dict(hidden_dims=[400, 400]), 5e-4, {"train_n_picks_final_only": True},
    ),
}


def build_network(name, cardnames, picks_per_draft, base_winrate, dropout, input_dropout):
    """Instantiate an architecture by name at a given regularization strength."""
    network_class, fixed, _, _ = ARCHITECTURES[name]
    kwargs = dict(
        cardnames=cardnames,
        picks_per_draft=picks_per_draft,
        base_winrate=base_winrate,
        input_dropout=input_dropout,
        **fixed,
    )
    if network_class is not sd.WinrateLinear:
        kwargs["dropout"] = dropout
    return network_class(**kwargs)


def train_overrides(name, picks_per_draft):
    overrides = dict(ARCHITECTURES[name][3])
    if overrides.pop("train_n_picks_final_only", False):
        overrides["train_n_picks"] = [picks_per_draft]
    return overrides


def learning_rate(name):
    return ARCHITECTURES[name][2]

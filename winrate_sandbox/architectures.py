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
    # --- Transfer from a trained DraftNet (frozen unless noted). ---
    # Scalar summaries only: 8 numbers distilled from DraftNet, no raw collection.
    "dn_scalars_only": (
        sd.WinrateTransferNet,
        dict(feature_modes=["scalars"], use_collection=False, hidden_dims=[]),
        3e-3, {},
    ),
    "dn_scalars_mlp": (
        sd.WinrateTransferNet,
        dict(feature_modes=["scalars"], use_collection=False, hidden_dims=[64, 64]),
        1e-3, {},
    ),
    # Scalars alongside the raw collection.
    "dn_scalars_collection": (
        sd.WinrateTransferNet,
        dict(feature_modes=["scalars"], use_collection=True, hidden_dims=[400, 400]),
        5e-4, {},
    ),
    # DraftNet's 400-d representation of the collection.
    "dn_hidden_only": (
        sd.WinrateTransferNet,
        dict(feature_modes=["hidden"], use_collection=False, hidden_dims=[400, 400]),
        5e-4, {},
    ),
    "dn_hidden_collection": (
        sd.WinrateTransferNet,
        dict(feature_modes=["hidden"], use_collection=True, hidden_dims=[400, 400]),
        5e-4, {},
    ),
    # Per-card contextual pick scores.
    "dn_context_collection": (
        sd.WinrateTransferNet,
        dict(feature_modes=["context"], use_collection=True, hidden_dims=[400, 400]),
        5e-4, {},
    ),
    "dn_all_features": (
        sd.WinrateTransferNet,
        dict(
            feature_modes=["scalars", "hidden", "context"],
            use_collection=True,
            hidden_dims=[400, 400],
        ),
        5e-4, {},
    ),
    # --- Explicit learnable colour-structure term, added to the logit. ---
    "mlp_400x2_colour": (
        sd.WinrateColourNet, dict(hidden_dims=[400, 400]), 5e-4, {},
    ),
    "dn_all_features_colour": (
        sd.WinrateTransferColourNet,
        dict(
            feature_modes=["scalars", "hidden", "context"],
            use_collection=True,
            hidden_dims=[400, 400],
        ),
        5e-4, {},
    ),
    # DraftNet's own trunk as the winrate model's trunk.
    "dn_trunk_frozen": (sd.WinrateFineTuneNet, dict(freeze_trunk=True), 1e-3, {}),
    "dn_trunk_finetune": (sd.WinrateFineTuneNet, dict(freeze_trunk=False), 1e-4, {}),
}

# Architectures that need to know which set's DraftNet to load.
TRANSFER_CLASSES = (sd.WinrateTransferNet, sd.WinrateFineTuneNet)
# Architectures that need the set's card colours.
COLOUR_CLASSES = (sd.WinrateColourNet, sd.WinrateTransferColourNet)


def build_network(
    name, cardnames, picks_per_draft, base_winrate, dropout, input_dropout,
    set_abbreviation="FIN", draft_mode="Premier",
):
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
    if issubclass(network_class, TRANSFER_CLASSES) or issubclass(network_class, COLOUR_CLASSES):
        kwargs["set_abbreviation"] = set_abbreviation
        kwargs["draft_mode"] = draft_mode
    return network_class(**kwargs)


def is_transfer(name):
    return issubclass(ARCHITECTURES[name][0], TRANSFER_CLASSES)


# Caching costs (n_drafts x picks x n_features) floats, so only the compact feature sets
# are worth holding; the 400-d hidden and 358-d context vectors are recomputed per batch.
MAX_CACHED_FEATURES = 32


def cacheable_feature_modes(name):
    """Feature modes to precompute for this architecture, or None to compute per batch."""
    network_class, fixed, _, _ = ARCHITECTURES[name]
    if not issubclass(network_class, sd.WinrateTransferNet):
        return None
    modes = fixed.get("feature_modes", ["scalars"])
    if modes == ["scalars"]:
        return modes
    return None


def train_overrides(name, picks_per_draft):
    overrides = dict(ARCHITECTURES[name][3])
    if overrides.pop("train_n_picks_final_only", False):
        overrides["train_n_picks"] = [picks_per_draft]
    return overrides


def learning_rate(name):
    return ARCHITECTURES[name][2]

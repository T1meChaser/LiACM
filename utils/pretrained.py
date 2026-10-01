"""Fixed inference configuration for the published pure state dictionaries."""

from copy import deepcopy


RELEASE_WEIGHTS = {
    "FORD": {
        "filename": "LiACM_Pro_FORD.pt",
        "sha256": "53123f90ee1f210a735a9e6e05921894e8db9fba06d2541e44f249b43eabb717",
    },
    "SKITTI": {
        "filename": "LiACM_Pro_SemanticKITTI.pt",
        "sha256": "5272e9c88f813df864a4b6e3607ed0363884cdbb3d46009d6680988acc65b63f",
    },
}

_MODEL_KWARGS = {
    "channels": 64,
    "kernel_size": 3,
    "context_layers": 3,
    "prior_context_layers": 1,
    "use_cross_layer_reuse": 1,
    "carry_feature_limit": 2048.0,
    "ancestor_depth": 5,
    "num_experts": 4,
    "window_size": 8,
    "base_cut": 64,
    "q_base": 4.0,
    "depth_offsets": [0, 1, 2, 3, 4, 5],
    # Constructor fields also participate in the existing bitstream fingerprint.
    "prefix_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "coord_shift_mm": 131072.0,
    "range_bins": 16,
    "beam_bins": 64,
    "max_range_m": 120.0,
    "beam_min_deg": -25.0,
    "beam_max_deg": 5.0,
    "use_prefix_loss": 1,
    "use_scale_context": 1,
    "use_lidar_prior": 1,
    "use_octant_prior": 1,
    "use_geo_modulation": 1,
    "use_ancestor_context": 1,
    "use_stage_memory": 1,
    "num_causal_stages": 8,
    "stage_schedule": "modulo",
    "debug_coord_check": 0,
    "use_macro_context": 1,
    "macro_window_size": 512,
}


def released_model_kwargs():
    """Return an independent copy, preserving all fingerprint-relevant values."""
    return deepcopy(_MODEL_KWARGS)

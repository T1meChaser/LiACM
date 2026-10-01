"""Shared configuration and runtime helpers for LiACM.

English: centralizes dataset defaults, path resolution, checkpoint loading,
runtime setup, and model-argument construction shared by entry points.
"""

import os
import io
import math
import hashlib
import json
import random
import warnings
from glob import glob
from typing import Dict, Iterable, List

import numpy as np
import torch
from torchsparse.nn import functional as F
from utils.defaults import DATASET_Q_BASE_MM
from utils.pretrained import RELEASE_WEIGHTS, released_model_kwargs


SKITTI_MODE = "SKITTI"
SEMANTICKITTI_MODE = "SEMANTICKITTI"
FORD_MODE = "FORD"

DEFAULT_SKITTI_ROOT = os.environ.get("LIACM_SKITTI_ROOT", "")
DEFAULT_FORD_ROOT = os.environ.get("LIACM_FORD_ROOT", "")

SKITTI_TRAIN_SEQS = [f"{idx:02d}" for idx in range(0, 11)]
SKITTI_TEST_SEQS = [f"{idx:02d}" for idx in range(11, 22)]
FORD_TRAIN_SEQS = ("Ford_01_q_1mm",)
FORD_TEST_SEQS = ("Ford_02_q_1mm", "Ford_03_q_1mm")
CHECKPOINT_FORMAT = "LiACM_Pro"
CHECKPOINT_SCHEMA = "LiACM_Pro-v2"


def normalize_dataset_mode(mode: str) -> str:
    mode_norm = str(mode).strip().upper()
    if mode_norm in {SKITTI_MODE, SEMANTICKITTI_MODE}:
        return SKITTI_MODE
    if mode_norm == FORD_MODE:
        return FORD_MODE
    raise ValueError(f"Unsupported dataset_mode={mode}. Expected one of: SKITTI, FORD.")


def normalize_split(split: str) -> str:
    split_norm = str(split).strip().lower()
    if split_norm in {"train", "training"}:
        return "train"
    if split_norm in {"test", "testing", "val", "valid", "validation"}:
        return "test"
    raise ValueError(f"Unsupported split={split}. Expected train/training or test/testing.")


def is_disabled_path_arg(value) -> bool:
    value_norm = str(value).strip().lower()
    return value_norm in {"", "0", "none", "null", "false"}


def default_is_data_pre_quantized(dataset_mode: str) -> int:
    return 1 if normalize_dataset_mode(dataset_mode) == FORD_MODE else 0


def default_q_base(dataset_mode: str) -> float:
    return DATASET_Q_BASE_MM[normalize_dataset_mode(dataset_mode)]


def apply_auto_data_pre_quantized(args):
    if int(args.is_data_pre_quantized) < 0:
        args.is_data_pre_quantized = default_is_data_pre_quantized(args.dataset_mode)
    return args


def apply_auto_q_base(args):
    if getattr(args, "q_base", None) is None:
        args.q_base = default_q_base(args.dataset_mode)
    return args


def parse_int_list(value, default: Iterable[int] = None) -> List[int]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return [int(item) for item in (default or [])]
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = str(value).replace(";", ",").split(",")
    return [int(str(item).strip()) for item in raw_items if str(item).strip()]


def parse_float_list(value, default: Iterable[float] = None) -> List[float]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return [float(item) for item in (default or [])]
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = str(value).replace(";", ",").split(",")
    return [float(str(item).strip()) for item in raw_items if str(item).strip()]


def parse_sequence_arg(value, default_seqs: Iterable[str]) -> List[str]:
    if value is None or is_disabled_path_arg(value):
        return [str(seq) for seq in default_seqs]
    if isinstance(value, (list, tuple, set)):
        seqs = value
    else:
        seqs = str(value).replace(";", ",").split(",")
    return [str(seq).strip() for seq in seqs if str(seq).strip()]


def configure_runtime(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings(
        "ignore",
        message=".*cumsum_cuda_kernel does not have a deterministic implementation.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Deterministic behavior was enabled.*CuBLAS.*",
        category=UserWarning,
    )

    conv_config = F.conv_config.get_default_conv_config()
    conv_config.kmap_mode = "hashmap"
    F.conv_config.set_global_conv_config(conv_config)


def build_model_kwargs(args) -> Dict[str, float]:
    depth_offsets = parse_int_list(getattr(args, "depth_offsets", "0,1,2,3,4,5"), default=[0, 1, 2, 3, 4, 5])
    prefix_weights = parse_float_list(
        getattr(args, "prefix_weights", "1,0,0,0,0,0"),
        default=[1, 0, 0, 0, 0, 0],
    )
    kwargs = {
        "channels": int(args.channels),
        "kernel_size": int(args.kernel_size),
        "context_layers": int(args.context_layers),
        "prior_context_layers": int(getattr(args, 'prior_context_layers', 1)),
        "use_cross_layer_reuse": int(getattr(args, 'use_cross_layer_reuse', 1)),
        "ancestor_depth": int(args.ancestor_depth),
        "num_experts": int(args.num_experts),
        "window_size": int(args.window_size),
        "base_cut": int(args.base_cut),
        "q_base": float(args.q_base),
        "depth_offsets": depth_offsets,
        "prefix_weights": prefix_weights,
        "coord_shift_mm": float(args.coord_shift_mm),
        "range_bins": int(args.range_bins),
        "beam_bins": int(args.beam_bins),
        "max_range_m": float(args.max_range_m),
        "beam_min_deg": float(args.beam_min_deg),
        "beam_max_deg": float(args.beam_max_deg),
        "use_prefix_loss": int(getattr(args, "use_prefix_loss", 1)),
        "use_scale_context": int(getattr(args, "use_scale_context", 1)),
        "use_lidar_prior": int(getattr(args, "use_lidar_prior", 1)),
        "use_octant_prior": int(getattr(args, "use_octant_prior", 1)),
        "use_geo_modulation": int(getattr(args, "use_geo_modulation", 1)),
        "use_ancestor_context": int(getattr(args, "use_ancestor_context", 1)),
        "use_stage_memory": int(getattr(args, "use_stage_memory", 1)),
        "num_causal_stages": int(getattr(args, "num_causal_stages", 8)),
        "stage_schedule": str(getattr(args, "stage_schedule", "modulo")),
        "debug_coord_check": int(getattr(args, "debug_coord_check", 0)),
        "use_macro_context": int(getattr(args, "use_macro_context", 1)),
        "macro_window_size": int(getattr(args, "macro_window_size", 512)),
    }
    limit = float(getattr(args, 'carry_feature_limit', 0.))
    kwargs['carry_feature_limit'] = limit
    return kwargs


def load_checkpoint(ckpt_path: str, dataset_mode: str = None):
    """Load released pure weights, or an explicit tensor-only inference checkpoint."""
    with open(ckpt_path, "rb") as fp:
        payload = fp.read()
    try:
        ckpt = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Unable to safely load model weights. Use the published LiACM_Pro_*.pt files "
            "or a tensor-only inference checkpoint, not a full training checkpoint."
        ) from exc

    if isinstance(ckpt, dict) and ckpt and all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in ckpt.items()
    ):
        file_sha = hashlib.sha256(payload).hexdigest()
        release_mode = next(
            (mode for mode, info in RELEASE_WEIGHTS.items() if info["sha256"] == file_sha),
            None,
        )
        if release_mode is None:
            raise RuntimeError(
                "Unrecognized pure state_dict: its SHA256 does not match the published weights. "
                "Restore the original release file. Custom weights require an explicit inference "
                "checkpoint with model_kwargs; architecture is never guessed from tensor shapes."
            )
        if dataset_mode is not None and normalize_dataset_mode(dataset_mode) != release_mode:
            raise ValueError(
                f"Dataset/weights mismatch: requested {normalize_dataset_mode(dataset_mode)}, "
                f"but this file contains the released {release_mode} model."
            )
        ckpt = {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "model": ckpt,
            "model_kwargs": released_model_kwargs(),
        }
    validate_checkpoint_schema(ckpt)
    return ckpt


def validate_checkpoint_schema(ckpt, require_optimizer: bool = False):
    required_keys = {"checkpoint_format", "checkpoint_schema", "model", "model_kwargs"}
    if require_optimizer:
        required_keys.update({"epoch", "global_step", "optimizer", "best_train_loss", "training_config", "rng_states"})
    if not isinstance(ckpt, dict):
        raise RuntimeError("Invalid LiACM checkpoint: expected a checkpoint dict.")
    missing = sorted(required_keys - set(ckpt.keys()))
    if missing:
        raise RuntimeError(f"Invalid LiACM checkpoint: missing keys: {missing}")
    if ckpt.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"Invalid LiACM checkpoint format: {ckpt.get('checkpoint_format')!r}. "
            f"Expected {CHECKPOINT_FORMAT!r}."
        )
    schema = ckpt.get("checkpoint_schema")
    if schema not in {"LiACM_Pro-v1", CHECKPOINT_SCHEMA}:
        raise RuntimeError(
            f"Invalid LiACM checkpoint schema: {ckpt.get('checkpoint_schema')!r}. "
            f"Expected current schema {CHECKPOINT_SCHEMA!r}."
        )
    kwargs = ckpt.get('model_kwargs')
    if not isinstance(kwargs, dict):
        raise RuntimeError('Invalid model_kwargs: expected a dict')
    limit = float(kwargs.get('carry_feature_limit', 0.))
    if not math.isfinite(limit) or limit < 0:
        raise RuntimeError('Invalid carry_feature_limit')
    if schema == 'LiACM_Pro-v1' and limit != 0:
        raise RuntimeError('Bounded state requires LiACM_Pro-v2; older codecs must not ignore it')
    if schema == CHECKPOINT_SCHEMA and 'carry_feature_limit' not in kwargs:
        raise RuntimeError('LiACM_Pro-v2 requires an explicit carry_feature_limit')


def load_ckpt(module: torch.nn.Module, ckpt):
    validate_checkpoint_schema(ckpt)
    module.load_state_dict(ckpt["model"], strict=True)


def resolve_model_kwargs(ckpt, args=None):
    validate_checkpoint_schema(ckpt)

    model_kwargs = dict(ckpt["model_kwargs"])
    from inspect import signature
    from model.liacm_network import Network
    required_keys = set(signature(Network.__init__).parameters) - {'self'}
    unexpected = sorted(set(model_kwargs) - required_keys)
    if unexpected:
        raise RuntimeError(f'Unsupported model_kwargs keys: {unexpected}')
    required_keys.discard('carry_feature_limit')
    missing = sorted(required_keys - set(model_kwargs.keys()))
    if missing:
        raise RuntimeError(f"Invalid LiACM checkpoint: missing model_kwargs keys: {missing}")

    return {
        "channels": int(model_kwargs["channels"]),
        "kernel_size": int(model_kwargs["kernel_size"]),
        "context_layers": int(model_kwargs["context_layers"]),
        "prior_context_layers": int(model_kwargs['prior_context_layers']),
        "use_cross_layer_reuse": int(model_kwargs['use_cross_layer_reuse']),
        "carry_feature_limit": float(model_kwargs.get('carry_feature_limit', 0.)),
        "ancestor_depth": int(model_kwargs["ancestor_depth"]),
        "num_experts": int(model_kwargs["num_experts"]),
        "window_size": int(model_kwargs["window_size"]),
        "base_cut": int(model_kwargs["base_cut"]),
        "q_base": float(model_kwargs["q_base"]),
        "depth_offsets": parse_int_list(model_kwargs["depth_offsets"], default=[0, 1, 2, 3, 4, 5]),
        "prefix_weights": parse_float_list(
            model_kwargs["prefix_weights"],
            default=[1, 0, 0, 0, 0, 0],
        ),
        "coord_shift_mm": float(model_kwargs["coord_shift_mm"]),
        "range_bins": int(model_kwargs["range_bins"]),
        "beam_bins": int(model_kwargs["beam_bins"]),
        "max_range_m": float(model_kwargs["max_range_m"]),
        "beam_min_deg": float(model_kwargs["beam_min_deg"]),
        "beam_max_deg": float(model_kwargs["beam_max_deg"]),
        "use_prefix_loss": int(model_kwargs["use_prefix_loss"]),
        "use_scale_context": int(model_kwargs["use_scale_context"]),
        "use_lidar_prior": int(model_kwargs["use_lidar_prior"]),
        "use_octant_prior": int(model_kwargs["use_octant_prior"]),
        "use_geo_modulation": int(model_kwargs["use_geo_modulation"]),
        "use_ancestor_context": int(model_kwargs["use_ancestor_context"]),
        "use_stage_memory": int(model_kwargs["use_stage_memory"]),
        "num_causal_stages": int(model_kwargs["num_causal_stages"]),
        "stage_schedule": str(model_kwargs["stage_schedule"]),
        "debug_coord_check": int(model_kwargs["debug_coord_check"]),
        "use_macro_context": int(model_kwargs["use_macro_context"]),
        "macro_window_size": int(model_kwargs["macro_window_size"]),
    }


def count_parameters(module):
    return sum(param.numel() for param in module.parameters())


def model_fingerprint(checkpoint):
    """Bind a stream to its exact learned weights and architecture, not a filename."""
    validate_checkpoint_schema(checkpoint)
    digest = hashlib.sha256(json.dumps(checkpoint['model_kwargs'], sort_keys=True).encode('utf-8'))
    for name, value in sorted(checkpoint['model'].items()):
        digest.update(name.encode('utf-8'))
        tensor = value.detach().cpu().contiguous()
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode('ascii'))
        digest.update(tensor.numpy().tobytes())
    return digest.digest()


def count_trainable_parameters(module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def _glob_sorted(pattern: str) -> List[str]:
    return sorted(glob(pattern, recursive=True))


def extract_sample_stem(file_path: str) -> str:
    return os.path.splitext(os.path.basename(str(file_path)))[0]


def extract_semkitti_sequence_id(file_path: str) -> str:
    parts = os.path.normpath(str(file_path)).split(os.sep)
    if "sequences" in parts:
        seq_idx = parts.index("sequences") + 1
        if seq_idx < len(parts):
            return str(parts[seq_idx])

    parent = os.path.basename(os.path.dirname(str(file_path)))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(str(file_path))))
    if parent.lower() == "velodyne" and grandparent:
        return str(grandparent)
    return ""


def extract_ford_sequence_id(file_path: str) -> str:
    parts = os.path.normpath(str(file_path)).split(os.sep)
    for part in reversed(parts[:-1]):
        if part.lower().startswith("ford_"):
            return str(part)
    return ""


def build_sample_id(file_path: str, dataset_mode: str) -> str:
    stem = extract_sample_stem(file_path)
    mode = normalize_dataset_mode(dataset_mode)
    if mode == SKITTI_MODE:
        seq_id = extract_semkitti_sequence_id(file_path)
        if seq_id:
            return f"{seq_id}_{stem}"
    if mode == FORD_MODE:
        seq_id = extract_ford_sequence_id(file_path)
        if seq_id:
            return f"{seq_id}_{stem}"
    return stem


def match_valid_sample(file_path: str, dataset_mode: str, valid_sample_names) -> bool:
    sample_id = build_sample_id(file_path, dataset_mode)
    stem = extract_sample_stem(file_path)
    norm_path = os.path.normpath(str(file_path))
    return sample_id in valid_sample_names or stem in valid_sample_names or norm_path in valid_sample_names


def collect_semkitti_files(root: str, seq_ids: Iterable[str]) -> List[str]:
    file_path_ls: List[str] = []
    for seq_id in seq_ids:
        pattern = os.path.join(root, str(seq_id), "velodyne", "*.bin")
        file_path_ls.extend(_glob_sorted(pattern))
    return file_path_ls


def collect_ford_files(root: str, seq_ids: Iterable[str]) -> List[str]:
    file_path_ls: List[str] = []
    for seq_id in seq_ids:
        seq_id = str(seq_id).strip()
        if not seq_id:
            continue
        if os.path.isabs(seq_id):
            candidate_dirs = [seq_id]
        else:
            short_seq_id = seq_id.replace("_q_1mm", "")
            q1mm_seq_id = seq_id if seq_id.endswith("_q_1mm") else f"{seq_id}_q_1mm"
            candidate_dirs = []
            for outer_dir in dict.fromkeys([q1mm_seq_id, short_seq_id]):
                for inner_dir in dict.fromkeys([outer_dir, short_seq_id]):
                    candidate_dirs.append(os.path.join(root, outer_dir, inner_dir))
                candidate_dirs.append(os.path.join(root, outer_dir))
        patterns = []
        for candidate_dir in candidate_dirs:
            patterns.extend([os.path.join(candidate_dir, "*.ply"), os.path.join(candidate_dir, "*.PLY")])
        for pattern in patterns:
            files = _glob_sorted(pattern)
            if files:
                file_path_ls.extend(files)
                break
    return file_path_ls


def resolve_dataset_files(
    dataset_mode: str,
    split: str,
    skitti_root: str = DEFAULT_SKITTI_ROOT,
    ford_root: str = DEFAULT_FORD_ROOT,
    ford_train_seqs=FORD_TRAIN_SEQS,
    ford_test_seqs=FORD_TEST_SEQS,
) -> np.ndarray:
    mode = normalize_dataset_mode(dataset_mode)
    split_norm = normalize_split(split)

    if mode == FORD_MODE:
        default_seqs = FORD_TRAIN_SEQS if split_norm == "train" else FORD_TEST_SEQS
        seq_arg = ford_train_seqs if split_norm == "train" else ford_test_seqs
        seq_ids = parse_sequence_arg(seq_arg, default_seqs)
        return np.array(collect_ford_files(ford_root, seq_ids))

    seq_ids = SKITTI_TRAIN_SEQS if split_norm == "train" else SKITTI_TEST_SEQS
    return np.array(collect_semkitti_files(skitti_root, seq_ids))


def default_save_folder(prefix: str, dataset_mode: str) -> str:
    mode = normalize_dataset_mode(dataset_mode)
    return os.path.join(".", "my_model", f"{prefix}_{mode}")


def checkpoint_candidates(dataset_mode: str, primary_prefix: str = "LiACM_Pro", filename: str = "best_model.pt") -> List[str]:
    mode = normalize_dataset_mode(dataset_mode)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(project_root, RELEASE_WEIGHTS[mode]["filename"]),
    ]
    out = []
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm not in out:
            out.append(norm)
    return out


def resolve_checkpoint_path(ckpt_path: str, dataset_mode: str, primary_prefix: str = "LiACM_Pro", filename: str = "best_model.pt") -> str:
    if ckpt_path and not is_disabled_path_arg(ckpt_path):
        return ckpt_path
    candidates = checkpoint_candidates(dataset_mode, primary_prefix=primary_prefix, filename=filename)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def default_output_folder(prefix: str, dataset_mode: str, suffix: str) -> str:
    mode = normalize_dataset_mode(dataset_mode)
    return os.path.join(".", "my_test", f"{prefix}_{mode}_{suffix}")

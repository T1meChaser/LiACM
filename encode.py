"""LiACM_Pro compression entry point.

English: loads a trained checkpoint, predicts entropy probabilities, arithmetic
encodes raw_nibble symbols, and writes LiACM_Pro bitstreams.
"""

import os
import sys

# Keep project root importable when this root entry is launched directly.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import argparse
import time
from glob import glob

import numpy as np
import torch
import torchac
from torchsparse import SparseTensor
from tqdm import tqdm

import utils.entropy_coding as op
from utils.config import (
    DEFAULT_FORD_ROOT,
    DEFAULT_SKITTI_ROOT,
    apply_auto_data_pre_quantized,
    build_sample_id,
    configure_runtime,
    default_output_folder,
    load_checkpoint,
    load_ckpt,
    model_fingerprint,
    normalize_dataset_mode,
    normalize_split,
    resolve_dataset_files,
    resolve_checkpoint_path,
    resolve_model_kwargs,
    validate_checkpoint_schema,
)
from data.point_dataset import read_point_cloud_single
from model.inference import InferenceNetwork as Network
from model.entropy_context import (
    LiACM_MAGIC,
    sort_by_morton,
)


def encode_symbols(prob: torch.Tensor, symbols: torch.Tensor) -> bytes:
    if prob.shape[0] == 0:
        return b""
    cdf = torch.cat((prob[:, 0:1] * 0, prob.cumsum(dim=-1)), dim=-1)
    cdf = torch.clamp(cdf, min=0, max=1)
    cdf_norm = op._convert_to_int_and_normalize(cdf, True).cpu()
    symbols_i16 = symbols.view(-1).to(torch.int16).cpu()
    return torchac.encode_int16_normalized_cdf(cdf_norm, symbols_i16)


def resolve_input_files(args):
    if args.input_glob:
        return sorted(glob(args.input_glob, recursive=True))
    return resolve_dataset_files(
        dataset_mode=args.dataset_mode,
        split=args.dataset_split,
        skitti_root=args.skitti_root,
        ford_root=args.ford_root,
        ford_train_seqs=args.ford_train_seqs,
        ford_test_seqs=args.ford_test_seqs,
    ).tolist()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        prog="encode.py",
        description="Compress point cloud geometry with LiACM_Pro.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset_mode", default="SKITTI", help="Dataset mode: SKITTI or FORD.")
    parser.add_argument("--dataset_split", default="test", choices=["train", "test", "training", "testing"])
    parser.add_argument("--input_glob", default="")
    parser.add_argument("--output_folder", default="")
    parser.add_argument(
        "--skitti_root",
        default=DEFAULT_SKITTI_ROOT,
        help="SemanticKITTI sequences root. May also be set with LIACM_SKITTI_ROOT.",
    )
    parser.add_argument(
        "--ford_root",
        default=DEFAULT_FORD_ROOT,
        help="Ford dataset root. May also be set with LIACM_FORD_ROOT.",
    )
    parser.add_argument("--ford_train_seqs", default="Ford_01_q_1mm")
    parser.add_argument("--ford_test_seqs", default="Ford_02_q_1mm,Ford_03_q_1mm")
    parser.add_argument(
        "--is_data_pre_quantized",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="-1 means auto: SKITTI=0, FORD q1mm=1.",
    )
    parser.add_argument("--q_base", default=None, type=float)
    parser.add_argument("--depth_offset", default=0, type=int)

    parser.add_argument("--ckpt", default="")

    parser.add_argument("--num_samples", default=-1, type=int)
    args = parser.parse_args()
    if args.depth_offset < 0 or args.num_samples == 0 or args.num_samples < -1:
        parser.error('depth_offset must be nonnegative; num_samples must be -1 or positive')
    args.dataset_mode = normalize_dataset_mode(args.dataset_mode)
    apply_auto_data_pre_quantized(args)
    args.ckpt = resolve_checkpoint_path(args.ckpt, args.dataset_mode, primary_prefix="LiACM_Pro")
    ckpt = load_checkpoint(args.ckpt, dataset_mode=args.dataset_mode)
    validate_checkpoint_schema(ckpt)
    saved_q_base = float(ckpt['model_kwargs']['q_base'])
    if args.q_base is not None and float(args.q_base) != saved_q_base:
        raise ValueError('q_base must match the trained checkpoint; use depth_offset to select a rate')
    args.q_base = saved_q_base

    split_tag = normalize_split(args.dataset_split)
    if args.output_folder == "":
        q_eff = float(args.q_base) * float(2 ** max(int(args.depth_offset), 0))
        args.output_folder = default_output_folder("LiACM_Pro", args.dataset_mode, f"{split_tag}_q{int(q_eff)}mm_compressed")
    os.makedirs(args.output_folder, exist_ok=True)
    configure_runtime(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path_ls = resolve_input_files(args)
    if args.num_samples != -1:
        np.random.shuffle(file_path_ls)
        file_path_ls = file_path_ls[: args.num_samples]
    if len(file_path_ls) == 0:
        raise RuntimeError("No input point clouds found for compression.")

    model_kwargs = resolve_model_kwargs(ckpt, args)
    fingerprint = model_fingerprint(ckpt)
    effective_q_base = float(model_kwargs["q_base"])
    effective_base_cut = int(model_kwargs["base_cut"])
    effective_coord_shift_mm = float(model_kwargs["coord_shift_mm"])
    net = Network(**model_kwargs).to(device)
    load_ckpt(net, ckpt)
    net.set_runtime_geometry(q_base=effective_q_base, coord_shift_mm=effective_coord_shift_mm)
    net.eval()
    print("Dataset mode:", args.dataset_mode)
    print("Checkpoint:", os.path.abspath(args.ckpt))
    print("Output folder:", os.path.abspath(args.output_folder))
    print("is_data_pre_quantized:", int(args.is_data_pre_quantized))
    print("q_base:", effective_q_base)
    print("depth_offset:", int(args.depth_offset))

    random_coords = torch.randint(low=0, high=65536, size=(4096, 3), device=device).int()
    dummy_input = SparseTensor(
        coords=torch.cat((random_coords[:, 0:1] * 0, random_coords), dim=-1),
        feats=torch.ones((4096, 1), device=device),
    ).to(device)
    net.set_runtime_geometry(q_base=effective_q_base, coord_shift_mm=effective_coord_shift_mm)
    net(dummy_input, base_cut=effective_base_cut)
    net.reset_frame_state()
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    enc_time_ls = []
    total_bits = 0.0
    total_points = 0

    with torch.no_grad():
        for file_path in tqdm(file_path_ls, desc="LiACM_Pro compress", unit="file"):
            net.reset_frame_state()
            file_name = build_sample_id(file_path, args.dataset_mode)
            compressed_file_path = os.path.join(args.output_folder, file_name + ".bin")
            net.set_runtime_geometry(q_base=effective_q_base, coord_shift_mm=effective_coord_shift_mm)

            xyz_raw = torch.tensor(read_point_cloud_single(file_path), dtype=torch.float32)
            raw_num_points = int(xyz_raw.shape[0])

            if args.is_data_pre_quantized:
                xyz = xyz_raw + effective_coord_shift_mm
            else:
                xyz = xyz_raw / 0.001 + effective_coord_shift_mm

            xyz = torch.round(xyz / effective_q_base).int()
            coords = torch.cat((xyz[:, 0:1] * 0, xyz), dim=-1).int()
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            x = SparseTensor(coords=coords, feats=feats).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            enc_time_start = time.time()

            data_ls = net.build_hierarchy(x, base_cut=effective_base_cut)
            full_num_depths = len(data_ls) - 1
            requested_depth_offset = max(int(args.depth_offset), 0)
            target_num_depths = max(full_num_depths - requested_depth_offset, 0)
            effective_depth_offset = full_num_depths - target_num_depths
            byte_stream_ls = []
            ancestor_cache = []

            for depth in range(target_num_depths):
                x_c, x_o = data_ls[depth]
                gt_x_up_c, gt_x_up_o = data_ls[depth + 1]
                gt_x_up_c, gt_x_up_o = sort_by_morton(gt_x_up_c, gt_x_up_o)

                x_prior, prior_geo = net.prior_branch(x_c, x_o, depth, full_num_depths=full_num_depths)
                ancestor_cache.append(net._make_cache_entry(x_c, x_prior.feats, prior_geo, depth))

                x_up, target_geo = net.target_branch(x_c, x_o, x_prior.feats, depth, full_num_depths=full_num_depths)
                net.assert_aligned_coords(x_up.coords, gt_x_up_c, depth=depth, context="compress")
                mod_feats, _, gate_ctx = net.build_geo_context(x_up, target_geo, depth, ancestor_cache)
                stage = net.build_stage_partition(x_up.coords, depth=depth, full_num_depths=full_num_depths)
                gt_primary, gt_secondary = net.raw_occ_to_symbols(
                    gt_x_up_o,
                    x_up.coords,
                    depth=depth,
                    full_num_depths=full_num_depths,
                )
                token_sum = mod_feats.new_zeros((stage["num_windows"], net.channels))
                token_count = mod_feats.new_zeros((stage["num_windows"], 1))

                for stage_idx in range(net.num_causal_stages):
                    stage_mask = net.select_stage(stage, stage_idx)
                    if net.stage_empty(stage, stage_idx, stage_mask):
                        byte_stream_ls.extend((b"", b""))
                        continue

                    feat_s = mod_feats[stage_mask].contiguous()
                    gate_s = gate_ctx[stage_mask].contiguous()
                    if net.use_stage_memory and token_sum.shape[0] > 0 and int(stage_idx) > 0:
                        mem_s = net.memory_from_state(token_sum, token_count)[stage["window_ids"][stage_mask].contiguous().long()]
                        feat_s, gate_s = net.fuse_stage_context(feat_s, gate_s, mem_s)
                    primary_s = gt_primary[stage_mask].contiguous().long()
                    secondary_s = gt_secondary[stage_mask].contiguous().long()
                    prob_primary = net.predict_primary(feat_s, gate_s)
                    prob_secondary = net.predict_secondary(feat_s, primary_s, gate_s)
                    stream_primary = encode_symbols(prob_primary, primary_s)
                    stream_secondary = encode_symbols(prob_secondary, secondary_s)
                    new_sum, new_count = net.build_stage_token_sums(
                        primary_s,
                        secondary_s,
                        gt_x_up_o.view(-1).long()[stage_mask].contiguous(),
                        feat_s,
                        gate_s,
                        stage["window_ids"][stage_mask].contiguous(),
                        stage["window_pos"][stage_mask].contiguous(),
                        stage["stage_ids"][stage_mask].contiguous(),
                        stage["num_windows"],
                    )

                    token_sum = token_sum + new_sum
                    token_count = token_count + new_count
                    byte_stream_ls.extend((stream_primary, stream_secondary))

            byte_stream = op.pack_byte_stream_ls(byte_stream_ls)

            if device.type == "cuda":
                torch.cuda.synchronize()
            enc_time_end = time.time()

            base_x_coords, base_x_feats = data_ls[0]
            base_x_len = int(base_x_coords.shape[0])
            base_x_coords = base_x_coords[:, 1:].cpu().numpy().astype(np.int32)
            base_x_feats = base_x_feats.cpu().numpy().astype(np.uint8)

            with open(compressed_file_path, "wb") as f:
                header_f32 = np.array([effective_q_base, effective_coord_shift_mm], dtype=np.float32)
                header_i32 = np.array(
                    [
                        base_x_len,
                        int(effective_base_cut),
                        int(args.is_data_pre_quantized),
                        int(full_num_depths),
                        int(target_num_depths),
                        int(effective_depth_offset),
                        1,
                        int(net.num_causal_stages),
                        int(net.symbol_id),
                        int(net.stage_schedule_id),
                        int(net.streams_per_stage * net.num_causal_stages),
                    ],
                    dtype=np.int32,
                )
                f.write(LiACM_MAGIC)
                f.write(fingerprint)
                f.write(header_f32.tobytes())
                f.write(header_i32.tobytes())
                f.write(base_x_coords.tobytes())
                f.write(base_x_feats.tobytes())
                f.write(byte_stream)

            enc_time_ls.append(enc_time_end - enc_time_start)
            file_bits = op.get_file_size_in_bits(compressed_file_path)
            total_bits += float(file_bits)
            total_points += int(raw_num_points)

    max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
    print(
        "Total: {total_n:d} | Dataset BPP:{bpp:.3f} | Encode time:{enc_time:.3f} | Max GPU Memory:{memory:.2f}MB".format(
            total_n=len(enc_time_ls),
            bpp=total_bits / total_points if total_points > 0 else 0.0,
            enc_time=np.array(enc_time_ls).mean() if len(enc_time_ls) else 0.0,
            memory=max_memory,
        )
    )


if __name__ == "__main__":
    main()



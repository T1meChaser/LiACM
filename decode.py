"""LiACM_Pro decompression entry point.

English: reads LiACM_Pro bitstreams, rebuilds the same hierarchy/context path, and
arithmetic decodes raw_nibble symbols back to point coordinates.
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
    configure_runtime,
    default_output_folder,
    load_checkpoint,
    load_ckpt,
    model_fingerprint,
    normalize_dataset_mode,
    normalize_split,
    resolve_checkpoint_path,
    resolve_model_kwargs,
    validate_checkpoint_schema,
)
from model.inference import InferenceNetwork as Network
from model.entropy_context import (
    LiACM_MAGIC,
    infer_num_depths,
)

LiACM_HEADER_F32_LEN = 2
LiACM_HEADER_I32_LEN = 11
LiACM_SUPPORTED_MAGIC = {LiACM_MAGIC}


def prob_to_cdf_int16(prob: torch.Tensor) -> torch.Tensor:
    cdf = torch.cat((prob[:, 0:1] * 0, prob.cumsum(dim=-1)), dim=-1)
    cdf = torch.clamp(cdf, min=0, max=1)
    return op._convert_to_int_and_normalize(cdf, True).cpu()


def decode_symbols(prob: torch.Tensor, byte_stream: bytes, device: torch.device) -> torch.Tensor:
    cdf_int = prob_to_cdf_int16(prob)
    return torchac.decode_int16_normalized_cdf(cdf_int, byte_stream).to(device).long().unsqueeze(-1)


def save_ply_ascii_geo(points: np.ndarray, file_path: str):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"Expected Nx3 points for PLY output, got shape={points.shape}.")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {int(points.shape[0])}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for xyz in points:
            f.write(f"{float(xyz[0])} {float(xyz[1])} {float(xyz[2])}\n")


def read_LiACM_header(fp, file_path: str):
    magic = fp.read(4)
    if magic not in LiACM_SUPPORTED_MAGIC:
        expected = ", ".join(sorted(item.decode("ascii") for item in LiACM_SUPPORTED_MAGIC))
        raise RuntimeError(f"Unsupported header in {file_path}, expected one of: {expected}.")

    fingerprint = fp.read(32)
    if len(fingerprint) != 32:
        raise RuntimeError('Truncated checkpoint fingerprint')
    hdr_f32 = np.frombuffer(fp.read(LiACM_HEADER_F32_LEN * 4), dtype=np.float32)
    hdr_i32 = np.frombuffer(fp.read(LiACM_HEADER_I32_LEN * 4), dtype=np.int32)
    if hdr_f32.size != LiACM_HEADER_F32_LEN or hdr_i32.size != LiACM_HEADER_I32_LEN:
        raise RuntimeError(f"Invalid LiACM_Pro header in {file_path}.")

    return {
        "magic": magic,
        "fingerprint": fingerprint,
        "q_base": float(hdr_f32[0]),
        "coord_shift_mm": float(hdr_f32[1]),
        "base_x_len": int(hdr_i32[0]),
        "base_cut": int(hdr_i32[1]),
        "is_data_pre_quantized": int(hdr_i32[2]),
        "full_num_depths": int(hdr_i32[3]),
        "target_num_depths": int(hdr_i32[4]),
        "depth_offset": int(hdr_i32[5]),
        "recon_center_mode": int(hdr_i32[6]),
        "num_causal_stages": int(hdr_i32[7]),
        "symbol_id": int(hdr_i32[8]),
        "stage_schedule_id": int(hdr_i32[9]),
        "streams_per_depth": int(hdr_i32[10]),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        prog="decode.py",
        description="Decompress point cloud geometry with LiACM_Pro.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset_mode", default="SKITTI", help="Dataset mode: SKITTI or FORD.")
    parser.add_argument("--dataset_split", default="test", choices=["train", "test", "training", "testing"])
    parser.add_argument("--input_glob", default="")
    parser.add_argument("--output_folder", default="")

    parser.add_argument(
        "--q_base",
        type=float,
        default=None,
        help="Highest precision lattice size in millimeters.",
    )
    parser.add_argument("--depth_offset", type=int, default=0, help="Used only for default folder names.")
    parser.add_argument("--ckpt", default="")

    args = parser.parse_args()
    args.dataset_mode = normalize_dataset_mode(args.dataset_mode)
    args.ckpt = resolve_checkpoint_path(args.ckpt, args.dataset_mode, primary_prefix="LiACM_Pro")
    ckpt = load_checkpoint(args.ckpt, dataset_mode=args.dataset_mode)
    validate_checkpoint_schema(ckpt)
    saved_q_base = float(ckpt['model_kwargs']['q_base'])
    if args.q_base is not None and float(args.q_base) != saved_q_base:
        raise ValueError('q_base must match checkpoint')
    args.q_base = saved_q_base
    split_tag = normalize_split(args.dataset_split)

    if args.input_glob == "":
        q_eff = float(args.q_base) * float(2 ** max(int(args.depth_offset), 0))
        args.input_glob = os.path.join(
            default_output_folder("LiACM_Pro", args.dataset_mode, f"{split_tag}_q{int(q_eff)}mm_compressed"),
            "*.bin",
        )
    if args.output_folder == "":
        q_eff = float(args.q_base) * float(2 ** max(int(args.depth_offset), 0))
        args.output_folder = default_output_folder("LiACM_Pro", args.dataset_mode, f"{split_tag}_q{int(q_eff)}mm_decompressed")
    os.makedirs(args.output_folder, exist_ok=True)
    file_path_ls = sorted(glob(args.input_glob))
    if len(file_path_ls) == 0:
        raise RuntimeError(f"No compressed files found by pattern: {args.input_glob}")

    configure_runtime(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = resolve_model_kwargs(ckpt, args)
    fingerprint = model_fingerprint(ckpt)
    effective_base_cut = int(model_kwargs["base_cut"])
    effective_coord_shift_mm = float(model_kwargs["coord_shift_mm"])
    net = Network(**model_kwargs).to(device)
    load_ckpt(net, ckpt)
    net.eval()

    random_coords = torch.randint(low=0, high=65536, size=(4096, 3), device=device).int()
    dummy_input = SparseTensor(
        coords=torch.cat((random_coords[:, 0:1] * 0, random_coords), dim=-1),
        feats=torch.ones((4096, 1), device=device),
    ).to(device)
    net.set_runtime_geometry(q_base=float(model_kwargs["q_base"]), coord_shift_mm=effective_coord_shift_mm)
    net(dummy_input, base_cut=effective_base_cut)
    net.reset_frame_state()
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    dec_time_ls = []
    print("Dataset mode:", args.dataset_mode)
    print("Checkpoint:", os.path.abspath(args.ckpt))
    print("Input glob:", args.input_glob)
    print("Output folder:", os.path.abspath(args.output_folder))

    with torch.no_grad():
        for file_path in tqdm(file_path_ls, desc="LiACM_Pro decompress", unit="file"):
            net.reset_frame_state()
            file_name = os.path.splitext(os.path.split(file_path)[-1])[0]
            decompressed_file_path = os.path.join(args.output_folder, file_name + ".ply")

            with open(file_path, "rb") as f:
                hdr = read_LiACM_header(f, file_path)
                if hdr['fingerprint'] != fingerprint:
                    raise RuntimeError('Stream was encoded with different weights or model settings')
                if not (0 <= hdr['target_num_depths'] <= hdr['full_num_depths'] <= 21
                        and hdr['depth_offset'] == hdr['full_num_depths'] - hdr['target_num_depths']
                        and 0 < hdr['base_x_len'] <= os.path.getsize(file_path)//13
                        and hdr['recon_center_mode'] == 1
                        and hdr['is_data_pre_quantized'] in (0, 1)):
                    raise RuntimeError('Invalid stream geometry header')
                if abs(hdr["coord_shift_mm"] - effective_coord_shift_mm) > 1e-6:
                    raise RuntimeError(
                        f"coord_shift_mm mismatch for {file_path}: file header uses {hdr['coord_shift_mm']}, "
                        f"but checkpoint/model expects {effective_coord_shift_mm}."
                    )
                if abs(hdr["q_base"] - float(model_kwargs["q_base"])) > 1e-6:
                    raise RuntimeError(
                        f"q_base mismatch for {file_path}: file header uses {hdr['q_base']}, "
                        f"but checkpoint/model expects {model_kwargs['q_base']}."
                    )
                if int(hdr["symbol_id"]) != int(net.symbol_id):
                    raise RuntimeError(
                        f"symbol id mismatch for {file_path}: file header uses {hdr['symbol_id']}, "
                        f"but checkpoint/model expects {net.symbol_id} ({net.symbol_name})."
                    )
                if int(hdr["stage_schedule_id"]) != int(net.stage_schedule_id):
                    raise RuntimeError(
                        f"stage_schedule mismatch for {file_path}: file header uses {hdr['stage_schedule_id']}, "
                        f"but checkpoint/model expects {net.stage_schedule_id} ({net.stage_schedule})."
                    )
                if int(hdr["num_causal_stages"]) != int(model_kwargs["num_causal_stages"]):
                    raise RuntimeError(
                        f"num_causal_stages mismatch for {file_path}: file header uses {hdr['num_causal_stages']}, "
                        f"but checkpoint/model expects {model_kwargs['num_causal_stages']}."
                    )
                expected_streams_per_depth = int(net.streams_per_stage) * int(model_kwargs["num_causal_stages"])
                if int(hdr["streams_per_depth"]) != expected_streams_per_depth:
                    raise RuntimeError(
                        f"streams_per_depth mismatch for {file_path}: file header uses {hdr['streams_per_depth']}, "
                        f"but LiACM_Pro expects {expected_streams_per_depth}."
                    )
                base_x_len = hdr["base_x_len"]
                base_x_coords = np.frombuffer(f.read(base_x_len * 4 * 3), dtype=np.int32)
                base_x_feats = np.frombuffer(f.read(base_x_len * 1), dtype=np.uint8)
                if base_x_coords.size != base_x_len*3 or base_x_feats.size != base_x_len:
                    raise RuntimeError('Truncated base point cloud')
                byte_stream = f.read()

            if device.type == "cuda":
                torch.cuda.synchronize()
            dec_time_start = time.time()
            net.set_runtime_geometry(q_base=hdr["q_base"], coord_shift_mm=hdr["coord_shift_mm"])

            base_x_coords = torch.tensor(base_x_coords.reshape(-1, 3), device=device, dtype=torch.int32)
            base_x_feats = torch.tensor(base_x_feats.reshape(-1, 1), device=device, dtype=torch.uint8)

            batch_col = torch.zeros((base_x_len, 1), device=device, dtype=torch.int32)
            x = SparseTensor(
                coords=torch.cat((batch_col, base_x_coords), dim=-1),
                feats=base_x_feats,
            ).to(device)

            byte_stream_ls = op.unpack_byte_stream(byte_stream)
            num_depths = infer_num_depths(len(byte_stream_ls), streams_per_depth=hdr["streams_per_depth"])
            if num_depths != int(hdr["target_num_depths"]):
                raise RuntimeError(
                    f"Stream depth mismatch for {file_path}: header target_num_depths={hdr['target_num_depths']}, "
                    f"but packed stream contains {num_depths} depths."
                )
            ancestor_cache = []

            for depth in range(0, num_depths):
                x_o = x.feats.long()
                x_prior, prior_geo = net.prior_branch(x.coords, x_o, depth, full_num_depths=hdr["full_num_depths"])
                ancestor_cache.append(net._make_cache_entry(x.coords, x_prior.feats, prior_geo, depth))

                x_up, target_geo = net.target_branch(x.coords, x_o, x_prior.feats, depth, full_num_depths=hdr["full_num_depths"])
                mod_feats, _, gate_ctx = net.build_geo_context(x_up, target_geo, depth, ancestor_cache)
                stage = net.build_stage_partition(
                    x_up.coords,
                    depth=depth,
                    full_num_depths=hdr["full_num_depths"],
                )

                x_up_o = torch.zeros((x_up.coords.shape[0], 1), device=device, dtype=torch.long)
                token_sum = mod_feats.new_zeros((stage["num_windows"], net.channels))
                token_count = mod_feats.new_zeros((stage["num_windows"], 1))

                for stage_idx in range(net.num_causal_stages):
                    stream_base = depth * hdr["streams_per_depth"] + stage_idx * net.streams_per_stage
                    stream_primary = byte_stream_ls[stream_base + 0]
                    stream_secondary = byte_stream_ls[stream_base + 1]

                    stage_mask = net.select_stage(stage, stage_idx)
                    if net.stage_empty(stage, stage_idx, stage_mask):
                        continue

                    feat_s = mod_feats[stage_mask].contiguous()
                    gate_s = gate_ctx[stage_mask].contiguous()
                    if net.use_stage_memory and token_sum.shape[0] > 0 and int(stage_idx) > 0:
                        mem_s = net.memory_from_state(token_sum, token_count)[stage["window_ids"][stage_mask].contiguous().long()]
                        feat_s, gate_s = net.fuse_stage_context(feat_s, gate_s, mem_s)

                    prob_primary = net.predict_primary(feat_s, gate_s)
                    dec_primary = decode_symbols(prob_primary, stream_primary, device=device)
                    prob_secondary = net.predict_secondary(feat_s, dec_primary, gate_s)
                    dec_secondary = decode_symbols(prob_secondary, stream_secondary, device=device)
                    raw_occ = net.symbols_to_raw_occ(
                        dec_primary,
                        dec_secondary,
                        x_up.coords[stage_mask].contiguous(),
                        depth=depth,
                        full_num_depths=hdr["full_num_depths"],
                    )
                    x_up_o[stage_mask] = raw_occ.view(-1, 1)
                    new_sum, new_count = net.build_stage_token_sums(
                        dec_primary.long(),
                        dec_secondary.long(),
                        raw_occ.long(),
                        feat_s,
                        gate_s,
                        stage["window_ids"][stage_mask].contiguous(),
                        stage["window_pos"][stage_mask].contiguous(),
                        stage["stage_ids"][stage_mask].contiguous(),
                        stage["num_windows"],
                    )

                    token_sum = token_sum + new_sum
                    token_count = token_count + new_count

                x = SparseTensor(coords=x_up.coords, feats=x_up_o).to(device)

            scan = net.fcg(x.coords, x.feats)

            scale = float(2 ** max(int(hdr["depth_offset"]), 0))
            center_offset = 0.5 * (scale - 1.0) if int(hdr["recon_center_mode"]) else 0.0
            scan_xyz = scan[:, 1:].float() * scale + center_offset
            if hdr["is_data_pre_quantized"]:
                scan = scan_xyz * hdr["q_base"] - hdr["coord_shift_mm"]
            else:
                scan = (scan_xyz * hdr["q_base"] - hdr["coord_shift_mm"]) * 0.001

            if device.type == "cuda":
                torch.cuda.synchronize()
            dec_time_end = time.time()
            dec_time_ls.append(dec_time_end - dec_time_start)

            scan_np = scan.float().cpu().numpy()
            save_ply_ascii_geo(scan_np, decompressed_file_path)

    max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
    print(
        "Total: {total_n:d} | Decode Time:{dec_time:.3f} | Max GPU Memory:{memory:.2f}MB".format(
            total_n=len(dec_time_ls),
            dec_time=np.array(dec_time_ls).mean() if len(dec_time_ls) else 0.0,
            memory=max_memory,
        )
    )


if __name__ == "__main__":
    main()



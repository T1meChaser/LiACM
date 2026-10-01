"""End-to-end LiACM_Pro pipeline runner.

English: launches compression, decompression, and evaluation with consistent
paths and quantization settings.
"""

import os
import sys

# Keep project root importable when this root entry is launched directly.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import argparse
import subprocess


def add_arg(cmd, name, value):
    if value is not None and str(value) != "":
        cmd.extend([name, str(value)])


def run_step(cmd, dry_run=False):
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def normalize_dataset_mode(mode):
    mode = str(mode).strip().upper()
    if mode == "SEMANTICKITTI":
        return "SKITTI"
    if mode in {"SKITTI", "FORD"}:
        return mode
    raise ValueError(f"Unsupported dataset_mode={mode}. Expected SKITTI or FORD.")


def normalize_split(split):
    split = str(split).strip().lower()
    if split in {"train", "training"}:
        return "train"
    if split in {"test", "testing", "val", "valid", "validation"}:
        return "test"
    raise ValueError(f"Unsupported dataset_split={split}. Expected train or test.")


def default_output_folder(prefix, dataset_mode, suffix):
    return os.path.join(".", "my_test", f"{prefix}_{normalize_dataset_mode(dataset_mode)}_{suffix}")


def parse_int_list(value):
    return [int(item.strip()) for item in str(value).replace(";", ",").split(",") if item.strip()]


def path_for_rate(path_value, default_path, q_eff, total_rates):
    if not path_value:
        return default_path
    base = path_value
    if total_rates <= 1:
        return base
    root, ext = os.path.splitext(base)
    if ext:
        return f"{root}_q{int(q_eff)}mm{ext}"
    return f"{base}_q{int(q_eff)}mm"


def main():
    parser = argparse.ArgumentParser(
        prog="run_rd_pipeline.py",
        description="Run LiACM_Pro compression, decompression, and evaluation in one command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_mode", default="FORD", help="Dataset mode: SKITTI or FORD.")
    parser.add_argument("--dataset_split", default="test", choices=["train", "test", "training", "testing"])
    parser.add_argument("--ckpt", default="", help="Empty selects the published model in the project root for the dataset.")
    parser.add_argument("--input_glob", default="", help="Optional compression input glob. Empty uses dataset defaults.")
    parser.add_argument("--eval_input_glob", default="", help="Optional reference PLY glob for evaluation.")
    parser.add_argument('--skitti_predata_root', default=os.environ.get('LIACM_SKITTI_PREDATA_ROOT', ''),
                        help='Reference PLY root for SemanticKITTI evaluation, in meters')
    parser.add_argument("--compressed_path", default="")
    parser.add_argument("--decompressed_path", default="")
    parser.add_argument("--output_csv", default="")

    parser.add_argument(
        "--skitti_root",
        default=os.environ.get("LIACM_SKITTI_ROOT", ""),
        help="SemanticKITTI sequences root. May also be set with LIACM_SKITTI_ROOT.",
    )
    parser.add_argument(
        "--ford_root",
        default=os.environ.get("LIACM_FORD_ROOT", ""),
        help="Ford dataset root. May also be set with LIACM_FORD_ROOT.",
    )
    parser.add_argument("--ford_train_seqs", default="Ford_01_q_1mm")
    parser.add_argument("--ford_test_seqs", default="Ford_02_q_1mm,Ford_03_q_1mm")

    parser.add_argument("--is_data_pre_quantized", type=int, default=-1, choices=[-1, 0, 1])
    parser.add_argument("--q_base", type=float, default=None)
    parser.add_argument("--depth_offsets", default="0,1,2,3,4,5")
    parser.add_argument("--num_samples", type=int, default=-1)

    parser.add_argument("--pcc_metric_path", default=os.path.join(_PROJECT_ROOT, "utils/third_party/pc_error_d"))
    parser.add_argument("--resolution", default="auto")
    parser.add_argument("--input_norm", choices=["auto", "none", "ref", "cache"], default="auto")
    parser.add_argument("--normal_path", default="")
    parser.add_argument("--enable_cd", action="store_true")
    parser.add_argument("--cd_mode", choices=["gaem", "mean", "mse"], default="gaem")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--diagnose_scale_samples", type=int, default=3)

    parser.add_argument("--skip_compress", action="store_true")
    parser.add_argument("--skip_decompress", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    mode = normalize_dataset_mode(args.dataset_mode)
    split = normalize_split(args.dataset_split)
    # The trained lattice, not a dataset-specific guess, defines the rate labels.
    from utils.config import load_checkpoint, validate_checkpoint_schema, resolve_checkpoint_path
    args.ckpt = resolve_checkpoint_path(args.ckpt, mode)
    checkpoint = load_checkpoint(args.ckpt, dataset_mode=mode)
    validate_checkpoint_schema(checkpoint)
    saved_q = float(checkpoint['model_kwargs']['q_base'])
    if args.q_base is not None and float(args.q_base) != saved_q:
        raise ValueError('q_base must match the checkpoint')
    args.q_base = saved_q
    script_dir = os.path.dirname(os.path.abspath(__file__))

    depth_offsets = parse_int_list(args.depth_offsets)
    if not depth_offsets or min(depth_offsets) < 0 or len(set(depth_offsets)) != len(depth_offsets):
        raise ValueError('Provide distinct nonnegative depth offsets')
    if not args.skip_eval:
        if not os.access(args.pcc_metric_path, os.X_OK):
            raise RuntimeError('pc_error_d is missing or not executable: '+args.pcc_metric_path)
        if mode == 'SKITTI' and not args.eval_input_glob and not args.skitti_predata_root:
            raise ValueError('SKITTI evaluation needs --eval_input_glob or --skitti_predata_root (reference PLYs in meters)')

    common_dataset_args = [
        "--dataset_mode",
        mode,
        "--dataset_split",
        split,
        "--skitti_root",
        args.skitti_root,
        "--ford_root",
        args.ford_root,
        "--ford_train_seqs",
        args.ford_train_seqs,
        "--ford_test_seqs",
        args.ford_test_seqs,
    ]

    for depth_offset in depth_offsets:
        q_eff = float(args.q_base) * float(2 ** int(depth_offset))
        compressed_path = path_for_rate(
            args.compressed_path,
            default_output_folder("LiACM_Pro", mode, f"{split}_q{int(q_eff)}mm_compressed"),
            q_eff,
            len(depth_offsets),
        )
        decompressed_path = path_for_rate(
            args.decompressed_path,
            default_output_folder("LiACM_Pro", mode, f"{split}_q{int(q_eff)}mm_decompressed"),
            q_eff,
            len(depth_offsets),
        )
        output_csv = path_for_rate(
            args.output_csv,
            os.path.join(".", "my_test", f"LiACM_{mode}_q{int(q_eff)}mm_eval.csv"),
            q_eff,
            len(depth_offsets),
        )

        print(f"[LiACM_Pro RD] depth_offset={depth_offset} q_eff={q_eff:g}mm")

        if not args.skip_compress:
            encode_cmd = [sys.executable, os.path.join(script_dir, "encode.py")]
            encode_cmd.extend(common_dataset_args)
            encode_cmd.extend(["--output_folder", compressed_path])
            encode_cmd.extend(["--is_data_pre_quantized", str(args.is_data_pre_quantized)])
            encode_cmd.extend(["--q_base", str(args.q_base)])
            encode_cmd.extend(["--depth_offset", str(depth_offset)])
            add_arg(encode_cmd, "--input_glob", args.input_glob)
            add_arg(encode_cmd, "--ckpt", args.ckpt)
            if int(args.num_samples) >= 0:
                encode_cmd.extend(["--num_samples", str(args.num_samples)])
            run_step(encode_cmd, dry_run=args.dry_run)

        if not args.skip_decompress:
            decode_cmd = [sys.executable, os.path.join(script_dir, "decode.py")]
            decode_cmd.extend(["--dataset_mode", mode, "--dataset_split", split])
            decode_cmd.extend(["--input_glob", os.path.join(compressed_path, "*.bin")])
            decode_cmd.extend(["--output_folder", decompressed_path])
            decode_cmd.extend(["--q_base", str(args.q_base), "--depth_offset", str(depth_offset)])
            add_arg(decode_cmd, "--ckpt", args.ckpt)
            run_step(decode_cmd, dry_run=args.dry_run)

        if not args.skip_eval:
            eval_cmd = [sys.executable, os.path.join(script_dir, "evaluate.py")]
            eval_cmd.extend(["--dataset_mode", mode])
            eval_cmd.extend(["--decompressed_path", decompressed_path])
            eval_cmd.extend(["--compressed_path", compressed_path])
            eval_cmd.extend(["--q_base", str(args.q_base), "--depth_offset", str(depth_offset)])
            eval_cmd.extend(["--pcc_metric_path", args.pcc_metric_path])
            eval_cmd.extend(["--resolution", str(args.resolution)])
            eval_cmd.extend(["--input_norm", args.input_norm])
            eval_cmd.extend(["--num_workers", str(args.num_workers)])
            eval_cmd.extend(["--output_csv", output_csv])
            eval_cmd.extend(["--diagnose_scale_samples", str(args.diagnose_scale_samples)])
            add_arg(eval_cmd, "--input_glob", args.eval_input_glob)
            add_arg(eval_cmd, '--skitti_predata_root', args.skitti_predata_root)
            add_arg(eval_cmd, "--normal_path", args.normal_path)
            if mode == "FORD":
                eval_cmd.extend(["--ford_root", args.ford_root, "--ford_seqs",
                                 args.ford_train_seqs if split == 'train' else args.ford_test_seqs])
            if args.enable_cd:
                eval_cmd.append("--enable_cd")
                eval_cmd.extend(["--cd_mode", args.cd_mode])
            run_step(eval_cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

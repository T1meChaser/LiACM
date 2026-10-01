"""LiACM_Pro evaluation entry point.

English: matches decoded outputs with references, computes bitrate statistics,
and calls external metric tools such as pc_error_d/TMC13 where needed.
"""

import os
import sys

# Keep project root importable when this root entry is launched directly.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import argparse
import csv
import re
import struct
import subprocess
from glob import glob
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm


SKITTI_RESOLUTION = 59.70
FORD_RESOLUTION = 30000.0
DEFAULT_SKITTI_PREDATA_ROOT = os.environ.get("LIACM_SKITTI_PREDATA_ROOT", "")
DEFAULT_SKITTI_TEST_SEQS = tuple(f"{idx:02d}" for idx in range(11, 22))
DEFAULT_FORD_ROOT = os.environ.get("LIACM_FORD_ROOT", "")
DEFAULT_FORD_SEQS = "Ford_02_q_1mm,Ford_03_q_1mm"

PLY_STRUCT_TYPES = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}

PLY_TYPE_ALIASES = {
    "float64": "double",
    "float32": "float",
    "uint8": "uchar",
    "int8": "char",
    "uint16": "ushort",
    "int16": "short",
    "uint32": "uint",
    "int32": "int",
}


def parse_csv_arg(value):
    return [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()]


def ford_sequence_candidates(seq):
    seq = str(seq).strip()
    short_seq = seq.replace("_q_1mm", "")
    q1mm_seq = seq if seq.endswith("_q_1mm") else f"{seq}_q_1mm"
    return list(dict.fromkeys([q1mm_seq, short_seq]))


def collect_ford_files(ford_root, ford_seqs):
    files = []
    seen = set()
    for seq in parse_csv_arg(ford_seqs):
        for outer_seq in ford_sequence_candidates(seq):
            patterns = []
            for inner_seq in ford_sequence_candidates(outer_seq):
                patterns.append(os.path.join(ford_root, outer_seq, inner_seq, "*.ply"))
                patterns.append(os.path.join(ford_root, outer_seq, inner_seq, "*.PLY"))
            patterns.append(os.path.join(ford_root, outer_seq, "*.ply"))
            patterns.append(os.path.join(ford_root, outer_seq, "*.PLY"))
            seq_files = []
            seq_seen = set()
            for pattern in patterns:
                for file_path in sorted(glob(pattern)):
                    abs_path = os.path.abspath(file_path)
                    if abs_path not in seen and abs_path not in seq_seen:
                        seq_seen.add(abs_path)
                        seq_files.append(file_path)
            if seq_files:
                for file_path in seq_files:
                    seen.add(os.path.abspath(file_path))
                    files.append(file_path)
                break
    return files


def collect_input_files(input_glob):
    files = []
    seen = set()
    for pattern in parse_csv_arg(input_glob):
        for file_path in sorted(glob(pattern, recursive=True)):
            abs_path = os.path.abspath(file_path)
            if abs_path not in seen:
                seen.add(abs_path)
                files.append(file_path)
    return files


def collect_skitti_predata_files(predata_root=DEFAULT_SKITTI_PREDATA_ROOT, seq_ids=DEFAULT_SKITTI_TEST_SEQS):
    files = []
    seen = set()
    for seq_id in seq_ids:
        patterns = [
            os.path.join(predata_root, str(seq_id), "*.ply"),
            os.path.join(predata_root, str(seq_id), "*.PLY"),
        ]
        for pattern in patterns:
            for file_path in sorted(glob(pattern)):
                abs_path = os.path.abspath(file_path)
                if abs_path not in seen:
                    seen.add(abs_path)
                    files.append(file_path)
    return files


def extract_stem(file_path):
    return os.path.splitext(os.path.basename(str(file_path)))[0]


def extract_ford_sequence_id(file_path):
    parts = os.path.normpath(str(file_path)).split(os.sep)
    for part in reversed(parts[:-1]):
        if part.lower().startswith("ford_"):
            return part
    return ""


def extract_semkitti_sequence_id(file_path):
    parts = os.path.normpath(str(file_path)).split(os.sep)
    if "sequences" in parts:
        seq_idx = parts.index("sequences") + 1
        if seq_idx < len(parts):
            return str(parts[seq_idx])
    parent = os.path.basename(os.path.dirname(str(file_path)))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(str(file_path))))
    if parent.lower() == "velodyne" and grandparent:
        return str(grandparent)
    if re.fullmatch(r"\d{2}", str(parent)):
        return str(parent)
    return ""


def normalize_dataset_mode(mode):
    mode = str(mode).strip().upper()
    if mode in {"SKITTI", "FORD"}:
        return mode
    if mode == "SEMANTICKITTI":
        return "SKITTI"
    raise ValueError(f"Unsupported dataset_mode={mode}. Expected SKITTI or FORD.")


def sample_id(file_path, dataset_mode="FORD"):
    mode = normalize_dataset_mode(dataset_mode)
    seq_id = ""
    if mode == "FORD":
        seq_id = extract_ford_sequence_id(file_path)
    elif mode == "SKITTI":
        seq_id = extract_semkitti_sequence_id(file_path)
    stem = extract_stem(file_path)
    return f"{seq_id}_{stem}" if seq_id else stem


def default_input_files(args):
    mode = normalize_dataset_mode(args.dataset_mode)
    if mode == "FORD":
        return collect_ford_files(args.ford_root, args.ford_seqs)
    return collect_skitti_predata_files(args.skitti_predata_root)


def default_progressive_path(dataset_mode, split, q_base, depth_offset, suffix):
    q_eff = float(q_base) * float(2 ** max(int(depth_offset), 0))
    return os.path.join(".", "my_test", f"LiACM_Pro_{normalize_dataset_mode(dataset_mode)}_{split}_q{int(q_eff)}mm_{suffix}")


def default_normal_path(dataset_mode):
    return os.path.join(".", "my_test", f"LiACM_Pro_{normalize_dataset_mode(dataset_mode)}_normal_cache")


def default_resolution(dataset_mode):
    return FORD_RESOLUTION if normalize_dataset_mode(dataset_mode) == "FORD" else SKITTI_RESOLUTION


def default_q_base(dataset_mode):
    from utils.defaults import DATASET_Q_BASE_MM
    return DATASET_Q_BASE_MM[normalize_dataset_mode(dataset_mode)]


def ply_struct_code(prop_type):
    prop_type = str(prop_type).strip().lower()
    prop_type = PLY_TYPE_ALIASES.get(prop_type, prop_type)
    if prop_type not in PLY_STRUCT_TYPES:
        raise RuntimeError(f"Unsupported PLY property type: {prop_type}")
    return PLY_STRUCT_TYPES[prop_type]


def parse_ply_header(file_path):
    with open(file_path, "rb") as f:
        header = []
        while True:
            line_bytes = f.readline()
            if not line_bytes:
                raise RuntimeError(f"Invalid PLY header: {file_path}")
            line = line_bytes.decode("ascii", errors="ignore").strip()
            header.append(line)
            if line == "end_header":
                break
        data_offset = f.tell()

    ply_format = ""
    num_vertices = None
    vertex_props = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "format":
            ply_format = parts[1]
        elif len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                num_vertices = int(parts[2])
        elif in_vertex and len(parts) >= 3 and parts[0] == "property":
            if parts[1] == "list":
                raise RuntimeError(f"Unsupported list vertex property in {file_path}")
            vertex_props.append((parts[1], parts[2]))

    if num_vertices is None:
        raise RuntimeError(f"Missing vertex count in {file_path}")

    prop_names = [name for _, name in vertex_props]
    if not {"x", "y", "z"}.issubset(prop_names):
        raise RuntimeError(f"PLY file does not contain x/y/z: {file_path}")

    xyz_indices = [prop_names.index(axis) for axis in ("x", "y", "z")]
    normal_indices = None
    if {"nx", "ny", "nz"}.issubset(prop_names):
        normal_indices = [prop_names.index(axis) for axis in ("nx", "ny", "nz")]
    return ply_format, num_vertices, vertex_props, xyz_indices, normal_indices, data_offset


def ply_has_normals(file_path):
    try:
        _, _, _, _, normal_indices, _ = parse_ply_header(file_path)
        return normal_indices is not None
    except Exception:
        return False


def read_ply_xyz(file_path):
    ply_format, num_vertices, vertex_props, xyz_indices, _, data_offset = parse_ply_header(file_path)
    pts = np.empty((num_vertices, 3), dtype=np.float64)

    with open(file_path, "rb") as f:
        f.seek(data_offset)
        if ply_format == "ascii":
            count = 0
            for _ in range(num_vertices):
                line = f.readline()
                if not line:
                    break
                vals = line.decode("ascii", errors="ignore").strip().split()
                if len(vals) < len(vertex_props):
                    continue
                pts[count] = [float(vals[idx]) for idx in xyz_indices]
                count += 1
            return pts[:count]

        if ply_format not in {"binary_little_endian", "binary_big_endian"}:
            raise RuntimeError(f"Unsupported PLY format={ply_format}: {file_path}")

        endian = "<" if ply_format == "binary_little_endian" else ">"
        prop_formats = []
        for prop_type, _ in vertex_props:
            fmt = endian + ply_struct_code(prop_type)
            prop_formats.append((fmt, struct.calcsize(fmt)))

        for idx in range(num_vertices):
            vals = []
            for fmt, size in prop_formats:
                data = f.read(size)
                if len(data) != size:
                    return pts[:idx]
                vals.append(struct.unpack(fmt, data)[0])
            pts[idx] = [vals[prop_idx] for prop_idx in xyz_indices]
    return pts


def decoded_name_candidates(input_f, dataset_mode="FORD"):
    sid = sample_id(input_f, dataset_mode)
    stem = extract_stem(input_f)
    seq_id = extract_ford_sequence_id(input_f) if normalize_dataset_mode(dataset_mode) == "FORD" else extract_semkitti_sequence_id(input_f)
    candidates = [
        sid + ".ply",
        sid + ".bin.ply",
        stem + ".ply",
        stem + ".bin.ply",
        stem + ".ply.bin.ply",
    ]
    if seq_id:
        candidates.extend(
            [
                f"{seq_id}_{stem}.ply",
                f"{seq_id}_{stem}.bin.ply",
                f"{seq_id}_{seq_id}_{stem}.ply",
            ]
        )

    out = []
    seen = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def find_named_file(root, input_f, suffix, dataset_mode="FORD", allow_recursive=True):
    if not root:
        return None
    root = os.path.abspath(root)
    for name in decoded_name_candidates(input_f, dataset_mode):
        stem = os.path.splitext(name)[0]
        candidate = os.path.join(root, stem + suffix)
        if os.path.exists(candidate):
            return candidate
    sid = sample_id(input_f, dataset_mode)
    stem = extract_stem(input_f)
    if allow_recursive:
        patterns = [
            os.path.join(root, "**", sid + "*" + suffix),
            os.path.join(root, "**", stem + "*" + suffix),
        ]
        for pattern in patterns:
            matches = sorted(glob(pattern, recursive=True))
            if matches:
                return os.path.abspath(matches[0])
    return None


def find_decoded_file(input_f, decompressed_path, dataset_mode):
    return find_named_file(decompressed_path, input_f, ".ply", dataset_mode=dataset_mode)


def find_compressed_file(input_f, compressed_path, dataset_mode):
    return find_named_file(compressed_path, input_f, ".bin", dataset_mode=dataset_mode, allow_recursive=False)


def find_normal_file(input_f, input_norm, normal_path, dataset_mode):
    if input_norm == "none":
        return None
    if input_norm == "ref":
        return input_f
    if input_norm == "auto":
        if ply_has_normals(input_f):
            return input_f
        cached = find_named_file(normal_path, input_f, ".ply", dataset_mode=dataset_mode)
        return cached
    if input_norm == "cache":
        return find_named_file(normal_path, input_f, ".ply", dataset_mode=dataset_mode)
    raise ValueError(f"Unsupported input_norm={input_norm}")


def file_size_bits(file_path):
    if file_path is None or not os.path.exists(file_path):
        return np.nan
    return float(os.path.getsize(file_path) * 8)


def parse_metric_float(value):
    value = str(value).strip()
    if value.lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if value.lower() in {"-inf", "-infinity"}:
        return float("-inf")
    if value.lower() == "nan":
        return float("nan")
    return float(value)


def parse_psnr(output_text):
    number = r"([-+0-9.eE]+|[-+]?inf(?:inity)?|nan)"
    d1 = re.search(r"mseF,PSNR \(p2point\):\s*" + number, output_text, flags=re.IGNORECASE)
    d2 = re.search(r"mseF,PSNR \(p2plane\):\s*" + number, output_text, flags=re.IGNORECASE)
    if d1 is None:
        raise RuntimeError("Failed to parse D1/p2point PSNR from pc_error output.")
    d1_psnr = parse_metric_float(d1.group(1))
    d2_psnr = parse_metric_float(d2.group(1)) if d2 is not None else np.nan
    return d1_psnr, d2_psnr


def chamfer_distance(ref_xyz, dec_xyz, mode):
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(f"scipy is required for CD: {exc}") from exc

    if ref_xyz.shape[0] == 0 or dec_xyz.shape[0] == 0:
        return np.nan
    tree_ref = cKDTree(ref_xyz)
    tree_dec = cKDTree(dec_xyz)
    d_ref_to_dec = tree_dec.query(ref_xyz, k=1)[0]
    d_dec_to_ref = tree_ref.query(dec_xyz, k=1)[0]
    if mode == "mse":
        return float((np.mean(d_ref_to_dec ** 2) + np.mean(d_dec_to_ref ** 2)) / 2.0)
    if mode == "mean":
        return float((np.mean(d_ref_to_dec) + np.mean(d_dec_to_ref)) / 2.0)
    return float(max(np.mean(d_ref_to_dec), np.mean(d_dec_to_ref)))


def cd_scale_to_millimeters(dataset_mode):
    """Return the coordinate scale that makes CD comparable in millimeters."""
    return 1000.0 if normalize_dataset_mode(dataset_mode) == "SKITTI" else 1.0


def aggregate_dataset_bpp(arr):
    """Compute total compressed bits divided by total reference points."""
    raw_points = arr[:, 2].astype(float)
    frame_bpp = arr[:, 3].astype(float)
    valid = np.isfinite(raw_points) & (raw_points > 0) & np.isfinite(frame_bpp)
    if not np.any(valid):
        return np.nan
    return float(np.sum(raw_points[valid] * frame_bpp[valid]) / np.sum(raw_points[valid]))


def point_cloud_diag(xyz):
    if xyz.shape[0] == 0:
        return np.nan
    xyz_min = np.min(xyz, axis=0)
    xyz_max = np.max(xyz, axis=0)
    return float(np.linalg.norm(xyz_max - xyz_min))


def diagnose_scale(pairs, max_samples):
    max_samples = max(0, int(max_samples))
    if max_samples == 0:
        return
    print("Scale diagnostics:")
    for input_f, dec_f, _, _ in pairs[:max_samples]:
        try:
            ref_diag = point_cloud_diag(read_ply_xyz(input_f))
            dec_diag = point_cloud_diag(read_ply_xyz(dec_f))
            ratio = dec_diag / ref_diag if np.isfinite(ref_diag) and ref_diag > 0 else np.nan
            print(
                "  {} -> {} | ref_diag={:.3f}, dec_diag={:.3f}, ratio={:.6f}".format(
                    os.path.basename(input_f),
                    os.path.basename(dec_f),
                    ref_diag,
                    dec_diag,
                    ratio,
                )
            )
            if np.isfinite(ratio) and (ratio < 0.1 or ratio > 10.0):
                print(
                    "    WARNING: coordinate scale mismatch is likely. "
                    "Check q1mm-vs-meter data, decompression header, and --resolution."
                )
            if "q_1mm" in os.path.normpath(input_f).lower() and np.isfinite(ref_diag) and ref_diag < 1000.0:
                print("    WARNING: q1mm reference path has meter-scale coordinates.")
            if "q_1mm" in os.path.normpath(input_f).lower() and np.isfinite(dec_diag) and dec_diag < 1000.0:
                print("    WARNING: decoded output is meter-scale while q1mm reference is expected.")
        except Exception as exc:
            print(f"  Scale diagnostic skipped for {input_f}: {exc}")


def process(task):
    input_f, dec_f, normal_f, comp_f, args_dict = task
    sid = sample_id(input_f, args_dict["dataset_mode"])
    seq_id = (
        extract_ford_sequence_id(input_f)
        if args_dict["dataset_mode"] == "FORD"
        else extract_semkitti_sequence_id(input_f)
    )
    if not seq_id:
        seq_id = args_dict["dataset_mode"]
    raw_points = int(parse_ply_header(input_f)[1])
    bpp = file_size_bits(comp_f) / max(1, raw_points) if comp_f else np.nan

    cmd = [
        args_dict["pcc_metric_path"],
        f"--fileA={input_f}",
        f"--fileB={dec_f}",
        f"--resolution={args_dict['resolution']}",
    ]
    if normal_f is not None:
        cmd.append(f"--inputNorm={normal_f}")

    cmd_text = " ".join(cmd)
    d1_psnr, d2_psnr, cd = -1.0, -1.0, np.nan
    status = "ok"
    error_msg = ""
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output_text = proc.stdout or ""
        d1_psnr, d2_psnr = parse_psnr(output_text)
        if proc.returncode != 0:
            status = "warn"
            error_msg = f"pc_error returned {proc.returncode}, but PSNR was parsed. tail={output_text[-500:].strip()}"
    except Exception as exc:
        status = "error"
        output_text = locals().get("output_text", "")
        error_msg = f"{exc}; cmd={cmd_text}; tail={output_text[-500:].strip()}"

    if args_dict["enable_cd"]:
        try:
            scale_to_mm = cd_scale_to_millimeters(args_dict["dataset_mode"])
            ref_xyz_mm = read_ply_xyz(input_f) * scale_to_mm
            dec_xyz_mm = read_ply_xyz(dec_f) * scale_to_mm
            cd = chamfer_distance(ref_xyz_mm, dec_xyz_mm, args_dict["cd_mode"])
        except Exception as exc:
            if status == "ok":
                status = "warn"
            error_msg = (error_msg + " | " if error_msg else "") + f"CD error: {exc}"

    return np.array(
        [seq_id, sid, raw_points, bpp, d1_psnr, d2_psnr, cd, status, error_msg, input_f, dec_f, normal_f or "", comp_f or ""],
        dtype=object,
    )


def print_group_summary(arr, enable_cd=False, cd_mode="gaem"):
    seqs = sorted(set(arr[:, 0].astype(str).tolist()))
    for seq in seqs:
        sub = arr[arr[:, 0].astype(str) == seq]
        d1 = sub[:, 4].astype(float)
        d2 = sub[:, 5].astype(float)
        cd = sub[:, 6].astype(float)
        valid_d1 = d1[~np.isnan(d1) & (d1 >= 0)]
        valid_d2 = d2[~np.isnan(d2) & (d2 >= 0)]
        valid_cd = cd[np.isfinite(cd)]
        seq_bpp = aggregate_dataset_bpp(sub)
        bpp_text = f"{seq_bpp:.4f}" if np.isfinite(seq_bpp) else "nan"
        summary = "[{}] files={} | BPP={} | D1={} | D2={}".format(
            seq,
            len(sub),
            bpp_text,
            round(float(valid_d1.mean()), 3) if valid_d1.size else "nan",
            round(float(valid_d2.mean()), 3) if valid_d2.size else "nan",
        )
        if enable_cd:
            cd_unit = "mm^2" if cd_mode == "mse" else "mm"
            cd_value = round(float(valid_cd.mean()), 6) if valid_cd.size else "nan"
            summary += f" | CD({cd_unit})={cd_value}"
        print(summary)


def main():
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Evaluate LiACM_Pro on SemanticKITTI or Ford.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_mode", default="FORD", help="Dataset mode: SKITTI or FORD.")
    parser.add_argument(
        "--skitti_predata_root",
        default=DEFAULT_SKITTI_PREDATA_ROOT,
        help="SemanticKITTI reference PLY root. May also be set with LIACM_SKITTI_PREDATA_ROOT.",
    )
    parser.add_argument("--ford_root", default=DEFAULT_FORD_ROOT)
    parser.add_argument("--ford_seqs", default=DEFAULT_FORD_SEQS)
    parser.add_argument("--input_glob", default="", help="Optional reference PLY glob(s), comma-separated.")
    parser.add_argument("--decompressed_path", default="")
    parser.add_argument("--compressed_path", default="", help="Used only for BPP. Empty means dataset default; none disables BPP.")
    parser.add_argument("--q_base", type=float, default=None)
    parser.add_argument("--depth_offset", type=int, default=0)
    parser.add_argument("--normal_path", default="")
    parser.add_argument("--pcc_metric_path", default=os.path.join(_PROJECT_ROOT, "utils/third_party/pc_error_d"))
    parser.add_argument("--resolution", default="auto")
    parser.add_argument("--input_norm", choices=["auto", "none", "ref", "cache"], default="auto")
    parser.add_argument("--enable_cd", action="store_true")
    parser.add_argument("--cd_mode", choices=["gaem", "mean", "mse"], default="gaem")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--max_error_prints", type=int, default=10)
    parser.add_argument(
        "--diagnose_scale_samples",
        type=int,
        default=3,
        help="Print coordinate range checks for the first N matched pairs. Use 0 to disable.",
    )
    args = parser.parse_args()
    args.dataset_mode = normalize_dataset_mode(args.dataset_mode)
    if args.q_base is None:
        args.q_base = default_q_base(args.dataset_mode)
    if args.decompressed_path == "":
        args.decompressed_path = default_progressive_path(args.dataset_mode, "test", args.q_base, args.depth_offset, "decompressed")
    if args.compressed_path == "":
        args.compressed_path = default_progressive_path(args.dataset_mode, "test", args.q_base, args.depth_offset, "compressed")
    if str(args.compressed_path).strip().lower() in {"none", "0", "false"}:
        args.compressed_path = ""
    if args.normal_path == "":
        args.normal_path = default_normal_path(args.dataset_mode)
    if str(args.resolution).strip().lower() in {"auto", "dataset", ""}:
        args.resolution = default_resolution(args.dataset_mode)
    else:
        args.resolution = float(args.resolution)
    if args.output_csv == "":
        q_eff = float(args.q_base) * float(2 ** max(int(args.depth_offset), 0))
        args.output_csv = os.path.join(".", "my_test", f"LiACM_Pro_{args.dataset_mode}_q{int(q_eff)}mm_eval.csv")
    if args.input_norm == "auto" and args.dataset_mode == "FORD":
        args.input_norm = "none"

    if args.input_glob:
        input_files = collect_input_files(args.input_glob)
    else:
        input_files = default_input_files(args)
    if len(input_files) == 0:
        raise RuntimeError("No reference PLY files found. Check --dataset_mode/--input_glob.")

    pairs = []
    missing_decoded = []
    missing_normals = []
    missing_compressed = []
    for input_f in input_files:
        dec_f = find_decoded_file(input_f, args.decompressed_path, args.dataset_mode)
        if dec_f is None:
            missing_decoded.append(input_f)
            continue

        normal_f = find_normal_file(input_f, args.input_norm, args.normal_path, args.dataset_mode)
        if args.input_norm == "cache" and normal_f is None:
            missing_normals.append(input_f)
            continue

        comp_f = find_compressed_file(input_f, args.compressed_path, args.dataset_mode) if args.compressed_path else None
        if args.compressed_path and comp_f is None:
            missing_compressed.append(input_f)

        pairs.append((input_f, dec_f, normal_f, comp_f))

    print("Dataset mode:", args.dataset_mode)
    if args.dataset_mode == "FORD":
        print("Ford seqs:", args.ford_seqs)
    print("Reference files:", len(input_files))
    print("Matched decoded files:", len(pairs))
    print("Missing decoded files:", len(missing_decoded))
    print("Missing normal files:", len(missing_normals))
    print("Missing compressed files:", len(missing_compressed))
    print("Decompressed path:", os.path.abspath(args.decompressed_path))
    print("Compressed path:", os.path.abspath(args.compressed_path) if args.compressed_path else "<disabled>")
    print("Normal mode/path:", args.input_norm, os.path.abspath(args.normal_path))
    print("Resolution:", args.resolution)
    if args.enable_cd:
        cd_unit = "mm^2" if args.cd_mode == "mse" else "mm"
        print("CD output unit:", cd_unit)

    if missing_decoded[:5]:
        print("First missing decoded examples:")
        for item in missing_decoded[:5]:
            print("  ", item)
    if missing_normals[:5]:
        print("First missing normal examples:")
        for item in missing_normals[:5]:
            print("  ", item)

    if len(pairs) == 0:
        raise RuntimeError("No matched files found. Please check decompressed_path and file names.")

    diagnose_scale(pairs, args.diagnose_scale_samples)

    args_dict = {
        "pcc_metric_path": args.pcc_metric_path,
        "dataset_mode": args.dataset_mode,
        "resolution": float(args.resolution),
        "enable_cd": bool(args.enable_cd),
        "cd_mode": args.cd_mode,
    }
    tasks = [(input_f, dec_f, normal_f, comp_f, args_dict) for input_f, dec_f, normal_f, comp_f in pairs]

    workers = max(1, int(args.num_workers))
    if workers == 1:
        arr = [process(task) for task in tqdm(tasks)]
    else:
        with Pool(workers) as pool:
            arr = list(tqdm(pool.imap(process, tasks), total=len(tasks)))

    arr = np.array(arr, dtype=object)
    d1_psnrs = arr[:, 4].astype(float)
    d2_psnrs = arr[:, 5].astype(float)
    cds = arr[:, 6].astype(float)
    statuses = arr[:, 7].astype(str)
    errors = arr[:, 8].astype(str)

    valid_d1 = d1_psnrs[~np.isnan(d1_psnrs) & (d1_psnrs >= 0)]
    valid_d2 = d2_psnrs[~np.isnan(d2_psnrs) & (d2_psnrs >= 0)]
    valid_cd = cds[np.isfinite(cds)]

    print("Metric status: ok={}, warn={}, error={}".format(
        int(np.sum(statuses == "ok")),
        int(np.sum(statuses == "warn")),
        int(np.sum(statuses == "error")),
    ))
    dataset_bpp = aggregate_dataset_bpp(arr)
    print("Dataset BPP (total bits / total points):", round(dataset_bpp, 4) if np.isfinite(dataset_bpp) else "nan")
    print("Avg. D1 PSNR:", round(float(valid_d1.mean()), 3) if valid_d1.size else "nan")
    print("Avg. D2 PSNR:", round(float(valid_d2.mean()), 3) if valid_d2.size else "nan")
    if args.enable_cd:
        cd_unit = "mm^2" if args.cd_mode == "mse" else "mm"
        print(f"Avg. CD ({cd_unit}):", round(float(valid_cd.mean()), 6) if valid_cd.size else "nan")

    print_group_summary(arr, enable_cd=args.enable_cd, cd_mode=args.cd_mode)

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        cd_column = "cd_mm2" if args.cd_mode == "mse" else "cd_mm"
        header = [
            "sequence",
            "name",
            "raw_points",
            "bpp",
            "d1_psnr",
            "d2_psnr",
            cd_column,
            "status",
            "error",
            "input",
            "decoded",
            "normal",
            "compressed",
        ]
        with open(args.output_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writer.writerows(arr.tolist())
        print("Saved CSV:", os.path.abspath(args.output_csv))

    bad_indices = np.where(statuses != "ok")[0]
    if bad_indices.size > 0:
        print("First metric warnings/errors:")
        for idx in bad_indices[: args.max_error_prints]:
            print(f"  [{statuses[idx]}] {arr[idx, 1]}: {errors[idx]}")
    if np.any(statuses == 'error') or missing_compressed or missing_normals:
        raise SystemExit('Evaluation incomplete; inspect the saved CSV and missing-file counts')


if __name__ == "__main__":
    main()

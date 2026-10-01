"""Dataset and point-cloud loading utilities for LiACM.

English: this module resolves raw LiDAR/PLY inputs, applies the configured
quantization convention, and builds TorchSparse batches for training/evaluation.
"""

import os
import struct
from typing import Optional

import numpy as np
import torch
from utils.defaults import DEFAULT_Q_BASE_MM
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn

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


def _read_ply_fallback(file_path: str) -> np.ndarray:
    with open(file_path, "rb") as f:
        header = []
        while True:
            line_bytes = f.readline()
            if not line_bytes:
                raise RuntimeError(f"Invalid .ply header in {file_path}.")
            line = line_bytes.decode("ascii", errors="ignore").strip()
            header.append(line)
            if line == "end_header":
                break

        fmt = ""
        num_vertices: Optional[int] = None
        vertex_props = []
        in_vertex = False
        for line in header:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "format":
                fmt = parts[1]
            elif len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    num_vertices = int(parts[2])
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                if parts[1] == "list":
                    raise RuntimeError(f"Unsupported list property inside vertex element in {file_path}.")
                vertex_props.append((parts[1], parts[2]))

        if num_vertices is None:
            raise RuntimeError(f"Missing vertex count in .ply file: {file_path}")
        prop_names = [name for _, name in vertex_props]
        if not {"x", "y", "z"}.issubset(prop_names):
            raise RuntimeError(f"PLY file does not contain x/y/z vertex properties: {file_path}")
        xyz_indices = [prop_names.index(axis) for axis in ("x", "y", "z")]

        if fmt == "ascii":
            pts = []
            while len(pts) < num_vertices:
                line_bytes = f.readline()
                if not line_bytes:
                    break
                vals = line_bytes.decode("ascii", errors="ignore").strip().split()
                if len(vals) >= len(vertex_props):
                    pts.append([float(vals[idx]) for idx in xyz_indices])
            if len(pts) != num_vertices:
                raise RuntimeError('Truncated ASCII PLY: '+file_path)
            return np.asarray(pts, dtype=np.float32).reshape(-1, 3)

        if fmt not in {"binary_little_endian", "binary_big_endian"}:
            raise RuntimeError(f"Unsupported PLY format={fmt} in {file_path}.")

        endian = "<" if fmt == "binary_little_endian" else ">"
        try:
            record_fmt = endian + "".join(PLY_STRUCT_TYPES[prop_type] for prop_type, _ in vertex_props)
        except KeyError as exc:
            raise RuntimeError(f"Unsupported PLY property type {exc} in {file_path}.") from exc
        record_size = struct.calcsize(record_fmt)
        pts = np.empty((num_vertices, 3), dtype=np.float32)
        for idx in range(num_vertices):
            data = f.read(record_size)
            if len(data) != record_size:
                raise RuntimeError(f"Unexpected EOF while reading vertices from {file_path}.")
            vals = struct.unpack(record_fmt, data)
            pts[idx] = [vals[prop_idx] for prop_idx in xyz_indices]
        return pts


def _read_point_cloud(file_path: str) -> np.ndarray:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".bin":
        arr = np.fromfile(file_path, dtype=np.float32)
        if arr.size % 4 != 0:
            raise RuntimeError(f"Invalid .bin point cloud shape in {file_path}. Expected Nx4 float32 layout.")
        return arr.reshape(-1, 4)[:, :3]

    if ext == ".npy":
        arr = np.load(file_path)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise RuntimeError(f"Invalid .npy point cloud shape in {file_path}. Expected NxC with C>=3.")
        return arr[:, :3].astype(np.float32)

    if ext == ".ply":
        # Read directly: DataLoader workers must not spawn nested process pools.
        return _read_ply_fallback(file_path)

    raise RuntimeError(f"Unsupported point cloud format: {file_path}")


def read_point_cloud_single(file_path: str) -> np.ndarray:
    points = _read_point_cloud(file_path)
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError('Point cloud must be nonempty and finite: '+file_path)
    return points


class PCDataset:
    """Lazy-loading point-cloud dataset for SemanticKITTI and Ford."""

    def __init__(
        self,
        file_path_ls,
        q_base: float = DEFAULT_Q_BASE_MM,
        is_pre_quantized: bool = True,
        coord_shift_mm: float = 131072.0,
        cache_in_memory: bool = False,
    ):
        self.file_path_ls = [str(path) for path in file_path_ls]
        self.q_base = float(q_base)
        self.is_pre_quantized = bool(is_pre_quantized)
        self.coord_shift_mm = float(coord_shift_mm)
        self.cache_in_memory = bool(cache_in_memory)
        self.cache = {} if self.cache_in_memory else None

    def __len__(self):
        return len(self.file_path_ls)

    def _load_xyz(self, idx: int) -> np.ndarray:
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]

        xyz = read_point_cloud_single(self.file_path_ls[idx])
        if self.cache is not None:
            self.cache[idx] = xyz
        return xyz

    def __getitem__(self, idx):
        xyz = torch.tensor(self._load_xyz(idx), dtype=torch.float32)
        raw_num_points = int(xyz.shape[0])

        if self.is_pre_quantized:
            xyz_q = xyz
        else:
            xyz_q = xyz / 0.001

        coords = torch.round((xyz_q + self.coord_shift_mm) / self.q_base).to(torch.int32)
        feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        input_tensor = SparseTensor(coords=coords, feats=feats)

        return {
            "input": input_tensor,
            "raw_num_points": raw_num_points,
            "file_path": self.file_path_ls[idx],
        }


def sparse_collate_with_counts(batch):
    collated = sparse_collate_fn([{"input": item["input"]} for item in batch])
    collated["raw_num_points"] = torch.tensor(
        [int(item["raw_num_points"]) for item in batch],
        dtype=torch.int64,
    )
    collated["file_path"] = [item["file_path"] for item in batch]
    return collated


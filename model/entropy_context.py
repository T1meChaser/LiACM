"""LiACM entropy-context modeling and symbol utilities.

English: this module contains the raw_nibble symbol transform, LiDAR geometry
features, causal/window context layers, and mixture heads used
by the main network.
"""

import math

import torch
from utils.defaults import DEFAULT_Q_BASE_MM
import torch.nn as nn
import torch.nn.functional as F


LiACM_MAGIC = b"LAP1"
COORD_SHIFT_MM = 131072.0
RAW_NIBBLE_SYMBOL_ID = 0


def pack_coords_to_keys(coords: torch.Tensor) -> torch.Tensor:
    coords64 = coords.long()
    keys = coords64[:, 0]
    keys = (keys << 21) | coords64[:, 1]
    keys = (keys << 21) | coords64[:, 2]
    keys = (keys << 21) | coords64[:, 3]
    return keys


def build_coord_lookup(coords: torch.Tensor):
    keys = pack_coords_to_keys(coords)
    sorted_keys, order = torch.sort(keys)
    return sorted_keys, order


def lookup_coords(query_coords: torch.Tensor, sorted_keys: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    query_keys = pack_coords_to_keys(query_coords)
    pos = torch.searchsorted(sorted_keys, query_keys)
    found = pos < sorted_keys.shape[0]

    out = query_coords.new_full((query_coords.shape[0],), -1, dtype=torch.long)
    if not torch.any(found):
        return out

    pos_valid = pos[found]
    key_match = sorted_keys[pos_valid] == query_keys[found]
    if torch.any(key_match):
        found_idx = torch.nonzero(found, as_tuple=False).squeeze(1)[key_match]
        out[found_idx] = order[pos_valid[key_match]]
    return out


def compute_octant_index(coords: torch.Tensor) -> torch.Tensor:
    xyz = coords[:, 1:].long()
    parity = torch.bitwise_and(xyz, 1)
    return parity[:, 0] + parity[:, 1] * 2 + parity[:, 2] * 4


def compute_multiscale_octant_indices(coords: torch.Tensor, num_scales: int = 3):
    xyz = coords[:, 1:].long()
    octants = []
    for scale_idx in range(int(num_scales)):
        scaled_xyz = xyz >> int(scale_idx)
        parity = torch.bitwise_and(scaled_xyz, 1)
        octants.append((parity[:, 0] + parity[:, 1] * 2 + parity[:, 2] * 4).long())
    return octants


def compute_physical_lidar_geometry(
    coords: torch.Tensor,
    q_base: float,
    coord_shift_mm: float,
    level_from_leaf: int = 0,
    use_cell_center: bool = True,
):
    xyz = coords[:, 1:].float()
    scale = float(2 ** max(int(level_from_leaf), 0))
    center_offset = 0.5 * (scale - 1.0) if bool(use_cell_center) else 0.0
    xyz_mm = (xyz * scale + center_offset) * float(q_base) - float(coord_shift_mm)
    xyz_m = xyz_mm * 0.001

    x = xyz_m[:, 0]
    y = xyz_m[:, 1]
    z = xyz_m[:, 2]
    xy_sq = x.pow(2) + y.pow(2)
    rho = torch.sqrt(xy_sq + 1e-8)
    range_m = torch.sqrt(xy_sq + z.pow(2) + 1e-8)
    azimuth = torch.atan2(y, x)
    elevation = torch.atan2(z, rho)
    return xyz_m, range_m, azimuth, elevation


def split_occupancy_nibbles(raw_occ: torch.Tensor):
    occ = raw_occ.view(-1).long().clamp(0, 255)
    return torch.bitwise_and(occ, 15).long(), torch.bitwise_right_shift(occ, 4).long()


def combine_occupancy_nibbles(s0: torch.Tensor, s1: torch.Tensor):
    lo = s0.view(-1).long().clamp(0, 15)
    hi = s1.view(-1).long().clamp(0, 15)
    return torch.bitwise_or(lo, torch.bitwise_left_shift(hi, 4)).long()


def extract_lidar_features_from_geometry(
    range_m: torch.Tensor,
    azimuth: torch.Tensor,
    elevation: torch.Tensor,
    max_range_m: float = 120.0,
) -> torch.Tensor:
    log_range = torch.log1p(range_m.clamp(max=float(max_range_m))) / math.log1p(float(max_range_m))
    return torch.stack(
        (
            log_range,
            torch.sin(azimuth),
            torch.cos(azimuth),
            torch.sin(elevation),
            torch.cos(elevation),
        ),
        dim=1,
    )


def compute_range_bins_from_range(
    range_m: torch.Tensor,
    num_bins: int = 16,
    max_range_m: float = 120.0,
) -> torch.Tensor:
    norm = torch.log1p(range_m.clamp(max=float(max_range_m))) / math.log1p(float(max_range_m))
    norm = norm.clamp(min=0.0, max=1.0 - 1e-6)
    return torch.floor(norm * int(num_bins)).long()


def compute_calibrated_beam_bins_from_elevation(
    elevation: torch.Tensor,
    num_bins: int = 64,
    min_deg: float = -25.0,
    max_deg: float = 5.0,
):
    elevation_deg = elevation * (180.0 / math.pi)
    num_bins = max(int(num_bins), 1)

    if num_bins == 64:
        upper_start = torch.tensor(2.0, device=elevation.device, dtype=elevation.dtype)
        upper_end = torch.tensor(-8.333333, device=elevation.device, dtype=elevation.dtype)
        lower_start = torch.tensor(-8.833333, device=elevation.device, dtype=elevation.dtype)
        lower_end = torch.tensor(-24.333333, device=elevation.device, dtype=elevation.dtype)

        upper_step = (upper_start - upper_end) / 31.0
        lower_step = (lower_start - lower_end) / 31.0

        upper_idx = torch.round((upper_start - elevation_deg) / upper_step).clamp(0, 31).long()
        lower_idx = torch.round((lower_start - elevation_deg) / lower_step).clamp(0, 31).long()

        upper_center = upper_start - upper_idx.to(elevation.dtype) * upper_step
        lower_center = lower_start - lower_idx.to(elevation.dtype) * lower_step
        use_upper = torch.abs(elevation_deg - upper_center) <= torch.abs(elevation_deg - lower_center)

        beam_ids = torch.where(use_upper, upper_idx, lower_idx + 32)
        center_deg = torch.where(use_upper, upper_center, lower_center)
        residual_scale = max(float(max_deg - min_deg) / 63.0, 1e-3)
    else:
        if num_bins == 1:
            beam_ids = torch.zeros_like(elevation_deg, dtype=torch.long)
            center_deg = torch.full_like(elevation_deg, float(max_deg + min_deg) * 0.5)
            residual_scale = max(float(max_deg - min_deg), 1e-3)
        else:
            step = float(max_deg - min_deg) / float(num_bins - 1)
            beam_ids = torch.round((float(max_deg) - elevation_deg) / step).clamp(0, num_bins - 1).long()
            center_deg = float(max_deg) - beam_ids.to(elevation.dtype) * step
            residual_scale = max(step, 1e-3)

    beam_residual = ((elevation_deg - center_deg) / residual_scale).clamp(min=-4.0, max=4.0)
    return beam_ids, beam_residual.unsqueeze(1)


def morton3d_encode(xyz: torch.Tensor, max_bits: int = 21) -> torch.Tensor:
    if not 0 <= max_bits <= 21:
        raise ValueError('Morton coordinates support at most 21 bits per axis')
    value = xyz.long() & ((1 << max_bits) - 1)
    for shift, mask in ((32, 0x1F00000000FFFF), (16, 0x1F0000FF0000FF),
                        (8, 0x100F00F00F00F00F), (4, 0x10C30C30C30C30C3),
                        (2, 0x1249249249249249)):
        value = (value | (value << shift)) & mask
    return value[:, 0] | (value[:, 1] << 1) | (value[:, 2] << 2)


def morton_sort_order(coords: torch.Tensor) -> torch.Tensor:
    if coords.shape[0] == 0:
        return coords.new_zeros((0,), dtype=torch.long)

    batch_ids = coords[:, 0].long()
    unique_batches = torch.unique(batch_ids, sorted=True)
    order_parts = []
    for batch_id in unique_batches.tolist():
        idx = torch.nonzero(batch_ids == int(batch_id), as_tuple=False).squeeze(1)
        morton = morton3d_encode(coords[idx, 1:])
        _, local_order = torch.sort(morton)
        order_parts.append(idx[local_order])
    return torch.cat(order_parts, dim=0)


def sort_by_morton(coords: torch.Tensor, *tensors):
    order = morton_sort_order(coords)
    outputs = [coords[order]]
    for tensor in tensors:
        outputs.append(tensor[order] if tensor is not None else None)
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def build_window_partition(coords: torch.Tensor, window_size: int):
    if coords.shape[0] == 0:
        empty = coords.new_zeros((0,), dtype=torch.long)
        return empty, empty, empty

    batch_ids = coords[:, 0].long()
    unique_batches = torch.unique(batch_ids, sorted=True)
    window_ids = coords.new_zeros((coords.shape[0],), dtype=torch.long)
    window_pos = coords.new_zeros((coords.shape[0],), dtype=torch.long)
    order = torch.arange(coords.shape[0], device=coords.device, dtype=torch.long)

    window_offset = 0
    for batch_id in unique_batches.tolist():
        idx = torch.nonzero(batch_ids == int(batch_id), as_tuple=False).squeeze(1)
        local_pos = torch.arange(idx.shape[0], device=coords.device, dtype=torch.long)
        local_window = torch.div(local_pos, int(window_size), rounding_mode="floor")
        window_ids[idx] = local_window + int(window_offset)
        window_pos[idx] = torch.remainder(local_pos, int(window_size))
        if idx.shape[0] > 0:
            window_offset += int(local_window[-1].item()) + 1

    return window_ids, window_pos, order


def scatter_mean_by_window(values: torch.Tensor, window_ids: torch.Tensor, num_windows: int):
    out = values.new_zeros((int(num_windows), values.shape[1]))
    counts = values.new_zeros((int(num_windows), 1))
    if values.shape[0] == 0 or num_windows == 0:
        return out

    out.index_add_(0, window_ids.long(), values)
    ones = values.new_ones((values.shape[0], 1))
    counts.index_add_(0, window_ids.long(), ones)
    return out / counts.clamp(min=1.0)


def infer_num_depths(num_streams: int, streams_per_depth: int = 2) -> int:
    streams_per_depth = max(int(streams_per_depth), 1)
    if num_streams % streams_per_depth != 0:
        raise RuntimeError(
            f"Invalid LiACM stream count: {num_streams}. "
            f"Expected a multiple of streams_per_depth={streams_per_depth}."
        )
    return num_streams // streams_per_depth


class GeoStem(nn.Module):
    def __init__(
        self,
        channels: int,
        max_depth: int = 64,
        range_bins: int = 16,
        beam_bins: int = 64,
        max_range_m: float = 120.0,
        beam_min_deg: float = -25.0,
        beam_max_deg: float = 5.0,
        q_base: float = DEFAULT_Q_BASE_MM,
        coord_shift_mm: float = COORD_SHIFT_MM,
        use_scale_context: int = 1,
        use_lidar_prior: int = 1,
        use_octant_prior: int = 1,
    ):
        super().__init__()
        self.channels = int(channels)
        self.range_bins = int(range_bins)
        self.beam_bins = int(beam_bins)
        self.max_range_m = float(max_range_m)
        self.beam_min_deg = float(beam_min_deg)
        self.beam_max_deg = float(beam_max_deg)
        self.q_base = float(q_base)
        self.coord_shift_mm = float(coord_shift_mm)
        self.use_scale_context = bool(int(use_scale_context))
        self.use_lidar_prior = bool(int(use_lidar_prior))
        self.use_octant_prior = bool(int(use_octant_prior))

        self.scale_emb = nn.Embedding(int(max_depth), self.channels)
        self.cell_size_proj = nn.Sequential(
            nn.Linear(1, self.channels),
            nn.ReLU(True),
            nn.Linear(self.channels, self.channels),
        )
        self.octant_emb_0 = nn.Embedding(8, self.channels)
        self.octant_emb_1 = nn.Embedding(8, self.channels)
        self.octant_emb_2 = nn.Embedding(8, self.channels)

        self.octree_proj = nn.Sequential(
            nn.Linear(self.channels * 5, self.channels),
            nn.ReLU(True),
            nn.Linear(self.channels, self.channels),
        )

        self.lidar_cont_proj = nn.Sequential(
            nn.Linear(5, self.channels),
            nn.ReLU(True),
            nn.Linear(self.channels, self.channels),
        )
        self.range_bin_emb = nn.Embedding(self.range_bins, self.channels)
        self.beam_bin_emb = nn.Embedding(self.beam_bins, self.channels)
        self.beam_residual_proj = nn.Sequential(
            nn.Linear(1, self.channels),
            nn.ReLU(True),
            nn.Linear(self.channels, self.channels),
        )
        self.lidar_proj = nn.Sequential(
            nn.Linear(self.channels * 4, self.channels),
            nn.ReLU(True),
            nn.Linear(self.channels, self.channels),
        )

        self.lidar_delta = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.ReLU(True),
            nn.Linear(self.channels, self.channels),
        )
        self.lidar_gate = nn.Sequential(
            nn.Linear(self.channels * 2, self.channels),
            nn.ReLU(True),
            nn.Linear(self.channels, self.channels),
        )
        if isinstance(self.lidar_gate[-1], nn.Linear) and self.lidar_gate[-1].bias is not None:
            nn.init.constant_(self.lidar_gate[-1].bias, -1.0)

    def set_runtime_geometry(self, q_base: float = None, coord_shift_mm: float = None):
        if q_base is not None:
            self.q_base = float(q_base)
        if coord_shift_mm is not None:
            self.coord_shift_mm = float(coord_shift_mm)

    def _scale_ids(self, coords: torch.Tensor, level_from_leaf: int) -> torch.Tensor:
        return torch.full(
            (coords.shape[0],),
            min(max(int(level_from_leaf) - 1, 0), self.scale_emb.num_embeddings - 1),
            device=coords.device,
            dtype=torch.long,
        )

    def forward(self, coords: torch.Tensor, depth: int, full_num_depths: int = None) -> torch.Tensor:
        if full_num_depths is None:
            raise ValueError('Physical geometry requires the full hierarchy depth')
        else:
            # Stored occupied parents are one level above the leaf lattice.
            level_from_leaf = max(int(full_num_depths) + 1 - int(depth), 0)
        if self.use_scale_context:
            scale_ctx = self.scale_emb(self._scale_ids(coords, level_from_leaf))
            log_cell = math.log2(max(float(self.q_base) * float(2 ** level_from_leaf), 1e-6) / max(float(self.q_base), 1e-6))
            cell_ctx = self.cell_size_proj(
                coords.new_full((coords.shape[0], 1), float(log_cell), dtype=torch.float32)
            )
        else:
            scale_ctx = coords.new_zeros((coords.shape[0], self.channels), dtype=torch.float32)
            cell_ctx = coords.new_zeros((coords.shape[0], self.channels), dtype=torch.float32)
        octant_0, octant_1, octant_2 = compute_multiscale_octant_indices(coords, num_scales=3)
        if self.use_octant_prior:
            octant_ctx_0 = self.octant_emb_0(octant_0)
            octant_ctx_1 = self.octant_emb_1(octant_1)
            octant_ctx_2 = self.octant_emb_2(octant_2)
        else:
            octant_ctx_0 = coords.new_zeros((coords.shape[0], self.channels), dtype=torch.float32)
            octant_ctx_1 = coords.new_zeros((coords.shape[0], self.channels), dtype=torch.float32)
            octant_ctx_2 = coords.new_zeros((coords.shape[0], self.channels), dtype=torch.float32)
        octree_geo = self.octree_proj(
            torch.cat(
                (
                    scale_ctx,
                    cell_ctx,
                    octant_ctx_0,
                    octant_ctx_1,
                    octant_ctx_2,
                ),
                dim=-1,
            )
        )
        if not self.use_lidar_prior:
            return octree_geo

        _, range_m, azimuth, elevation = compute_physical_lidar_geometry(
            coords,
            q_base=self.q_base,
            coord_shift_mm=self.coord_shift_mm,
            level_from_leaf=level_from_leaf,
            use_cell_center=True,
        )
        lidar_cont = self.lidar_cont_proj(
            extract_lidar_features_from_geometry(
                range_m,
                azimuth,
                elevation,
                max_range_m=self.max_range_m,
            )
        )
        range_bin_ids = compute_range_bins_from_range(
            range_m,
            num_bins=self.range_bins,
            max_range_m=self.max_range_m,
        )
        beam_bin_ids, beam_residual = compute_calibrated_beam_bins_from_elevation(
            elevation,
            num_bins=self.beam_bins,
            min_deg=self.beam_min_deg,
            max_deg=self.beam_max_deg,
        )
        beam_residual_ctx = self.beam_residual_proj(beam_residual)
        lidar_geo = self.lidar_proj(
            torch.cat(
                (
                    lidar_cont,
                    self.range_bin_emb(range_bin_ids),
                    self.beam_bin_emb(beam_bin_ids),
                    beam_residual_ctx,
                ),
                dim=-1,
            )
        )

        lidar_gate = torch.sigmoid(self.lidar_gate(torch.cat((octree_geo, lidar_geo), dim=-1)))
        return octree_geo + lidar_gate * self.lidar_delta(lidar_geo)


class GeoFeatureModulation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Linear(channels, channels)
        self.beta = nn.Linear(channels, channels)

    def forward(self, feats: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma = torch.tanh(self.gamma(context))
        beta = self.beta(context)
        return feats * (1.0 + 0.5 * gamma) + beta


class GeoConditionedMoEHead(nn.Module):
    def __init__(self, channels: int, gate_channels: int, num_experts: int = 3, num_classes: int = 16):
        super().__init__()
        self.num_experts = int(num_experts)
        gate_hidden = max(32, int(gate_channels))
        hidden = max(32, int(channels))
        self.gating_net = nn.Sequential(
            nn.Linear(gate_channels, gate_hidden),
            nn.ReLU(True),
            nn.Linear(gate_hidden, self.num_experts),
            nn.Softmax(dim=-1),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(channels, hidden),
                    nn.ReLU(True),
                    nn.Linear(hidden, num_classes),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward_logits(self, feats: torch.Tensor, gate_ctx: torch.Tensor) -> torch.Tensor:
        gating = self.gating_net(gate_ctx).unsqueeze(-1)
        logits = torch.stack([expert(feats) for expert in self.experts], dim=1)
        return torch.sum(gating * logits, dim=1)

    def forward(self, feats: torch.Tensor, gate_ctx: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward_logits(feats, gate_ctx), dim=-1)


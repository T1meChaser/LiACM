"""Main LiACM network.

English: builds the multi-depth hierarchy, aligns decoded context/features with
raw_nibble symbols, and produces entropy logits for training and codec replay.
"""

import math

import torch
import torch.nn as nn
from utils.defaults import DEFAULT_Q_BASE_MM

from torchsparse import SparseTensor
from torchsparse import nn as spnn

from model.sparse_blocks import FCG, FOG, ResNet, TargetEmbedding, LightweightMultiScaleFusion2

from model.entropy_context import (
    COORD_SHIFT_MM,
    GeoConditionedMoEHead,
    GeoFeatureModulation,
    GeoStem,
    build_coord_lookup,
    build_window_partition,
    compute_octant_index,
    compute_physical_lidar_geometry,
    lookup_coords,
    RAW_NIBBLE_SYMBOL_ID,
    scatter_mean_by_window,
    combine_occupancy_nibbles,
    split_occupancy_nibbles,
    sort_by_morton,
)


class ContextStack(nn.Module):
    """Sparse context stem followed by residual/multi-scale refinement pairs."""

    def __init__(self, channels: int, kernel_size: int, context_layers: int = 3):
        super().__init__()
        layers = [
            spnn.Conv3d(channels, channels, kernel_size),
            spnn.ReLU(True),
        ]
        for _ in range(int(context_layers)):
            layers.append(ResNet(channels))
            layers.append(LightweightMultiScaleFusion2(channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.net(x)


class Network(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        kernel_size: int = 3,
        context_layers: int = 3,
        prior_context_layers: int = 1,
        use_cross_layer_reuse: int = 1,
        carry_feature_limit: float = 0.,
        ancestor_depth: int = 5,
        num_experts: int = 4,
        window_size: int = 8,
        base_cut: int = 64,
        q_base: float = DEFAULT_Q_BASE_MM,
        depth_offsets=None,
        prefix_weights=None,
        coord_shift_mm: float = COORD_SHIFT_MM,
        range_bins: int = 16,
        beam_bins: int = 64,
        max_range_m: float = 120.0,
        beam_min_deg: float = -25.0,
        beam_max_deg: float = 5.0,
        use_prefix_loss: int = 1,
        use_scale_context: int = 1,
        use_lidar_prior: int = 1,
        use_octant_prior: int = 1,
        use_geo_modulation: int = 1,
        use_ancestor_context: int = 1,
        use_stage_memory: int = 1,
        num_causal_stages: int = 8,
        stage_schedule: str = "modulo",
        debug_coord_check: int = 0,
        use_macro_context: int = 1,
        macro_window_size: int = 512,
    ):
        super().__init__()

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.context_layers = int(context_layers)
        self.prior_context_layers = int(prior_context_layers)
        self.use_cross_layer_reuse = bool(int(use_cross_layer_reuse))
        self.carry_feature_limit = float(carry_feature_limit)
        if not math.isfinite(self.carry_feature_limit) or self.carry_feature_limit < 0:
            raise ValueError('carry_feature_limit must be finite and nonnegative')
        self.rate_loss = 'floored'
        if not 0 <= self.prior_context_layers <= self.context_layers:
            raise ValueError('prior_context_layers must be between 0 and context_layers')
        self.ancestor_depth = max(int(ancestor_depth), 0)
        self.num_experts = max(int(num_experts), 1)
        self.window_size = max(int(window_size), 2)
        self.base_cut = max(int(base_cut), 2)
        self.q_base = float(q_base)
        if not math.isfinite(self.q_base) or self.q_base <= 0:
            raise ValueError('q_base must be positive and finite')
        self.depth_offsets = tuple(int(item) for item in (depth_offsets if depth_offsets is not None else [0, 1, 2, 3, 4, 5]))
        if not self.depth_offsets or min(self.depth_offsets) < 0 or len(set(self.depth_offsets)) != len(self.depth_offsets):
            raise ValueError('depth_offsets must be distinct nonnegative integers')
        raw_prefix_weights = tuple(float(item) for item in (prefix_weights if prefix_weights is not None else [1, 0, 0, 0, 0, 0]))
        if (not raw_prefix_weights or len(raw_prefix_weights) != len(self.depth_offsets)
                or any(not math.isfinite(v) or v < 0 for v in raw_prefix_weights)
                or sum(raw_prefix_weights) <= 0):
            raise ValueError('prefix_weights must be finite, nonnegative and match depth_offsets')
        weight_sum = sum(raw_prefix_weights) if len(raw_prefix_weights) else 1.0
        self.prefix_weights = tuple(item / max(weight_sum, 1e-12) for item in raw_prefix_weights)
        self.coord_shift_mm = float(coord_shift_mm)
        self.range_bins = max(int(range_bins), 1)
        self.beam_bins = max(int(beam_bins), 1)
        self.max_range_m = float(max_range_m)
        self.beam_min_deg = float(beam_min_deg)
        self.beam_max_deg = float(beam_max_deg)
        self.use_prefix_loss = bool(int(use_prefix_loss))
        self.use_scale_context = bool(int(use_scale_context))
        self.use_lidar_prior = bool(int(use_lidar_prior))
        self.use_octant_prior = bool(int(use_octant_prior))
        self.use_geo_modulation = bool(int(use_geo_modulation))
        self.use_ancestor_context = bool(int(use_ancestor_context))
        self.use_stage_memory = bool(int(use_stage_memory))
        self.num_causal_stages = max(int(num_causal_stages), 1)
        self.stage_schedule = str(stage_schedule).strip().lower().replace("-", "_")
        if self.stage_schedule not in ("topology", "modulo", "morton"):
            raise ValueError(f"Unsupported stage_schedule={stage_schedule}.")
        self.symbol_id = int(RAW_NIBBLE_SYMBOL_ID)
        self.symbol_name = "raw_nibble"
        self.streams_per_stage = 2
        self.debug_coord_check = bool(int(debug_coord_check))
        self.use_macro_context = bool(int(use_macro_context))
        self.macro_window_size = max(int(macro_window_size), self.window_size)

        self.geo_stem = GeoStem(
            self.channels,
            range_bins=self.range_bins,
            beam_bins=self.beam_bins,
            max_range_m=self.max_range_m,
            beam_min_deg=self.beam_min_deg,
            beam_max_deg=self.beam_max_deg,
            q_base=self.q_base,
            coord_shift_mm=self.coord_shift_mm,
            use_scale_context=self.use_scale_context,
            use_lidar_prior=self.use_lidar_prior,
            use_octant_prior=self.use_octant_prior,
        )
        self.prior_geo_mod = GeoFeatureModulation(self.channels)
        self.target_geo_mod = GeoFeatureModulation(self.channels)
        self.context_geo_mod = GeoFeatureModulation(self.channels)

        self.prior_embedding = nn.Embedding(256, self.channels)
        self.nibble0_emb = nn.Embedding(16, self.channels)
        self.nibble1_emb = nn.Embedding(16, self.channels)
        self.ancestor_step_emb = nn.Embedding(self.ancestor_depth + 1, self.channels)
        self.rel_octant_emb = nn.Embedding(8, self.channels)
        self.prior_resnet = ContextStack(
            self.channels,
            self.kernel_size,
            context_layers=self.context_layers,
        )
        self.target_embedding = TargetEmbedding(self.channels)
        self.target_resnet = ContextStack(
            self.channels,
            self.kernel_size,
            context_layers=self.context_layers,
        )

        anc_in = self.channels * max(1, self.ancestor_depth)
        self.ancestor_proj = nn.Sequential(
            nn.Linear(anc_in, self.channels),
            nn.ReLU(True),
            nn.Linear(self.channels, self.channels),
        )
        self.gate_context_fuse = nn.Sequential(
            nn.Linear(self.channels * 2, self.channels),
            nn.ReLU(True),
            nn.Linear(self.channels, self.channels),
        )
        self.macro_context_fuse = nn.Sequential(
            nn.Linear(self.channels * 2, self.channels),
            nn.ReLU(True),
            nn.Linear(self.channels, self.channels),
        )

        self.moe_head_s0 = GeoConditionedMoEHead(
            channels=self.channels, gate_channels=self.channels,
            num_experts=self.num_experts, num_classes=16,
        )
        self.moe_head_s1 = GeoConditionedMoEHead(
            channels=self.channels, gate_channels=self.channels,
            num_experts=self.num_experts, num_classes=16,
        )
        self.window_pos_emb = nn.Embedding(self.window_size, self.channels)
        self.stage_id_emb = nn.Embedding(self.num_causal_stages, self.channels)
        self.stage_mem_token = nn.Sequential(
            nn.Linear(self.channels * 4, self.channels),
            nn.ReLU(True),
            nn.Linear(self.channels, self.channels),
        )
        self.stage_feat_fuse = nn.Sequential(
            nn.Linear(self.channels * 2, self.channels),
            nn.ReLU(True),
            nn.Linear(self.channels, self.channels),
        )
        self.stage_gate_fuse = nn.Sequential(
            nn.Linear(self.channels * 2, self.channels),
            nn.ReLU(True),
            nn.Linear(self.channels, self.channels),
        )

        self.fog = FOG()
        self.fcg = FCG()
        # Preserve the validated initialization order, then discard unused pairs.
        self.prior_resnet.net = nn.Sequential(
            *list(self.prior_resnet.net.children())[:2 + 2*self.prior_context_layers])
        if self.use_cross_layer_reuse:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(110915)
                self.carry_projection = nn.Linear(self.channels, self.channels, bias=False)
            nn.init.eye_(self.carry_projection.weight)
            self.carry_logit = nn.Parameter(torch.full((self.channels,), math.log(.1/.9)))
        self.reset_frame_state()

    def reset_frame_state(self):
        self._previous_target = None
        self._previous_coords = None
        self._previous_depth = None

    def _prior_input(self, coords, occupancy, geometry, depth):
        features = self.prior_embedding(occupancy.int().clamp(0, 255)).view(-1, self.channels) + geometry
        if self.use_cross_layer_reuse and int(depth) > 0:
            if self._previous_depth != int(depth) or self._previous_target is None:
                raise RuntimeError('Missing immediately preceding layer state')
            if self._previous_target.shape[0] != coords.shape[0]:
                raise RuntimeError('Cross-layer state cardinality mismatch')
            if self.debug_coord_check and not torch.equal(self._previous_coords, coords):
                raise RuntimeError('Cross-layer coordinate order mismatch')
            features = features + torch.sigmoid(self.carry_logit) * self.carry_projection(self._previous_target)
        return features

    def _commit_target(self, target, depth):
        if self.use_cross_layer_reuse:
            state = target.feats
            if self.carry_feature_limit > 0:
                # Preserve each feature direction; ordinary states are unchanged.
                scale = (state.abs().amax(dim=-1, keepdim=True) / self.carry_feature_limit).clamp(min=1.)
                state = state / scale
            self._previous_target = state
            self._previous_coords = target.coords
            self._previous_depth = int(depth) + 1

    def select_stage(self, stage, index):
        return stage['stage_ids'] == int(index)

    def stage_empty(self, stage, index, selection):
        return not bool(torch.any(selection))

    @property
    def stage_schedule_id(self) -> int:
        if self.stage_schedule == "topology":
            return 1
        if self.stage_schedule == "morton":
            return 2
        return 0

    def set_runtime_geometry(self, q_base: float = None, coord_shift_mm: float = None):
        if q_base is not None:
            self.q_base = float(q_base)
        if coord_shift_mm is not None:
            self.coord_shift_mm = float(coord_shift_mm)
        self.geo_stem.set_runtime_geometry(q_base=self.q_base, coord_shift_mm=self.coord_shift_mm)

    def _resolve_num_points(self, raw_num_points, fallback_num_points: int) -> int:
        if raw_num_points is None:
            return max(int(fallback_num_points), 1)
        if isinstance(raw_num_points, torch.Tensor):
            return max(int(raw_num_points.sum().item()), 1)
        if isinstance(raw_num_points, (list, tuple)):
            return max(int(sum(raw_num_points)), 1)
        return max(int(raw_num_points), 1)

    def _make_cache_entry(self, coords: torch.Tensor, feat: torch.Tensor, geo: torch.Tensor, depth: int):
        sorted_keys, order = build_coord_lookup(coords)
        return {
            "coords": coords,
            "feat": feat,
            "geo": geo,
            "depth": int(depth),
            "lookup_keys": sorted_keys,
            "lookup_order": order,
        }

    def _gather_ancestor_indices(self, target_coords: torch.Tensor, current_depth: int, ancestor_cache):
        anc_idx = target_coords.new_full((target_coords.shape[0], self.ancestor_depth), -1, dtype=torch.long)
        rel_oct = target_coords.new_zeros((target_coords.shape[0], self.ancestor_depth), dtype=torch.long)
        if self.ancestor_depth == 0:
            return anc_idx, rel_oct

        for step in range(1, self.ancestor_depth + 1):
            layer_idx = current_depth + 1 - step
            if layer_idx < 0:
                continue
            anc_coords = target_coords.clone()
            anc_coords[:, 1:] = torch.div(target_coords[:, 1:], 2 ** step, rounding_mode="floor")
            anc_idx[:, step - 1] = lookup_coords(
                anc_coords,
                ancestor_cache[layer_idx]["lookup_keys"],
                ancestor_cache[layer_idx]["lookup_order"],
            )
            rel_coords = torch.div(target_coords[:, 1:], 2 ** (step - 1), rounding_mode="floor")
            rel_oct[:, step - 1] = compute_octant_index(
                torch.cat((target_coords[:, :1], rel_coords), dim=-1)
            ).long()
        return anc_idx, rel_oct

    def _encode_ancestor_context(self, target_coords: torch.Tensor, current_depth: int, ancestor_cache):
        if (not self.use_ancestor_context) or self.ancestor_depth == 0:
            return target_coords.new_zeros((target_coords.shape[0], self.channels), dtype=torch.float32)

        anc_idx, rel_oct = self._gather_ancestor_indices(target_coords, current_depth, ancestor_cache)
        tokens = []
        for step in range(self.ancestor_depth):
            token = target_coords.new_zeros((target_coords.shape[0], self.channels), dtype=torch.float32)
            layer_idx = current_depth - step
            if layer_idx < 0:
                tokens.append(token)
                continue
            valid = anc_idx[:, step] >= 0
            if torch.any(valid):
                cache = ancestor_cache[layer_idx]
                idx_valid = anc_idx[valid, step]
                step_ids = torch.full(
                    (idx_valid.shape[0],),
                    step + 1,
                    device=target_coords.device,
                    dtype=torch.long,
                )
                token[valid] = (
                    cache["feat"][idx_valid]
                    + cache["geo"][idx_valid]
                    + self.ancestor_step_emb(step_ids)
                    + self.rel_octant_emb(rel_oct[valid, step])
                )
            tokens.append(token)
        return self.ancestor_proj(torch.cat(tokens, dim=-1))

    def build_hierarchy(self, x: SparseTensor, base_cut: int = None):
        active_base_cut = self.base_cut if base_cut is None else max(int(base_cut), 2)
        data_ls = []
        while True:
            x = self.fog(x)
            x_c, x_f = sort_by_morton(x.coords, x.feats)
            x = SparseTensor(coords=x_c, feats=x_f)
            data_ls.append((x.coords.clone(), x.feats.clone()))
            if x.coords.shape[0] < active_base_cut:
                break
        return data_ls[::-1]

    def assert_aligned_coords(
        self,
        predicted_coords: torch.Tensor,
        target_coords: torch.Tensor,
        depth: int,
        context: str = "train",
    ):
        if not self.debug_coord_check:
            return
        if predicted_coords.shape != target_coords.shape:
            raise RuntimeError(
                f"LiACM coordinate alignment failed in {context} at depth={depth}: "
                f"shape mismatch predicted={tuple(predicted_coords.shape)} target={tuple(target_coords.shape)}."
            )
        if torch.equal(predicted_coords, target_coords):
            return

        diff_mask = torch.any(predicted_coords.long() != target_coords.long(), dim=1)
        mismatch_count = int(diff_mask.sum().item())
        first_idx = int(torch.nonzero(diff_mask, as_tuple=False).view(-1)[0].item()) if mismatch_count > 0 else -1
        pred_item = predicted_coords[first_idx].detach().cpu().tolist() if first_idx >= 0 else []
        target_item = target_coords[first_idx].detach().cpu().tolist() if first_idx >= 0 else []
        raise RuntimeError(
            f"LiACM coordinate alignment failed in {context} at depth={depth}: "
            f"mismatch_count={mismatch_count}, first_pred={pred_item}, first_target={target_item}."
        )

    def prior_branch(self, x_c: torch.Tensor, x_o: torch.Tensor, depth: int, full_num_depths: int = None):
        if int(depth) == 0:
            self.reset_frame_state()
        geo_ctx = self.geo_stem(x_c, depth, full_num_depths=full_num_depths)
        x_f = self._prior_input(x_c, x_o, geo_ctx, depth)
        x = SparseTensor(coords=x_c, feats=x_f)
        x = self.prior_resnet(x)
        if self.use_geo_modulation:
            x.feats = self.prior_geo_mod(x.feats, geo_ctx)
        return x, geo_ctx

    def target_branch(self, x_c: torch.Tensor, x_o: torch.Tensor, prior_feats: torch.Tensor, depth: int, full_num_depths: int = None):
        x_up_c, x_up_f = self.fcg(x_c, x_o, prior_feats)
        x_up_c, x_up_f = sort_by_morton(x_up_c, x_up_f)

        target_geo = self.geo_stem(x_up_c, depth + 1, full_num_depths=full_num_depths)
        x_up_f = self.target_embedding(x_up_f, x_up_c) + target_geo
        x_up = SparseTensor(coords=x_up_c, feats=x_up_f)
        x_up = self.target_resnet(x_up)
        if self.use_geo_modulation:
            x_up.feats = self.target_geo_mod(x_up.feats, target_geo)
        self._commit_target(x_up, depth)
        return x_up, target_geo

    def build_geo_context(self, x_up: SparseTensor, target_geo: torch.Tensor, current_depth: int, ancestor_cache):
        ancestor_ctx = self._encode_ancestor_context(x_up.coords, current_depth, ancestor_cache)
        gate_ctx = self.gate_context_fuse(torch.cat((target_geo, ancestor_ctx), dim=-1))
        if self.use_macro_context and x_up.coords.shape[0] > 0:
            macro_ids, _, _ = build_window_partition(x_up.coords, self.macro_window_size)
            num_macro = int(macro_ids.max().item()) + 1 if macro_ids.numel() > 0 else 0
            macro_summary = scatter_mean_by_window(gate_ctx, macro_ids.long(), num_macro)
            gate_ctx = self.macro_context_fuse(torch.cat((gate_ctx, macro_summary[macro_ids.long()]), dim=-1))
        mod_feats = self.context_geo_mod(x_up.feats, gate_ctx) if self.use_geo_modulation else x_up.feats
        return mod_feats, ancestor_ctx, gate_ctx

    def build_stage_partition(self, target_coords: torch.Tensor, depth: int = 0, full_num_depths: int = 0):
        window_ids, window_pos, order = build_window_partition(target_coords, self.window_size)
        if self.stage_schedule == "modulo":
            stage_ids = torch.remainder(window_pos, self.num_causal_stages).long()
        elif self.stage_schedule == "topology":
            octant = compute_octant_index(target_coords).long() if target_coords.shape[0] > 0 else window_pos.long()
            xyz = target_coords[:, 1:].long() if target_coords.shape[0] > 0 else target_coords.new_zeros((0, 3))
            if target_coords.shape[0] > 0:
                level_from_leaf = max(int(full_num_depths) - int(depth), 0)
                xyz_m, _, _, _ = compute_physical_lidar_geometry(
                    target_coords,
                    q_base=self.q_base,
                    coord_shift_mm=self.coord_shift_mm,
                    level_from_leaf=level_from_leaf,
                    use_cell_center=True,
                )
                axis = torch.argmax(torch.abs(xyz_m), dim=1).long()
                sign = (xyz_m.gather(1, axis.view(-1, 1)).view(-1) < 0).long()
            else:
                axis = window_pos.long()
                sign = window_pos.long()
            parity = torch.remainder(xyz[:, 0] + xyz[:, 1] * 3 + xyz[:, 2] * 5, 7).long() if xyz.shape[0] > 0 else window_pos.long()
            priority = octant + axis * 3 + sign * 5 + parity + window_pos.long()
            stage_ids = torch.remainder(priority, self.num_causal_stages).long()
        elif self.stage_schedule == "morton":
            stage_ids = torch.remainder(window_pos + compute_octant_index(target_coords), self.num_causal_stages).long()
        else:
            raise ValueError(f"Unsupported stage_schedule={self.stage_schedule}.")
        num_windows = int(window_ids.max().item()) + 1 if target_coords.shape[0] > 0 else 0
        return {
            "order": order,
            "window_ids": window_ids,
            "window_pos": window_pos,
            "stage_ids": stage_ids,
            "num_windows": num_windows,
        }

    def build_stage_tokens(
        self,
        s0_symbols: torch.Tensor,
        s1_symbols: torch.Tensor,
        raw_occ: torch.Tensor,
        feat_group: torch.Tensor,
        gate_group: torch.Tensor,
        window_pos: torch.Tensor,
        stage_ids: torch.Tensor,
    ):
        if not self.use_stage_memory or raw_occ.shape[0] == 0:
            return feat_group.new_zeros((raw_occ.shape[0], self.channels))

        occ_emb = self.prior_embedding(raw_occ.view(-1).long().clamp(0, 255))
        s0_emb = self.nibble0_emb(s0_symbols.view(-1).long().clamp(0, 15))
        s1_emb = self.nibble1_emb(s1_symbols.view(-1).long().clamp(0, 15))
        stage_emb = self.stage_id_emb(stage_ids.view(-1).long().clamp(0, self.num_causal_stages - 1))
        pos_ids = window_pos.clamp(max=self.window_size - 1).long()
        pos_emb = self.window_pos_emb(pos_ids)
        token_in = torch.cat((feat_group, gate_group, occ_emb, pos_emb + s0_emb + s1_emb + stage_emb), dim=-1)
        return self.stage_mem_token(token_in)

    def build_stage_token_sums(self, s0_symbols, s1_symbols, raw_occ, feat_group,
                               gate_group, window_ids, window_pos, stage_ids, num_windows):
        token_sum = feat_group.new_zeros((num_windows, self.channels))
        token_count = feat_group.new_zeros((num_windows, 1))
        tokens = self.build_stage_tokens(s0_symbols, s1_symbols, raw_occ,
                                         feat_group, gate_group, window_pos, stage_ids)
        if num_windows == 0 or tokens.shape[0] == 0:
            return token_sum, token_count
        # Keep accumulation dtype aligned with the window buffers for index_add_.
        token_sum.index_add_(0, window_ids.long(), tokens.to(token_sum.dtype))
        token_count.index_add_(0, window_ids.long(), token_count.new_ones((tokens.shape[0], 1)))
        return token_sum, token_count

    def memory_from_state(self, token_sum: torch.Tensor, token_count: torch.Tensor):
        if token_sum.shape[0] == 0 or not self.use_stage_memory:
            return token_sum
        return token_sum / token_count.clamp(min=1.0)

    def fuse_stage_context(self, stage_feats: torch.Tensor, gate_ctx: torch.Tensor, memory_ctx: torch.Tensor):
        stage_feats = self.stage_feat_fuse(torch.cat((stage_feats, memory_ctx), dim=-1))
        gate_ctx = self.stage_gate_fuse(torch.cat((gate_ctx, memory_ctx), dim=-1))
        return stage_feats, gate_ctx

    def predict_primary(self, target_feats: torch.Tensor, gate_ctx: torch.Tensor):
        return self.moe_head_s0(target_feats, gate_ctx)

    def predict_secondary(self, target_feats: torch.Tensor, primary_symbols: torch.Tensor, gate_ctx: torch.Tensor):
        primary_symbols = primary_symbols.view(-1).long().clamp(0, 15)
        context = target_feats + self.nibble0_emb(primary_symbols)
        return self.moe_head_s1(context, gate_ctx)

    def raw_occ_to_symbols(
        self,
        raw_occ: torch.Tensor,
        target_coords: torch.Tensor,
        depth: int,
        full_num_depths: int,
    ):
        return split_occupancy_nibbles(raw_occ)

    def symbols_to_raw_occ(
        self,
        primary_symbols: torch.Tensor,
        secondary_symbols: torch.Tensor,
        target_coords: torch.Tensor,
        depth: int,
        full_num_depths: int,
    ):
        return combine_occupancy_nibbles(primary_symbols, secondary_symbols)

    def forward_from_hierarchy(self, data_ls, raw_num_points=None, return_details: bool = False):
        self.reset_frame_state()
        try:
            return self._forward_from_hierarchy(data_ls, raw_num_points, return_details)
        finally:
            self.reset_frame_state()

    def _forward_from_hierarchy(self, data_ls, raw_num_points=None, return_details: bool = False):
        n_points = self._resolve_num_points(raw_num_points, data_ls[-1][0].shape[0])
        device = data_ls[0][0].device
        total_bits = torch.tensor(0.0, device=device)
        depth_bits = []
        depth_outputs = [] if return_details else None
        ancestor_cache = []
        full_num_depths = len(data_ls) - 1

        for depth in range(len(data_ls) - 1):
            x_c, x_o = data_ls[depth]
            gt_x_up_c, gt_x_up_o = data_ls[depth + 1]
            gt_x_up_c, gt_x_up_o = sort_by_morton(gt_x_up_c, gt_x_up_o)
            depth_bit_total = torch.tensor(0.0, device=device)

            x_prior, prior_geo = self.prior_branch(x_c, x_o, depth, full_num_depths=full_num_depths)
            ancestor_cache.append(self._make_cache_entry(x_c, x_prior.feats, prior_geo, depth))

            x_up, target_geo = self.target_branch(x_c, x_o, x_prior.feats, depth, full_num_depths=full_num_depths)
            self.assert_aligned_coords(x_up.coords, gt_x_up_c, depth=depth, context="train")
            mod_feats, ancestor_ctx, gate_ctx = self.build_geo_context(x_up, target_geo, depth, ancestor_cache)

            gt_primary, gt_secondary = self.raw_occ_to_symbols(
                gt_x_up_o,
                x_up.coords,
                depth=depth,
                full_num_depths=full_num_depths,
            )

            stage = self.build_stage_partition(x_up.coords, depth=depth, full_num_depths=full_num_depths)
            token_sum = mod_feats.new_zeros((stage["num_windows"], self.channels))
            token_count = mod_feats.new_zeros((stage["num_windows"], 1))
            stage_details = [] if return_details else None

            for stage_idx in range(self.num_causal_stages):
                stage_mask = self.select_stage(stage, stage_idx)
                if self.stage_empty(stage, stage_idx, stage_mask):
                    if return_details:
                        stage_details.append(
                            {
                                "stage": stage_idx,
                                "prob_primary": mod_feats.new_zeros((0, 16)),
                                "prob_secondary": mod_feats.new_zeros((0, 16)),
                            }
                        )
                    continue

                feat_s = mod_feats[stage_mask].contiguous()
                gate_s = gate_ctx[stage_mask].contiguous()
                if self.use_stage_memory and token_sum.shape[0] > 0 and int(stage_idx) > 0:
                    mem_s = self.memory_from_state(token_sum, token_count)[stage["window_ids"][stage_mask].contiguous().long()]
                    feat_s, gate_s = self.fuse_stage_context(feat_s, gate_s, mem_s)

                primary_s = gt_primary[stage_mask].contiguous().long()
                secondary_s = gt_secondary[stage_mask].contiguous().long()
                if self.rate_loss == 'log_softmax':
                    logits_primary = self.moe_head_s0.forward_logits(feat_s, gate_s)
                    logits_secondary = self.moe_head_s1.forward_logits(feat_s + self.nibble0_emb(primary_s), gate_s)
                    log_primary = torch.log_softmax(logits_primary, dim=-1).gather(1, primary_s.view(-1, 1))
                    log_secondary = torch.log_softmax(logits_secondary, dim=-1).gather(1, secondary_s.view(-1, 1))
                    depth_bit_total = depth_bit_total - log_primary.sum() / math.log(2.)
                    depth_bit_total = depth_bit_total - log_secondary.sum() / math.log(2.)
                    if return_details:
                        prob_primary = torch.softmax(logits_primary, dim=-1)
                        prob_secondary = torch.softmax(logits_secondary, dim=-1)
                elif self.rate_loss == 'floored':
                    prob_primary = self.predict_primary(feat_s, gate_s)
                    prob_secondary = self.predict_secondary(feat_s, primary_s, gate_s)
                    gt_prob_primary = prob_primary.gather(1, primary_s.view(-1, 1))
                    gt_prob_secondary = prob_secondary.gather(1, secondary_s.view(-1, 1))
                    depth_bit_total = depth_bit_total + torch.sum(torch.clamp(-torch.log2(gt_prob_primary + 1e-9), 0, 50))
                    depth_bit_total = depth_bit_total + torch.sum(torch.clamp(-torch.log2(gt_prob_secondary + 1e-9), 0, 50))
                else:
                    raise ValueError('Unknown rate_loss: ' + self.rate_loss)

                new_sum, new_count = self.build_stage_token_sums(
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

                if return_details:
                    stage_details.append(
                        {
                            "stage": stage_idx,
                            "prob_primary": prob_primary,
                            "prob_secondary": prob_secondary,
                            "feat": feat_s,
                            "gate": gate_s,
                        }
                    )

            if return_details:
                depth_outputs.append(
                    {
                        "depth": depth,
                        "stage_details": stage_details,
                        "target_geo": target_geo,
                        "ancestor_ctx": ancestor_ctx,
                        "gate_ctx": gate_ctx,
                    }
                )

            depth_bits.append(depth_bit_total)
            total_bits = total_bits + depth_bit_total

        bpp = total_bits / max(int(n_points), 1)
        prefix_bpp = {}
        prefix_loss = torch.tensor(0.0, device=device)
        depth_count = len(depth_bits)
        offset_weight_pairs = zip(self.depth_offsets, self.prefix_weights)
        for offset, weight in offset_weight_pairs:
            target_depths = max(depth_count - max(int(offset), 0), 0)
            if target_depths == 0:
                prefix_bits = torch.tensor(0.0, device=device)
            else:
                prefix_bits = torch.stack(depth_bits[:target_depths]).sum()
            prefix_rate = prefix_bits / max(int(n_points), 1)
            prefix_bpp[int(offset)] = prefix_rate
            prefix_loss = prefix_loss + float(weight) * prefix_rate
        if not self.use_prefix_loss:
            prefix_loss = bpp
        train_loss = prefix_loss

        out = {
            "loss": train_loss,
            "bpp": bpp.detach(),
            "symbol": self.symbol_name,
            "prefix_bpp": {int(offset): value.detach() for offset, value in prefix_bpp.items()},
            "full_num_depths": full_num_depths,
        }
        if return_details:
            out["depth_outputs"] = depth_outputs
            out["depth_bits"] = depth_bits
        return out

    def forward(self, x, raw_num_points=None, return_details: bool = False, base_cut: int = None):
        data_ls = self.build_hierarchy(x, base_cut=base_cut)
        return self.forward_from_hierarchy(data_ls, raw_num_points=raw_num_points, return_details=return_details)


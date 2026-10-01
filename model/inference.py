"""Validated inference execution; learned parameters are identical to Network."""
import torch
from torchsparse import SparseTensor

from model.liacm_network import Network
from model.entropy_context import build_window_partition, morton_sort_order, scatter_mean_by_window
from model.sparse_blocks import LightweightMultiScaleFusion2


class InferenceNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for module in self.modules():
            if isinstance(module, LightweightMultiScaleFusion2):
                module.preserve_cache = True

    def reset_frame_state(self):
        super().reset_frame_state()
        self._parent_maps = {}
        self._previous_target_geo = None
        self._order_safe = False
        self._single_batch = False
        self._stage_fast = False
        self._stage_call = 0
        self._scratch = None

    def prior_branch(self, coords, occupancy, depth, full_num_depths=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Use Network for training; InferenceNetwork requires no_grad')
        if int(depth) == 0:
            self.reset_frame_state()
            remaining = int(full_num_depths) if full_num_depths is not None else 64
            if 0 <= remaining <= 21 and coords.numel():
                xyz = coords[:, 1:].long()
                scale = 1 << remaining
                valid = torch.all(xyz >= 0) & torch.all(xyz*scale + scale-1 < (1 << 21))
                self._order_safe = bool(valid) and torch.equal(
                    morton_sort_order(coords), torch.arange(len(coords), device=coords.device))
        cached = self._previous_target_geo
        self._previous_target_geo = None
        if cached is None:
            geometry = self.geo_stem(coords, depth, full_num_depths=full_num_depths)
        else:
            previous_depth, previous_full, previous_coords, geometry = cached
            if (previous_depth != int(depth) or previous_full != full_num_depths
                    or previous_coords.shape != coords.shape):
                raise RuntimeError('Geometry cache level or shape mismatch')
            if self.debug_coord_check and not torch.equal(previous_coords, coords):
                raise RuntimeError('Geometry cache coordinate mismatch')
        x = self.prior_resnet(SparseTensor(coords=coords,
            feats=self._prior_input(coords, occupancy, geometry, depth)))
        if self.use_geo_modulation:
            x.feats = self.prior_geo_mod(x.feats, geometry)
        return x, geometry

    def target_branch(self, coords, occupancy, prior_feats, depth, full_num_depths=None):
        child_coords, features = self.fcg(coords, occupancy, prior_feats)
        bits = (occupancy.long().view(-1, 1) >> torch.arange(8, device=occupancy.device)) & 1
        parent_ids = torch.nonzero(bits, as_tuple=False)[:, 0]
        if not self._order_safe:
            order = morton_sort_order(child_coords)
            child_coords, features, parent_ids = child_coords[order], features[order], parent_ids[order]
        self._parent_maps[int(depth)+1] = parent_ids
        if self.debug_coord_check:
            parent = child_coords.clone()
            parent[:, 1:] = torch.div(parent[:, 1:], 2, rounding_mode='floor')
            if not torch.equal(parent, coords[parent_ids]):
                raise RuntimeError('Direct parent links mismatch')
        geometry = self.geo_stem(child_coords, depth+1, full_num_depths=full_num_depths)
        x = self.target_resnet(SparseTensor(coords=child_coords,
            feats=self.target_embedding(features, child_coords)+geometry))
        if self.use_geo_modulation:
            x.feats = self.target_geo_mod(x.feats, geometry)
        self._previous_target_geo = (int(depth)+1, full_num_depths, x.coords, geometry)
        self._commit_target(x, depth)
        return x, geometry

    def _make_cache_entry(self, coords, feat, geo, depth):
        if int(depth) == 0:
            self._single_batch = bool(len(coords)) and bool(torch.all(coords[:, 0] == 0))
        return dict(coords=coords, feat=feat, geo=geo, depth=int(depth), feat_geo=feat+geo)

    def _gather_ancestor_indices(self, coords, current_depth, ancestor_cache):
        indices = coords.new_full((len(coords), self.ancestor_depth), -1, dtype=torch.long)
        octants = coords.new_zeros((len(coords), self.ancestor_depth), dtype=torch.long)
        current = None
        for step in range(1, self.ancestor_depth+1):
            layer = int(current_depth)+1-step
            if layer < 0:
                continue
            mapping = self._parent_maps[layer+1]
            current = mapping if current is None else mapping[current]
            indices[:, step-1] = current
            relative = torch.div(coords[:, 1:], 2**(step-1), rounding_mode='floor')
            octants[:, step-1] = relative[:, 0] % 2 + 2*(relative[:, 1] % 2) + 4*(relative[:, 2] % 2)
        return indices, octants

    def _encode_ancestor_context(self, coords, current_depth, ancestor_cache):
        if not self.use_ancestor_context or self.ancestor_depth == 0:
            return coords.new_zeros((len(coords), self.channels), dtype=torch.float32)
        indices, octants = self._gather_ancestor_indices(coords, current_depth, ancestor_cache)
        tokens = []
        for step in range(self.ancestor_depth):
            layer = int(current_depth)-step
            if layer < 0:
                tokens.append(coords.new_zeros((len(coords), self.channels), dtype=torch.float32))
                continue
            token = ancestor_cache[layer]['feat_geo'][indices[:, step]] + self.ancestor_step_emb.weight[step+1]
            tokens.append(token + self.rel_octant_emb(octants[:, step]))
        return self.ancestor_proj(torch.cat(tokens, dim=-1))

    def build_geo_context(self, x_up, target_geo, current_depth, ancestor_cache):
        ancestor = self._encode_ancestor_context(x_up.coords, current_depth, ancestor_cache)
        gate = self.gate_context_fuse(torch.cat((target_geo, ancestor), dim=-1))
        if self.use_macro_context and len(x_up.coords):
            if self._single_batch:
                macro_ids = torch.div(torch.arange(len(x_up.coords), device=x_up.coords.device),
                                       self.macro_window_size, rounding_mode='floor')
                num_macro = (len(x_up.coords)+self.macro_window_size-1)//self.macro_window_size
            else:
                macro_ids, _, _ = build_window_partition(x_up.coords, self.macro_window_size)
                num_macro = int(macro_ids.max().item())+1
            summary = scatter_mean_by_window(gate, macro_ids.long(), num_macro)
            gate = self.macro_context_fuse(torch.cat((gate, summary[macro_ids.long()]), dim=-1))
        features = self.context_geo_mod(x_up.feats, gate) if self.use_geo_modulation else x_up.feats
        return features, ancestor, gate

    def build_stage_partition(self, coords, depth=0, full_num_depths=0):
        self._stage_call, self._scratch = 0, None
        self._stage_fast = bool(self.use_stage_memory and self.stage_schedule == 'modulo'
            and self.window_size == self.num_causal_stages and len(coords)
            and bool(torch.all(coords[:, 0] == 0)))
        if not self._stage_fast:
            return super().build_stage_partition(coords, depth, full_num_depths)
        order = torch.arange(len(coords), device=coords.device)
        positions = torch.remainder(order, self.window_size)
        self._stage_steps = min(len(coords), self.num_causal_stages)
        return dict(order=order, window_ids=torch.div(order, self.window_size, rounding_mode='floor'),
            window_pos=positions, stage_ids=positions,
            num_windows=(len(coords)+self.window_size-1)//self.window_size, _slice_count=len(coords))

    def select_stage(self, stage, index):
        if '_slice_count' in stage:
            return slice(int(index), stage['_slice_count'], self.num_causal_stages)
        return super().select_stage(stage, index)

    def stage_empty(self, stage, index, selection):
        if isinstance(selection, slice):
            return int(index) >= stage['_slice_count']
        return super().stage_empty(stage, index, selection)

    def build_stage_token_sums(self, s0, s1, raw_occ, feat, gate, window_ids,
                               window_pos, stage_ids, num_windows):
        if not self._stage_fast:
            return super().build_stage_token_sums(s0, s1, raw_occ, feat, gate,
                window_ids, window_pos, stage_ids, num_windows)
        step = self._stage_call
        self._stage_call += 1
        if self._stage_call > self._stage_steps:
            raise RuntimeError('Unexpected extra stage update')
        if self._scratch is None:
            self._scratch = (feat.new_zeros((num_windows, self.channels)), feat.new_zeros((num_windows, 1)))
        sums, counts = self._scratch
        sums.zero_()
        counts.zero_()
        # No same-level prediction reads the final stage; carry uses base features.
        if step < self._stage_steps-1:
            tokens = self.build_stage_tokens(s0, s1, raw_occ, feat, gate, window_pos, stage_ids)
            sums[:len(tokens)].copy_(tokens)
            counts[:len(tokens)].fill_(1)
        return sums, counts

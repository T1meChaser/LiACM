"""Bitstream and entropy-coding helpers for LiACM."""

import os

import numpy as np
import torch


def pack_byte_stream_ls(byte_stream_ls):
    if len(byte_stream_ls) > 65535:
        raise ValueError('Too many entropy streams')
    stream = np.array(len(byte_stream_ls), dtype=np.uint16).tobytes()
    for byte_stream in byte_stream_ls:
        stream += np.array(len(byte_stream), dtype=np.uint32).tobytes()
        stream += byte_stream
    return stream


def unpack_byte_stream(stream):
    if len(stream) < 2:
        raise ValueError('Truncated stream count')
    len_bytes_stream_ls = np.frombuffer(stream[:2], dtype=np.uint16)[0]
    byte_stream_ls = []
    cursor = 2
    for _ in range(len_bytes_stream_ls):
        if cursor + 4 > len(stream):
            raise ValueError('Truncated stream length')
        len_bytes_stream = np.frombuffer(stream[cursor : cursor + 4], dtype=np.uint32)[0]
        if cursor + 4 + int(len_bytes_stream) > len(stream):
            raise ValueError('Truncated entropy payload')
        byte_stream = stream[cursor + 4 : cursor + 4 + len_bytes_stream]
        byte_stream_ls.append(byte_stream)
        cursor = cursor + 4 + len_bytes_stream
    if cursor != len(stream):
        raise ValueError('Trailing bytes after entropy payload')
    return byte_stream_ls


def _convert_to_int_and_normalize(cdf_float, needs_normalization):
    """Convert floating-point CDFs to strictly increasing int16 CDFs."""

    lp = cdf_float.shape[-1]
    factor = torch.tensor(2, dtype=torch.float32, device=cdf_float.device).pow_(16)
    new_max_value = factor - (lp - 1) if needs_normalization else factor
    cdf_float = cdf_float.mul(new_max_value).round()

    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
    if needs_normalization:
        cdf.add_(torch.arange(lp, dtype=torch.int16, device=cdf.device))
    return cdf


def get_file_size_in_bits(file_path):
    return os.stat(file_path).st_size * 8

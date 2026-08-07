#!/usr/bin/env python3
"""Quantize a llama2.c checkpoint into the RQ42 pack for the ESP32-S3 record run.

Reads karpathy's legacy .bin (fp32) and writes a single flashable image:

  [header 128B]
  [NORMS   fp32   ]  rms_att[L,dim], rms_ffn[L,dim], rms_final[dim] -> SRAM at boot
  [TABLE   Q4/Q3  ]  tied embedding/output rows for the KEPT vocab, group-quantized
  [CORE    Q4     ]  per layer: wq wk wv wo w1 w2 w3, group-quantized

The tied table is the vocab-trim target: stories42M is 21.5MB at 4-bit and the
flash ceiling is ~15.9MB, so the table keeps only the tokens that TinyStories
text actually uses. --keep-ids supplies that set (built by rank_vocab.py from a
generated corpus); byte-fallback tokens are force-included so any prompt still
encodes. Core weights are never trimmed.

Quantization: symmetric int per group of --group params along the input dim,
range [-(2^{b-1}-1) .. 2^{b-1}-1], one fp16 scale per group.
  Q4 g128: 4.125 bits/param    Q3 g128: 3.125 bits/param

Usage:
  python3 quantize.py model.bin -o pack.bin [--keep-ids ids.txt]
      [--table-bits 4] [--core-bits 4] [--group 128] [--seq-len N]
  python3 quantize.py --describe model.bin      # print config and exit
"""

import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

MAGIC = 0x32345152          # "RQ42" little-endian
VERSION = 1
HEADER_BYTES = 128


# ------------------------------------------------------------- checkpoint io

def read_checkpoint(path):
    """Parse llama2.c legacy .bin: 7-int header, fp32 weights, tied classifier
    when vocab_size is positive. Returns (config dict, dict of arrays)."""
    raw = Path(path).read_bytes()
    dim, hidden, n_layers, n_heads, n_kv, vocab, seq = struct.unpack_from("<7i", raw)
    tied = vocab > 0
    vocab = abs(vocab)
    head_size = dim // n_heads
    kv_dim = n_kv * head_size

    off = 28
    def take(*shape):
        nonlocal off
        n = int(np.prod(shape))
        a = np.frombuffer(raw, dtype="<f4", count=n, offset=off).reshape(shape)
        off += 4 * n
        return a.astype(np.float32)

    w = {}
    w["tok_emb"]  = take(vocab, dim)
    w["rms_att"]  = take(n_layers, dim)
    w["wq"]       = take(n_layers, dim, dim)
    w["wk"]       = take(n_layers, kv_dim, dim)
    w["wv"]       = take(n_layers, kv_dim, dim)
    w["wo"]       = take(n_layers, dim, dim)
    w["rms_ffn"]  = take(n_layers, dim)
    w["w1"]       = take(n_layers, hidden, dim)
    w["w2"]       = take(n_layers, dim, hidden)
    w["w3"]       = take(n_layers, hidden, dim)
    w["rms_final"] = take(dim)
    off += 4 * seq * (head_size // 2) * 2          # legacy freq_cis, skipped
    if not tied:
        w["wcls"] = take(vocab, dim)

    if off > len(raw):
        raise SystemExit(f"{path}: truncated ({off} needed, {len(raw)} present)")
    cfg = dict(dim=dim, hidden=hidden, n_layers=n_layers, n_heads=n_heads,
               n_kv_heads=n_kv, vocab=vocab, seq_len=seq, tied=int(tied))
    return cfg, w


# --------------------------------------------------------------- quantizers

def pad_cols(mat, gs):
    """Zero-pad axis 1 to a multiple of gs (zeros quantize exactly and add
    nothing to any dot product)."""
    rows, cols = mat.shape
    pad = (-cols) % gs
    if pad == 0:
        return mat
    return np.concatenate([mat, np.zeros((rows, pad), np.float32)], axis=1)


def pad_rows(mat, gs_of_consumer):
    """Zero-pad axis 0 to a multiple of the CONSUMER's group size. Extra
    output rows are exact zeros; SiLU(0)*0 = 0 keeps them inert."""
    rows, cols = mat.shape
    pad = (-rows) % gs_of_consumer
    if pad == 0:
        return mat
    return np.concatenate([mat, np.zeros((pad, cols), np.float32)], axis=0)


def quant_groups(mat, bits, gs):
    """Quantize a 2D array row-wise in groups of gs along axis 1.
    Returns (packed bytes, max group quantization error)."""
    rows, cols = mat.shape
    if cols % gs:
        raise SystemExit(f"internal: input dim {cols} not divisible by group "
                         f"{gs} after padding: this is a bug, report it")
    qmax = (1 << (bits - 1)) - 1                    # 7 for Q4, 3 for Q3
    g = mat.reshape(rows, cols // gs, gs)
    scale = np.abs(g).max(axis=2) / qmax
    scale[scale == 0] = 1e-10
    q = np.clip(np.round(g / scale[..., None]), -qmax, qmax).astype(np.int8)
    err = float(np.abs(q * scale[..., None] - g).max())

    out = bytearray()
    sc16 = scale.astype("<f2")
    if bits == 4:
        u = (q + 8).astype(np.uint8)                # offset-binary nibbles
        packed = (u[..., 0::2] | (u[..., 1::2] << 4)).astype(np.uint8)
        for r in range(rows):
            for k in range(cols // gs):
                out += packed[r, k].tobytes()
                out += sc16[r, k].tobytes()
    elif bits == 3:
        u = (q + 4).astype(np.uint16)               # 3-bit offset-binary
        for r in range(rows):
            for k in range(cols // gs):
                bits_acc, nbits, buf = 0, 0, bytearray()
                for v in u[r, k]:
                    bits_acc |= int(v) << nbits
                    nbits += 3
                    while nbits >= 8:
                        buf.append(bits_acc & 0xFF)
                        bits_acc >>= 8
                        nbits -= 8
                if nbits:
                    buf.append(bits_acc & 0xFF)
                out += buf                            # gs=128 -> 48 bytes
                out += sc16[r, k].tobytes()
    else:
        raise SystemExit(f"unsupported bit width {bits}")
    return bytes(out), err


def group_bytes(cols, bits, gs):
    per = {4: gs // 2, 3: (gs * 3 + 7) // 8}[bits] + 2
    return (cols // gs) * per


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Build the RQ42 pack.")
    ap.add_argument("model", help="llama2.c fp32 .bin checkpoint")
    ap.add_argument("-o", "--out", default="pack.bin")
    ap.add_argument("--keep-ids", help="file with one kept token id per line; "
                    "omit to keep the full vocabulary")
    ap.add_argument("--table-bits", type=int, default=4, choices=(3, 4))
    ap.add_argument("--core-bits", type=int, default=4, choices=(4,))
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=0,
                    help="override stored seq_len (device KV budget)")
    ap.add_argument("--describe", action="store_true")
    args = ap.parse_args()

    cfg, w = read_checkpoint(args.model)
    params_core = sum(int(np.prod(w[k].shape)) for k in
                      ("wq", "wk", "wv", "wo", "w1", "w2", "w3"))
    if args.describe:
        total = params_core + cfg["vocab"] * cfg["dim"] * (1 if cfg["tied"] else 2)
        print(f"config : {cfg}")
        print(f"params : core {params_core/1e6:.2f}M + table "
              f"{cfg['vocab']*cfg['dim']/1e6:.2f}M = {total/1e6:.2f}M "
              f"({'tied' if cfg['tied'] else 'untied'})")
        return

    if not cfg["tied"]:
        raise SystemExit("untied checkpoints not supported in v1 (42M is tied)")
    if cfg["dim"] % args.group:
        raise SystemExit(
            f"dim {cfg['dim']} not divisible by group {args.group}; padding dim "
            f"would corrupt rmsnorm: use a group size that divides it "
            f"(e.g. --group 32 for the 260K smoke model)")

    # The FFN hidden dim rarely divides the group size (42M: 1376 % 128 = 96).
    # Pad w1/w3 output rows and w2 input columns with zeros to the next
    # multiple: SiLU(0)*0 = 0, so padded lanes are inert end to end. The header
    # stores the PADDED hidden; the runtime never needs the original.
    hidden_true = cfg["hidden"]
    hidden_pad = -(-cfg["hidden"] // args.group) * args.group
    if hidden_pad != cfg["hidden"]:
        extra = hidden_pad - cfg["hidden"]
        w["w1"] = np.pad(w["w1"], ((0, 0), (0, extra), (0, 0)))
        w["w2"] = np.pad(w["w2"], ((0, 0), (0, 0), (0, extra)))
        w["w3"] = np.pad(w["w3"], ((0, 0), (0, extra), (0, 0)))
        print(f"  note: hidden {cfg['hidden']} padded to {hidden_pad} "
              f"(+{100.0 * extra / cfg['hidden']:.1f}% inert zero lanes)")
        cfg["hidden"] = hidden_pad

    # ---- kept vocabulary ----------------------------------------------------
    if args.keep_ids:
        keep = sorted({int(l) for l in Path(args.keep_ids).read_text().split()})
        if any(i < 0 or i >= cfg["vocab"] for i in keep):
            raise SystemExit("keep-ids out of range")
        # BOS/EOS/UNK plus the 256 byte-fallback tokens (ids 3..258 in the
        # llama2.c tokenizer) must survive, or prompts stop encoding.
        forced = set(range(0, min(259, cfg["vocab"])))
        keep = sorted(set(keep) | forced)
    else:
        keep = list(range(cfg["vocab"]))
    keep = np.asarray(keep, dtype=np.int64)
    V = len(keep)

    table = w["tok_emb"][keep]                       # (V, dim)

    # ---- quantize -----------------------------------------------------------
    t_bytes, t_err = quant_groups(table, args.table_bits, args.group)
    core_parts, c_err = [], 0.0
    for L in range(cfg["n_layers"]):
        for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
            b, e = quant_groups(w[name][L], args.core_bits, args.group)
            core_parts.append(b)
            c_err = max(c_err, e)
    core_bytes = b"".join(core_parts)

    norms = np.concatenate([w["rms_att"].ravel(), w["rms_ffn"].ravel(),
                            w["rms_final"].ravel()]).astype("<f4").tobytes()

    # ---- assemble -----------------------------------------------------------
    seq_out = args.seq_len or cfg["seq_len"]
    norms_off = HEADER_BYTES
    table_off = norms_off + len(norms)
    core_off = table_off + len(t_bytes)
    payload = norms + t_bytes + core_bytes
    remap = keep.astype("<u2").tobytes()             # kept row -> original id
    remap_off = core_off + len(core_bytes)

    header = struct.pack(
        "<IIiiiiiiiiiiiiIIIIIIIII",
        MAGIC, VERSION,
        cfg["dim"], cfg["hidden"], cfg["n_layers"], cfg["n_heads"],
        cfg["n_kv_heads"], V, cfg["vocab"], seq_out,
        args.group, args.core_bits, args.table_bits,
        hidden_true,
        1,                                            # flags: tied
        norms_off, len(norms),
        table_off, len(t_bytes),
        core_off, len(core_bytes),
        remap_off, len(remap))
    crc = zlib.crc32(payload + remap) & 0xFFFFFFFF
    header += struct.pack("<I", crc)
    header += b"\0" * (HEADER_BYTES - len(header))

    blob = header + payload + remap
    Path(args.out).write_bytes(blob)

    hs = cfg["dim"] // cfg["n_heads"]
    kvd = hs * cfg["n_kv_heads"]
    core_true = cfg["n_layers"] * (2 * cfg["dim"] * cfg["dim"]
                                   + 2 * kvd * cfg["dim"]
                                   + 3 * hidden_true * cfg["dim"])
    stored = core_true + V * cfg["dim"]
    print(f"wrote {args.out}")
    print(f"  vocab      {cfg['vocab']} -> {V} kept")
    print(f"  stored     {stored/1e6:.2f}M params "
          f"(core {core_true/1e6:.2f}M Q{args.core_bits} + "
          f"table {V*cfg['dim']/1e6:.2f}M Q{args.table_bits})")
    print(f"  size       {len(blob)/1048576:.2f} MB "
          f"(norms {len(norms)/1024:.0f}K, table {len(t_bytes)/1048576:.2f}M, "
          f"core {len(core_bytes)/1048576:.2f}M)")
    print(f"  max q-err  table {t_err:.4f}, core {c_err:.4f}")
    print(f"  crc32      {crc:08x}")


if __name__ == "__main__":
    main()

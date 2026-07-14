from pathlib import Path

code = r'''#!/usr/bin/env python3
"""
run_full_model_simple.py

Safe, unfused full FLAIR NPU pipeline.

This intentionally avoids the fused decoder and uses batch=1 first so failures
are easy to localize.

Pipeline:
  1. host: build padded encoder windows -> all_x_windows.bin
  2. host: pack padded encoder params   -> enc_params.bin
  3. NPU : encoder                      -> all_latents.bin
  4. host: latent_to_hidden             -> all_h0.bin
  5. host: pack decoder GRU params      -> dec_gru_params.bin
  6. NPU : unfused decoder GRU          -> all_hidden_seq.bin
  7. host: hidden_to_output + MSE       -> npu_scores_simple.npy/.csv

Run from npu/:
  python3 run_full_model_simple.py --limit 20

For a one-window smoke test:
  python3 run_full_model_simple.py --limit 1 --skip-cpu-baseline

To reuse already-built xclbins and batch_infer.exe:
  python3 run_full_model_simple.py --limit 20 --skip-build
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


NPU_DIR = Path(__file__).resolve().parent
REPO = NPU_DIR.parent
sys.path.insert(0, str(REPO))

INPUT_DIM = 45
INPUT_DIM_PADDED = 48
HIDDEN_DIM = 64
OUTPUT_DIM = 21
SEQ_LEN_DEFAULT = 10

CKPT = REPO / "experiments" / "results" / "flair_minimal.pt"
NPZ_DEFAULT = REPO / "data" / "processed" / "preprocessed.npz"


def sh(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=NPU_DIR, check=True)


def check_finite(name: str, arr: np.ndarray) -> None:
    arr_f = arr.astype(np.float32)
    bad = ~np.isfinite(arr_f)
    if bad.any():
        idx = np.argwhere(bad)[0]
        raise RuntimeError(
            f"{name} has non-finite value at {idx.tolist()}: {arr_f[tuple(idx)]}"
        )
    print(f"[OK] {name}: shape={arr.shape}, min={arr_f.min():.6g}, max={arr_f.max():.6g}")


def write_bf16(path: Path, arr: np.ndarray) -> None:
    arr.astype(bfloat16).tofile(path)


def read_bf16(path: Path, count: int, shape: tuple[int, ...]) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=bfloat16)
    if data.size < count:
        raise RuntimeError(f"{path.name} too small: got {data.size}, expected {count}")
    if data.size > count:
        print(f"[WARN] {path.name} has {data.size} values; using first {count}")
        data = data[:count]
    return data.reshape(shape)


def build_artifacts(seq_len: int) -> None:
    # Compile encoder and decoder xclbins with batch=1, then build generic host.
    sh([
        "python3", "gru_encoder.py",
        "--dev", "npu",
        "--input-dim", str(INPUT_DIM_PADDED),
        "--hidden-dim", str(HIDDEN_DIM),
        "--seq-len", str(seq_len),
        "--batch", "1",
        "--xclbin-path", "build/gru.xclbin",
        "--insts-path", "build/insts.bin",
    ])

    sh([
        "python3", "gru_decoder.py",
        "--dev", "npu",
        "--hidden-dim", str(HIDDEN_DIM),
        "--seq-len", str(seq_len),
        "--batch", "1",
        "--xclbin-path", "build/decoder.xclbin",
        "--insts-path", "build/decoder_insts.bin",
    ])

    sh(["make", "-f", "Makefile.batch"])


def run_batch_infer(
    *,
    xclbin: str,
    insts: str,
    in1: str,
    in2: str,
    out: str,
    N: int,
    in1_vol: int,
    in2_vol: int,
    out_vol: int,
) -> None:
    # batch=1 by design for this simple debug pipeline.
    sh([
        "powershell.exe", "./batch_infer.exe",
        xclbin,
        insts,
        in1,
        in2,
        out,
        str(N),
        "1",
        str(in1_vol),
        str(in2_vol),
        str(out_vol),
        "MLIR_AIE",
    ])


def main() -> None:
    import torch
    from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig

    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str, default=str(NPZ_DEFAULT))
    p.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT)
    p.add_argument("--limit", type=int, default=20, help="0 = all windows")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-cpu-baseline", action="store_true")
    args = p.parse_args()

    T = args.seq_len
    if T != 10:
        print("[WARN] This project has mostly been validated with SEQ_LEN=10.")

    ckpt = torch.load(str(CKPT), map_location="cpu")
    sd = ckpt["model_state_dict"]

    bundle = np.load(args.npz, allow_pickle=True)
    X_num = bundle["X_num"].astype(np.float32)
    X_cat = bundle["X_cat"].astype(np.int64)
    y = bundle["y_seq"].astype(np.int64) if "y_seq" in bundle else None

    N_total = X_num.shape[0]
    N = N_total if args.limit == 0 else min(args.limit, N_total)
    X_num = X_num[:N]
    X_cat = X_cat[:N]
    if y is not None:
        y = y[:N]

    print(f"Dataset slice: N={N}, T={T}")
    if y is not None:
        print(f"Labels: anomalies={int(y.sum())}, normal={int((y == 0).sum())}")

    # ------------------------------------------------------------------
    # Build files for encoder input and encoder params.
    # ------------------------------------------------------------------
    sport_w = sd["sport_emb.weight"].numpy()
    dport_w = sd["dport_emb.weight"].numpy()
    proto_w = sd["proto_emb.weight"].numpy()

    x_windows = np.zeros((N, T, INPUT_DIM_PADDED), dtype=bfloat16)
    for w in range(N):
        for t in range(T):
            xin = np.concatenate([
                X_num[w, t],
                sport_w[X_cat[w, t, 0]],
                dport_w[X_cat[w, t, 1]],
                proto_w[X_cat[w, t, 2]],
            ]).astype(bfloat16)
            if xin.size != INPUT_DIM:
                raise RuntimeError(f"expected {INPUT_DIM} encoder features, got {xin.size}")
            x_windows[w, t, :INPUT_DIM] = xin

    check_finite("x_windows", x_windows)
    write_bf16(NPU_DIR / "all_x_windows.bin", x_windows.reshape(N, -1))

    w_ih_e = sd["encoder.gru.weight_ih_l0"].numpy().astype(bfloat16)
    w_hh_e = sd["encoder.gru.weight_hh_l0"].numpy().astype(bfloat16)
    b_ih_e = sd["encoder.gru.bias_ih_l0"].numpy().astype(bfloat16)
    b_hh_e = sd["encoder.gru.bias_hh_l0"].numpy().astype(bfloat16)

    w_ih_e_pad = np.zeros((3 * HIDDEN_DIM, INPUT_DIM_PADDED), dtype=bfloat16)
    w_ih_e_pad[:, :INPUT_DIM] = w_ih_e

    enc_params = np.concatenate([
        w_ih_e_pad.reshape(-1),
        w_hh_e.reshape(-1),
        b_ih_e,
        b_hh_e,
    ]).astype(bfloat16)
    check_finite("enc_params", enc_params)
    write_bf16(NPU_DIR / "enc_params.bin", enc_params)

    enc_in1_vol = T * INPUT_DIM_PADDED
    enc_out_vol = HIDDEN_DIM
    enc_params_vol = enc_params.size
    print(f"Encoder vols: in1={enc_in1_vol}, params={enc_params_vol}, out={enc_out_vol}")

    # ------------------------------------------------------------------
    # Build xclbins/host if requested.
    # ------------------------------------------------------------------
    if not args.skip_build:
        build_artifacts(T)
    else:
        print("[SKIP] build")

    # ------------------------------------------------------------------
    # Encoder NPU pass.
    # ------------------------------------------------------------------
    run_batch_infer(
        xclbin="build/gru.xclbin",
        insts="build/insts.bin",
        in1="all_x_windows.bin",
        in2="enc_params.bin",
        out="all_latents.bin",
        N=N,
        in1_vol=enc_in1_vol,
        in2_vol=enc_params_vol,
        out_vol=enc_out_vol,
    )

    latents_bf = read_bf16(NPU_DIR / "all_latents.bin", N * HIDDEN_DIM, (N, HIDDEN_DIM))
    check_finite("latents_bf", latents_bf)
    latents = latents_bf.astype(np.float32)

    # ------------------------------------------------------------------
    # Host bridge: latent_to_hidden -> h0.
    # Use bf16-quantized weights, because h0 is sent to the NPU as bf16.
    # ------------------------------------------------------------------
    W_lh = sd["decoder.latent_to_hidden.weight"].numpy().astype(bfloat16).astype(np.float32)
    b_lh = sd["decoder.latent_to_hidden.bias"].numpy().astype(bfloat16).astype(np.float32)

    h0 = np.tanh(latents @ W_lh.T + b_lh).astype(bfloat16)
    check_finite("h0", h0)
    write_bf16(NPU_DIR / "all_h0.bin", h0)

    # ------------------------------------------------------------------
    # Pack decoder GRU params only, not hidden_to_output.
    # ------------------------------------------------------------------
    w_ih_d = sd["decoder.gru.weight_ih_l0"].numpy().astype(bfloat16)
    w_hh_d = sd["decoder.gru.weight_hh_l0"].numpy().astype(bfloat16)
    b_ih_d = sd["decoder.gru.bias_ih_l0"].numpy().astype(bfloat16)
    b_hh_d = sd["decoder.gru.bias_hh_l0"].numpy().astype(bfloat16)

    dec_gru_params = np.concatenate([
        w_ih_d.reshape(-1),
        w_hh_d.reshape(-1),
        b_ih_d,
        b_hh_d,
    ]).astype(bfloat16)
    check_finite("dec_gru_params", dec_gru_params)
    write_bf16(NPU_DIR / "dec_gru_params.bin", dec_gru_params)

    dec_in1_vol = HIDDEN_DIM
    dec_params_vol = dec_gru_params.size
    dec_out_vol = T * HIDDEN_DIM
    print(f"Decoder vols: in1={dec_in1_vol}, params={dec_params_vol}, out={dec_out_vol}")

    # ------------------------------------------------------------------
    # Decoder NPU pass, unfused: h0 -> hidden_seq.
    # ------------------------------------------------------------------
    run_batch_infer(
        xclbin="build/decoder.xclbin",
        insts="build/decoder_insts.bin",
        in1="all_h0.bin",
        in2="dec_gru_params.bin",
        out="all_hidden_seq.bin",
        N=N,
        in1_vol=dec_in1_vol,
        in2_vol=dec_params_vol,
        out_vol=dec_out_vol,
    )

    hidden_seq_bf = read_bf16(
        NPU_DIR / "all_hidden_seq.bin

# DECODERRRRR+data


"""
gen_decoder_data.py

Writes binary inputs + a more NPU-like golden output for the decoder host
harness:

  decoder_h0.bin            : (HIDDEN_DIM,)        bf16
  decoder_gru_params.bin    : (N_PARAMS,)          bf16 [w_ih|w_hh|b_ih|b_hh]
  decoder_hidden_golden.bin : (SEQ_LEN,HIDDEN_DIM) f32 reference hidden sequence
  decoder_golden_recon.bin  : (SEQ_LEN,21)         f32 reference reconstruction
  decoder_golden_score.bin  : (1,)                 f32 reference MSE score

The hidden-sequence golden intentionally mimics the NPU kernel more closely than
pure PyTorch/NumPy float math:
  - inputs and weights are bf16-quantized
  - matvec outputs are rounded back to bf16
  - sigmoid/tanh intermediates are rounded through bf16
  - tanh uses the same structure as the kernel: tanh(x) = 2*sigmoid(2x)-1

Usage from npu/:
  python3 gen_decoder_data.py --seq-len 10 --window-index 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from src.models.flair_model import FLAIRAutoencoder, FLAIRConfig

LATENT_DIM = 64
HIDDEN_DIM = 64
SEQ_LEN = 10
OUTPUT_DIM = 21

_CKPT = _REPO / "experiments" / "results" / "flair_minimal.pt"
_NPZ = _REPO / "data" / "processed" / "preprocessed.npz"


def q_bf16(x):
    """Round a scalar or ndarray through bf16, then return float32 values."""
    return np.asarray(x, dtype=np.float32).astype(bfloat16).astype(np.float32)


def sigmoid_npu_like(x):
    x = q_bf16(x)
    neg_x = q_bf16(-x)
    e = q_bf16(np.exp(neg_x))
    denom = q_bf16(1.0 + e)
    return q_bf16(1.0 / denom)


def tanh_npu_like(x):
    # Mirrors kernel structure: tanh(x) = 2*sigmoid(2x)-1.
    return q_bf16(q_bf16(2.0 * sigmoid_npu_like(q_bf16(2.0 * x))) - 1.0)


def matvec_bias_npu_like(w, x, bias):
    rows = w.shape[0]
    out = np.zeros(rows, dtype=np.float32)

    for row in range(rows):
        acc = np.float32(bias[row]) if bias is not None else np.float32(0.0)
        for i in range(w.shape[1]):
            acc = np.float32(acc + np.float32(w[row, i]) * np.float32(x[i]))

        # The kernel stores matvec output as bf16.
        out[row] = q_bf16(acc)

    return out


def gru_step_golden(x, h, w_ih, w_hh, b_ih, b_hh):
    """NPU-like GRU step reference matching the kernel's bf16-heavy path."""
    H = h.shape[0]

    x = q_bf16(x)
    h = q_bf16(h)

    gi = matvec_bias_npu_like(w_ih, x, b_ih)
    gh = matvec_bias_npu_like(w_hh, h, b_hh)

    gi_r, gi_z, gi_n = gi[:H], gi[H:2 * H], gi[2 * H:]
    gh_r, gh_z, gh_n = gh[:H], gh[H:2 * H], gh[2 * H:]

    r = sigmoid_npu_like(q_bf16(gi_r + gh_r))
    z = sigmoid_npu_like(q_bf16(gi_z + gh_z))

    r_gh_n = q_bf16(r * gh_n)
    n = tanh_npu_like(q_bf16(gi_n + r_gh_n))

    one_minus_z = q_bf16(1.0 - z)
    term1 = q_bf16(one_minus_z * n)
    term2 = q_bf16(z * h)

    return q_bf16(term1 + term2)


def pad_even_bf16(arr):
    arr = arr.astype(bfloat16)
    if arr.size % 2 != 0:
        arr = np.concatenate([arr, np.zeros(1, dtype=bfloat16)])
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window-index", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = p.parse_args()

    ckpt = torch.load(str(_CKPT), map_location="cpu")
    sd = ckpt["model_state_dict"]
    cfg = FLAIRConfig(**ckpt["model_cfg"])

    model = FLAIRAutoencoder(cfg)
    model.load_state_dict(sd)
    model.eval()

    bundle = np.load(str(_NPZ), allow_pickle=True)

    x_num_np = bundle["X_num"][args.window_index : args.window_index + 1].astype(np.float32)
    x_cat_np = bundle["X_cat"][args.window_index : args.window_index + 1].astype(np.int64)

    x_num = torch.from_numpy(x_num_np)
    x_cat = torch.from_numpy(x_cat_np)

    with torch.no_grad():
        out = model(x_num, x_cat)
        latent = out["latent"].squeeze(0).numpy().astype(np.float32)

    # Quantize decoder input and weights to bf16, matching what the NPU sees.
    latent_bf = latent.astype(bfloat16)

    W_lh_bf = sd["decoder.latent_to_hidden.weight"].numpy().astype(bfloat16)
    b_lh_bf = sd["decoder.latent_to_hidden.bias"].numpy().astype(bfloat16)

    w_ih_bf = sd["decoder.gru.weight_ih_l0"].numpy().astype(bfloat16)
    w_hh_bf = sd["decoder.gru.weight_hh_l0"].numpy().astype(bfloat16)
    b_ih_bf = sd["decoder.gru.bias_ih_l0"].numpy().astype(bfloat16)
    b_hh_bf = sd["decoder.gru.bias_hh_l0"].numpy().astype(bfloat16)

    W_out_bf = sd["decoder.hidden_to_output.weight"].numpy().astype(bfloat16)
    b_out_bf = sd["decoder.hidden_to_output.bias"].numpy().astype(bfloat16)

    # Float views of bf16-quantized values.
    latent_f = latent_bf.astype(np.float32)

    W_lh = W_lh_bf.astype(np.float32)
    b_lh = b_lh_bf.astype(np.float32)

    w_ih = w_ih_bf.astype(np.float32)
    w_hh = w_hh_bf.astype(np.float32)
    b_ih = b_ih_bf.astype(np.float32)
    b_hh = b_hh_bf.astype(np.float32)

    W_out = W_out_bf.astype(np.float32)
    b_out = b_out_bf.astype(np.float32)

    # Decoder h0 is generated on the host and passed to the NPU as bf16.
    # Therefore the golden loop should use the exact bf16-rounded h0 input.
    h0 = np.tanh(W_lh @ latent_f + b_lh).astype(np.float32)
    h0 = q_bf16(h0)

    h = h0.copy()
    hidden_seq = []

    for _ in range(args.seq_len):
        h = gru_step_golden(h0, h, w_ih, w_hh, b_ih, b_hh)
        hidden_seq.append(h.copy())

    hidden_seq = np.stack(hidden_seq, axis=0).astype(np.float32)

    # hidden_to_output reference. This is still host-side reference math; the
    # current decoder NPU kernel validates hidden_seq, not this projection.
    recon = hidden_seq @ W_out.T + b_out
    recon = recon.astype(np.float32)

    x_num_window = x_num_np[0, : args.seq_len].astype(np.float32)
    score = np.array([np.mean((recon - x_num_window) ** 2)], dtype=np.float32)

    decoder_gru_params = np.concatenate([
        w_ih_bf.reshape(-1),
        w_hh_bf.reshape(-1),
        b_ih_bf.reshape(-1),
        b_hh_bf.reshape(-1),
    ]).astype(bfloat16)

    decoder_params = np.concatenate([
        W_lh_bf.reshape(-1),
        b_lh_bf.reshape(-1),
        w_ih_bf.reshape(-1),
        w_hh_bf.reshape(-1),
        b_ih_bf.reshape(-1),
        b_hh_bf.reshape(-1),
        W_out_bf.reshape(-1),
        b_out_bf.reshape(-1),
    ])
    decoder_params = pad_even_bf16(decoder_params)

    (_HERE / "decoder_latent.bin").write_bytes(latent_bf.tobytes())
    (_HERE / "decoder_h0.bin").write_bytes(h0.astype(bfloat16).tobytes())
    (_HERE / "decoder_gru_params.bin").write_bytes(decoder_gru_params.tobytes())
    (_HERE / "decoder_params.bin").write_bytes(decoder_params.tobytes())
    (_HERE / "decoder_x_num.bin").write_bytes(x_num_window.astype(bfloat16).tobytes())
    (_HERE / "decoder_hidden_golden.bin").write_bytes(hidden_seq.tobytes())
    (_HERE / "decoder_golden_recon.bin").write_bytes(recon.tobytes())
    (_HERE / "decoder_golden_score.bin").write_bytes(score.tobytes())

    print(f"latent:        {latent_bf.shape} bf16 -> decoder_latent.bin ({latent_bf.nbytes} B)")
    print(f"h0:            {h0.shape} bf16 -> decoder_h0.bin ({h0.astype(bfloat16).nbytes} B)")
    print(f"gru params:    {decoder_gru_params.shape} bf16 -> decoder_gru_params.bin ({decoder_gru_params.nbytes} B)")
    print(f"full params:   {decoder_params.shape} bf16 -> decoder_params.bin ({decoder_params.nbytes} B)")
    print(f"x_num:         {x_num_window.shape} bf16 -> decoder_x_num.bin ({x_num_window.astype(bfloat16).nbytes} B)")
    print(f"hidden_seq:    {hidden_seq.shape} f32 -> decoder_hidden_golden.bin ({hidden_seq.nbytes} B)")
    print(f"recon:         {recon.shape} f32 -> decoder_golden_recon.bin ({recon.nbytes} B)")
    print(f"MSE score:     {score[0]:.8f} -> decoder_golden_score.bin")
    print("golden:        NPU-like bf16-rounded reference")
    print(f"recon[0, :5] = {recon[0, :5]}")


if __name__ == "__main__":
    main()





#ENCODERRRRR DATA


"""
gen_encoder_data.py

Writes binary inputs + a more NPU-like golden output for the fused encoder host
harness:

  gru_state.bin  : (SEQ_LEN*INPUT_DIM,) bf16   x_window
  gru_params.bin : (N_PARAMS,)          bf16   encoder GRU [w_ih|w_hh|b_ih|b_hh]
  gru_golden.bin : (HIDDEN_DIM,)        f32    reference latent / last hidden

This golden intentionally mimics the NPU kernel more closely than a pure
PyTorch/NumPy float reference:
  - inputs and weights are bf16-quantized
  - matvec outputs are rounded back to bf16
  - sigmoid/tanh intermediates are rounded through bf16
  - tanh uses the same structure as the kernel: tanh(x) = 2*sigmoid(2x)-1

Usage from npu/:
  python3 gen_encoder_data.py --seq-len 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HIDDEN_DIM = 64
SEQ_LEN = 10

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_CKPT = _REPO / "experiments" / "results" / "flair_minimal.pt"
_NPZ = _REPO / "data" / "processed" / "preprocessed.npz"


def q_bf16(x):
    """Round a scalar or ndarray through bf16, then return float32 values."""
    return np.asarray(x, dtype=np.float32).astype(bfloat16).astype(np.float32)


def sigmoid_npu_like(x):
    x = q_bf16(x)
    neg_x = q_bf16(-x)
    e = q_bf16(np.exp(neg_x))
    denom = q_bf16(1.0 + e)
    return q_bf16(1.0 / denom)


def tanh_npu_like(x):
    # Mirrors kernel structure: tanh(x) = 2*sigmoid(2x)-1.
    return q_bf16(q_bf16(2.0 * sigmoid_npu_like(q_bf16(2.0 * x))) - 1.0)


def matvec_bias_npu_like(w, x, bias):
    rows = w.shape[0]
    out = np.zeros(rows, dtype=np.float32)

    for row in range(rows):
        acc = np.float32(bias[row]) if bias is not None else np.float32(0.0)
        for i in range(w.shape[1]):
            acc = np.float32(acc + np.float32(w[row, i]) * np.float32(x[i]))

        # The kernel stores matvec output as bf16.
        out[row] = q_bf16(acc)

    return out


def gru_step_golden(x, h, w_ih, w_hh, b_ih, b_hh):
    """NPU-like GRU step reference matching the kernel's bf16-heavy path."""
    H = h.shape[0]

    x = q_bf16(x)
    h = q_bf16(h)

    gi = matvec_bias_npu_like(w_ih, x, b_ih)
    gh = matvec_bias_npu_like(w_hh, h, b_hh)

    gi_r, gi_z, gi_n = gi[:H], gi[H:2 * H], gi[2 * H:]
    gh_r, gh_z, gh_n = gh[:H], gh[H:2 * H], gh[2 * H:]

    r = sigmoid_npu_like(q_bf16(gi_r + gh_r))
    z = sigmoid_npu_like(q_bf16(gi_z + gh_z))

    r_gh_n = q_bf16(r * gh_n)
    n = tanh_npu_like(q_bf16(gi_n + r_gh_n))

    one_minus_z = q_bf16(1.0 - z)
    term1 = q_bf16(one_minus_z * n)
    term2 = q_bf16(z * h)

    return q_bf16(term1 + term2)


def main() -> None:
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--window-index", type=int, default=0)
    args = p.parse_args()

    ckpt = torch.load(str(_CKPT), map_location="cpu")
    sd = ckpt["model_state_dict"]

    w_ih_bf = sd["encoder.gru.weight_ih_l0"].numpy().astype(bfloat16)
    w_hh_bf = sd["encoder.gru.weight_hh_l0"].numpy().astype(bfloat16)
    b_ih_bf = sd["encoder.gru.bias_ih_l0"].numpy().astype(bfloat16)
    b_hh_bf = sd["encoder.gru.bias_hh_l0"].numpy().astype(bfloat16)

    input_dim = int(w_ih_bf.shape[1])
    hidden_dim = int(w_hh_bf.shape[1])
    seq_len = int(args.seq_len)

    params = np.concatenate(
        [w_ih_bf.reshape(-1), w_hh_bf.reshape(-1), b_ih_bf, b_hh_bf]
    ).astype(bfloat16)

    bundle = np.load(str(_NPZ), allow_pickle=True)
    x_num = bundle["X_num"][args.window_index].astype(np.float32)
    x_cat = bundle["X_cat"][args.window_index].astype(np.int64)

    sport_w = sd["sport_emb.weight"].numpy().astype(np.float32)
    dport_w = sd["dport_emb.weight"].numpy().astype(np.float32)
    proto_w = sd["proto_emb.weight"].numpy().astype(np.float32)

    x_in_steps = []
    for t in range(seq_len):
        xin = np.concatenate([
            x_num[t],
            sport_w[x_cat[t, 0]],
            dport_w[x_cat[t, 1]],
            proto_w[x_cat[t, 2]],
        ]).astype(bfloat16)

        if xin.size != input_dim:
            raise ValueError(
                f"Combined encoder input has {xin.size} values, but checkpoint expects {input_dim}."
            )

        x_in_steps.append(xin)

    x_window = np.concatenate(x_in_steps).astype(bfloat16)

    # NPU-like golden latent from bf16-quantized inputs/weights.
    w_ih = w_ih_bf.astype(np.float32)
    w_hh = w_hh_bf.astype(np.float32)
    b_ih = b_ih_bf.astype(np.float32)
    b_hh = b_hh_bf.astype(np.float32)

    h = np.zeros(hidden_dim, dtype=np.float32)
    for t in range(seq_len):
        h = gru_step_golden(x_in_steps[t].astype(np.float32), h, w_ih, w_hh, b_ih, b_hh)

    latent = h.astype(np.float32)

    (_HERE / "gru_state.bin").write_bytes(x_window.tobytes())
    (_HERE / "gru_params.bin").write_bytes(params.tobytes())
    (_HERE / "gru_golden.bin").write_bytes(latent.tobytes())

    print(f"x_window: {x_window.shape} bf16 -> gru_state.bin ({x_window.nbytes} B)")
    print(f"params:   {params.shape} bf16 -> gru_params.bin ({params.nbytes} B)")
    print(f"latent:   {latent.shape} f32  -> gru_golden.bin ({latent.nbytes} B)")
    print("golden:   NPU-like bf16-rounded reference")
    print(f"latent[:8] = {latent[:8]}")


if __name__ == "__main__":
    main()










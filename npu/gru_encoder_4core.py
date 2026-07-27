#
# Authors: Daniel Eyraud, Mary-Claire Ridgeway
#
# Four-core data-parallel IRON wrapper for the FLAIR encoder. Each compute
# tile runs gru_encoder_bf16 on an independent batch slice; the memtile
# scatters input windows, broadcasts shared parameters, and gathers latents.
#
# Buffers:
#   x_windows : (4 * BATCH * SEQ_LEN * INPUT_DIM) bf16 input windows
#   params    : encoder GRU weights [w_ih | w_hh | b_ih | b_hh], shared
#   latents   : (4 * BATCH * HIDDEN_DIM) bf16 final hidden states
#
# BATCH is windows per compute tile, so one dispatch processes 4 * BATCH
# independent windows.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.utils import config
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli

_KERNELS_DIR = Path(__file__).parent / "kernels"
_KERNEL_SRC = _KERNELS_DIR / "gru_encoder.cc"

INPUT_DIM = 48
HIDDEN_DIM = 64
SEQ_LEN = 10
BATCH = 1        # windows PER CORE per dispatch
N_CORES = 4


def _make_encoder_kernel(arg_types, compile_flags):
    """Build gru_encoder_bf16 with the AIE runtime LUT implementation."""
    header_base = Path(config.cxx_header_path())
    runtime_dir = Path(config.root_path()) / "aie_runtime_lib" / "AIE2"
    lut_cpp = runtime_dir / "lut_based_ops.cpp"

    include_dirs = [
        str(header_base),
        str(header_base / "aie_kernels"),
        str(runtime_dir),
        str(_KERNELS_DIR),
    ]
    source = f'#include "{_KERNEL_SRC}"\n#include "{lut_cpp}"\n'

    return ExternalFunction(
        "gru_encoder_bf16",
        source_string=source,
        arg_types=arg_types,
        include_dirs=include_dirs,
        compile_flags=compile_flags,
    )


@iron.jit
def gru_encoder_4core(
    x_windows: In,
    params: In,
    latents: Out,
    *,
    input_dim: CompileTime[int] = INPUT_DIM,
    hidden_dim: CompileTime[int] = HIDDEN_DIM,
    seq_len: CompileTime[int] = SEQ_LEN,
    batch: CompileTime[int] = BATCH,
):
    h3 = 3 * hidden_dim
    
    # params layout:
    #   w_ih (H3*INPUT_DIM) | w_hh (H3*H) | b_ih (H3) | b_hh (H3)
    n_params = h3 * input_dim + h3 * hidden_dim + h3 + h3
    
    # The host transfers one contiguous input and output buffer. The memtile
    # splits and joins fixed-size slices for the four compute tiles.
    per_core_win = batch * seq_len * input_dim   # x_window for one core's batch
    per_core_lat = batch * hidden_dim            # latent for one core's batch
    total_win = N_CORES * per_core_win
    total_lat = N_CORES * per_core_lat

    dtype = np.dtype[bfloat16]
    
    # Host-visible types span all tiles; core types describe one tile's slice.
    x_all_ty = np.ndarray[(total_win,), dtype]        # shim<->memtile (all cores)
    x_core_ty = np.ndarray[(per_core_win,), dtype]    # memtile->one core
    params_ty = np.ndarray[(n_params,), dtype]
    lat_all_ty = np.ndarray[(total_lat,), dtype]      # memtile<->shim (all cores)
    lat_core_ty = np.ndarray[(per_core_lat,), dtype]  # one core->memtile

    # Every tile runs the same encoder kernel, compiled for batch windows.
    kernel = _make_encoder_kernel(
        arg_types=[x_core_ty, params_ty, lat_core_ty],
        compile_flags=[
            f"-DINPUT_DIM={input_dim}",
            f"-DHIDDEN_DIM={hidden_dim}",
            f"-DSEQ_LEN={seq_len}",
            f"-DBATCH={batch}",
        ],
    )

    # Scatter contiguous window slices from the shim through the memtile.
    x_fifo = ObjectFifo(x_all_ty, name="x_windows")
    x_offsets = [per_core_win * i for i in range(N_CORES)]
    x_core_fifos = x_fifo.cons().split(
        x_offsets,
        obj_types=[x_core_ty] * N_CORES,
        names=[f"x_core{i}" for i in range(N_CORES)],
    )

    # Broadcast one parameter buffer to all four tiles. At the default
    # dimensions, params occupies 43,776 bytes; depth 2 would allocate two
    # copies in each tile's 64 KB L1. The weights are read-only and reused
    # across the tile batch, so double-buffering provides no prefetch benefit.
    params_fifo = ObjectFifo(params_ty, depth=1, name="params")
    params_bcast = params_fifo.cons().forward(
        obj_type=params_ty, depth=1, name="params_bcast"
    )

    # Gather one latent slice from each tile into the host-visible output.
    lat_fifo = ObjectFifo(lat_all_ty, name="latents")
    lat_offsets = [per_core_lat * i for i in range(N_CORES)]
    lat_core_fifos = lat_fifo.prod().join(
        lat_offsets,
        obj_types=[lat_core_ty] * N_CORES,
        names=[f"lat_core{i}" for i in range(N_CORES)],
    )

    def core_fn(x_c, params_c, lat_p, k):
        # Hold one element from each FIFO for the duration of the kernel call.
        ew = x_c.acquire(1)
        ep = params_c.acquire(1)
        el = lat_p.acquire(1)
        k(ew, ep, el)
        x_c.release(1)
        params_c.release(1)
        lat_p.release(1)

    workers = []
    for i in range(N_CORES):
        workers.append(
            Worker(
                core_fn,
                [
                    x_core_fifos[i].cons(),
                    params_bcast.cons(),   # same fifo -> broadcast to every core
                    lat_core_fifos[i].prod(),
                    kernel,
                ],
            )
        )

    rt = Runtime()
    with rt.sequence(x_all_ty, params_ty, lat_all_ty) as (x_arg, params_arg, lat_arg):
        rt.start(*workers)
        rt.fill(x_fifo.prod(), x_arg)
        rt.fill(params_fifo.prod(), params_arg)
        rt.drain(lat_fifo.cons(), lat_arg, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="FLAIR GRU encoder (4-core data-parallel)")
    add_compile_args(p)
    p.add_argument("--input-dim", type=int, default=INPUT_DIM)
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--batch", type=int, default=BATCH,
                   help="windows PER CORE per dispatch (total = 4*batch)")
    return p


def _compile_kwargs(opts):
    return dict(
        input_dim=opts.input_dim,
        hidden_dim=opts.hidden_dim,
        seq_len=opts.seq_len,
        batch=opts.batch,
    )


def _run_and_verify(opts):
    """Reject local execution; this design is run through the Windows host."""
    raise SystemExit(
        "Compile-only design (WSL). Run via batch_infer.exe on Windows."
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        gru_encoder_4core,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        # One column = 1 shim + 1 memtile + 4 compute tiles -- exactly what the
        # 4-core data-parallel design needs.
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()

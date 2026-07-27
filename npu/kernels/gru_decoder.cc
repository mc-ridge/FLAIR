/*
Authors: Mary-Claire Ridgeway, Daniel Eyraud

FLAIR decoder GRU kernels (unfused, fused, and diagnostic variants)

Constraints: 
* Batch is the number of windows processed by one compute tile
* The decoder input is the same at every timestep
* GRU timesteps remain sequential 

Diagnostic variants are used to isolate decoder overhead:
* final_bf16         - full GRU compute, but only writes the final hidden state
* noop_bf16          - preserves data movement with almost no computation
* matvec_only_bf16   - measures recurrent matvec cost without gate nonlinearities
These are for performance analysis only and are not used in final scoring

*/

#include <aie_api/aie.hpp>
#include "aie_kernel_utils.h"
#include "lut_based_ops.h"
#include "gru_common.h"

#ifndef HIDDEN_DIM
#define HIDDEN_DIM 64
#endif

#ifndef SEQ_LEN
#define SEQ_LEN 10
#endif

#ifndef BATCH
#define BATCH 1
#endif

// Unfused decoder. Returns the hidden state from every timestep
// Parameter layout: [w_ih | w_hh | b_ih | b_hh]
extern "C" void gru_decoder_bf16(
    bfloat16 *h0_vec,
    bfloat16 *params,
    bfloat16 *hidden_seq
) {
    constexpr int H = HIDDEN_DIM;
    constexpr int H3 = 3 * H;

    // Decoder input_dim is HIDDEN_DIM because x_t = h0_vec
    constexpr int INPUT_DIM = HIDDEN_DIM;

    bfloat16 *w_ih = params;
    bfloat16 *w_hh = w_ih + H3 * INPUT_DIM;
    bfloat16 *b_ih = w_hh + H3 * H;
    bfloat16 *b_hh = b_ih + H3;

    for (int b = 0; b < BATCH; b++) {
        bfloat16 *h0_vec_b = h0_vec + b * H;
        bfloat16 *hidden_seq_b = hidden_seq + b * SEQ_LEN * H;

        // gru_step_with_gi vector-loads and stores this local hidden state
        alignas(aie::vector_decl_align) bfloat16 h[H];

        for (int i = 0; i < H; i++) {
            h[i] = h0_vec_b[i];
        }

        // x_t is constant, so w_ih @ x_t + b_ih is computed once per window
        alignas(aie::vector_decl_align) bfloat16 gi[H3];
        flair::matvec_bias(w_ih, h0_vec_b, b_ih, gi, H3, INPUT_DIM);

        // Full decoder GRU sequence.
        for (int t = 0; t < SEQ_LEN; t++) {
            flair::gru_step_with_gi(gi, h, w_hh, b_hh);

            // Save h_t into hidden_seq_b[t].
            for (int i = 0; i < H; i++) {
                hidden_seq_b[t * H + i] = h[i];
            }
        }
    }
}

#ifndef OUTPUT_DIM
#define OUTPUT_DIM 21
#endif

// Fused decoder. Applies hidden_to_output on the compute tile and returns the
// reconstructed numeric sequence.
//
// Parameter layout: [w_ih | w_hh | b_ih | b_hh | w_out | b_out]
extern "C" void gru_decoder_fused_bf16(
    bfloat16 *h0_vec,
    bfloat16 *params,
    bfloat16 *recon
) {
    constexpr int H = HIDDEN_DIM;
    constexpr int H3 = 3 * H;
    constexpr int INPUT_DIM = HIDDEN_DIM;
    constexpr int OUT = OUTPUT_DIM;

    bfloat16 *w_ih = params;
    bfloat16 *w_hh = w_ih + H3 * INPUT_DIM;
    bfloat16 *b_ih = w_hh + H3 * H;
    bfloat16 *b_hh = b_ih + H3;
    bfloat16 *w_out = b_hh + H3;        // (OUT, H)
    bfloat16 *b_out = w_out + OUT * H;  // (OUT,)

    for (int b = 0; b < BATCH; b++) {
        bfloat16 *h0_vec_b = h0_vec + b * H;
        bfloat16 *recon_b = recon + b * SEQ_LEN * OUT;

        alignas(aie::vector_decl_align) bfloat16 h[H];
        for (int i = 0; i < H; i++) {
            h[i] = h0_vec_b[i];
        }

        for (int t = 0; t < SEQ_LEN; t++) {
            flair::gru_step(
                h0_vec_b,
                h,
                w_ih,
                w_hh,
                b_ih,
                b_hh,
                INPUT_DIM
            );

            // recon[t] = w_out @ h_t + b_out
            flair::matvec_bias(w_out, h, b_out, recon_b + t * OUT, OUT, H);
        }
    }
}

// Diagnostic variant
// Runs the full decoder but returns only the final hidden state
// to isolate output-buffer and DMA cost
extern "C" void gru_decoder_final_bf16(
    bfloat16 *h0_vec,
    bfloat16 *params,
    bfloat16 *final_h
) {
    constexpr int H = HIDDEN_DIM;
    constexpr int H3 = 3 * H;
    constexpr int INPUT_DIM = HIDDEN_DIM;

    bfloat16 *w_ih = params;
    bfloat16 *w_hh = w_ih + H3 * INPUT_DIM;
    bfloat16 *b_ih = w_hh + H3 * H;
    bfloat16 *b_hh = b_ih + H3;

    for (int b = 0; b < BATCH; b++) {
        bfloat16 *h0_vec_b = h0_vec + b * H;
        bfloat16 *final_h_b = final_h + b * H;

        alignas(aie::vector_decl_align) bfloat16 h[H];
        for (int i = 0; i < H; i++) {
            h[i] = h0_vec_b[i];
        }

        // Same gi-hoisting as gru_decoder_bf16 
        alignas(aie::vector_decl_align) bfloat16 gi[H3];
        flair::matvec_bias(w_ih, h0_vec_b, b_ih, gi, H3, INPUT_DIM);

        for (int t = 0; t < SEQ_LEN; t++) {
            flair::gru_step_with_gi(gi, h, w_hh, b_hh);
        }

        // Write ONLY the final hidden state (batch*H total).
        for (int i = 0; i < H; i++) {
            final_h_b[i] = h[i];
        }
    }
}

// Diagnostic variant
// Preserves the normal buffer flow without running GRU computation 
// to measure fixed dispatch and data-movement overhead
extern "C" void gru_decoder_noop_bf16(
    bfloat16 *h0_vec,
    bfloat16 *params,
    bfloat16 *hidden_seq
) {
    constexpr int H = HIDDEN_DIM;

    for (int b = 0; b < BATCH; b++) {
        bfloat16 *h0_vec_b = h0_vec + b * H;
        bfloat16 *hidden_seq_b = hidden_seq + b * SEQ_LEN * H;

        // Touch params (one element) so it isn't compiled away entirely,
        // without doing any real matvec/gate compute.
        bfloat16 touch = params[0];

        for (int i = 0; i < H; i++) {
            hidden_seq_b[i] = h0_vec_b[i] + touch - touch; // == h0_vec_b[i]
        }
        for (int t = 1; t < SEQ_LEN; t++) {
            for (int i = 0; i < H; i++) {
                hidden_seq_b[t * H + i] = (bfloat16)0.0f;
            }
        }
    }
}

// Diagnostic variant
// Runs the recurrent matrix multiplication but skips the sigmoid,
// tanh, and gate-combination calculations
extern "C" void gru_decoder_matvec_only_bf16(
    bfloat16 *h0_vec,
    bfloat16 *params,
    bfloat16 *hidden_seq
) {
    constexpr int H = HIDDEN_DIM;
    constexpr int H3 = 3 * H;
    constexpr int INPUT_DIM = HIDDEN_DIM;

    bfloat16 *w_ih = params;
    bfloat16 *w_hh = w_ih + H3 * INPUT_DIM;
    bfloat16 *b_ih = w_hh + H3 * H;
    bfloat16 *b_hh = b_ih + H3;

    for (int b = 0; b < BATCH; b++) {
        bfloat16 *h0_vec_b = h0_vec + b * H;
        bfloat16 *hidden_seq_b = hidden_seq + b * SEQ_LEN * H;

        alignas(aie::vector_decl_align) bfloat16 h[H];
        for (int i = 0; i < H; i++) {
            h[i] = h0_vec_b[i];
        }

        (void)w_ih; (void)b_ih; // unused here (matches gru_step_with_gi's shape)

        for (int t = 0; t < SEQ_LEN; t++) {
            alignas(aie::vector_decl_align) bfloat16 gh[H3];
            flair::matvec_bias(w_hh, h, b_hh, gh, H3, H); // same matvec as gru_step_with_gi

            // No sigmoid/tanh at all -- just take gh's first H elements as
            // the "new" h (garbage values, but same memory traffic shape).
            for (int i = 0; i < H; i++) {
                h[i] = gh[i];
            }

            for (int i = 0; i < H; i++) {
                hidden_seq_b[t * H + i] = h[i];
            }
        }
    }
}

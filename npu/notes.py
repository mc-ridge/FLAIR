#Add these helpers to both files
def q_bf16(x):
    return np.asarray(x, dtype=np.float32).astype(bfloat16).astype(np.float32)


def sigmoid_npu_like(x):
    x = q_bf16(x)
    neg_x = q_bf16(-x)
    e = q_bf16(np.exp(neg_x))
    denom = q_bf16(1.0 + e)
    return q_bf16(1.0 / denom)


def tanh_npu_like(x):
    # Matches kernel structure: tanh(x) = 2*sigmoid(2x)-1
    return q_bf16(q_bf16(2.0 * sigmoid_npu_like(q_bf16(2.0 * x))) - 1.0)


def matvec_bias_npu_like(w, x, bias):
    rows = w.shape[0]
    out = np.zeros(rows, dtype=np.float32)

    for row in range(rows):
        acc = np.float32(bias[row]) if bias is not None else np.float32(0.0)
        for i in range(w.shape[1]):
            acc = np.float32(acc + np.float32(w[row, i]) * np.float32(x[i]))

        # Kernel stores matvec output as bf16.
        out[row] = q_bf16(acc)

    return out
#Replace gru_step_golden with this
def gru_step_golden(x, h, w_ih, w_hh, b_ih, b_hh):
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
#One decoder-specific detail

#In gen_decoder_data.py, after computing h0, force it through bf16 before using it in the golden loop:

h0 = np.tanh(W_lh @ latent_f + b_lh).astype(np.float32)
h0 = q_bf16(h0)

#Then the decoder loop stays:

h = h0.copy()
hidden_seq = []

for _ in range(args.seq_len):
    h = gru_step_golden(h0, h, w_ih, w_hh, b_ih, b_hh)
    hidden_seq.append(h.copy())



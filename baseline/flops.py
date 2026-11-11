"""
Analytic FLOP count (1 multiply-accumulate = 2 FLOPs) for one training step of the model,
broken down per block, plus achieved GFLOP/s from a measured step time.

    python baseline/flops.py --config small --step-ms 60.0
Note: forward() processes T = seq_len + 1 positions per row (the whole batch row).
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import CONFIGS

def count(cfg, V):
    B, T, D, H, F, L = cfg["batch_size"], cfg["seq_len"] + 1, cfg["d_model"], cfg["n_heads"], cfg["d_ff"], cfg["n_layers"]
    N, dh = B * T, D // H
    mac = {}
    mac["qkv projections"]   = L * 3 * N * D * D
    mac["attention scores QK^T"] = L * B * H * T * T * dh
    mac["attention x V"]     = L * B * H * T * T * dh
    mac["output projection"] = L * N * D * D
    mac["feed-forward W1"]   = L * N * D * F
    mac["feed-forward W2"]   = L * N * F * D
    mac["vocab projection"]  = N * D * V
    fwd = {k: 2 * v for k, v in mac.items()}
    fwd_total = sum(fwd.values())
    # Backward does two matrix products per forward product (dX and dW), so about 2x forward matmul FLOPs.
    bwd_total = 2 * fwd_total
    return fwd, fwd_total, bwd_total, N

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="small", choices=CONFIGS)
    ap.add_argument("--vocab", type=int, default=54, help="vocab size printed by train.py (len of set of characters)")
    ap.add_argument("--step-ms", type=float, default=None, help="measured median train step in ms")
    a = ap.parse_args()
    cfg = CONFIGS[a.config]
    fwd, ft, bt, N = count(cfg, a.vocab)
    print(f"config={a.config}  positions per step N={N}")
    for k, v in fwd.items():
        print(f"  forward  {k:<24s} {v/1e6:10.2f} MFLOP  ({100*v/ft:5.1f}% of forward)")
    print(f"  forward total {ft/1e6:.2f} MFLOP | backward (approx 2x) {bt/1e6:.2f} MFLOP | train step {(ft+bt)/1e6:.2f} MFLOP")
    if a.step_ms:
        print(f"  achieved: {(ft+bt)/(a.step_ms*1e-3)/1e9:.3f} GFLOP/s over the whole train step (matmul FLOPs only)")

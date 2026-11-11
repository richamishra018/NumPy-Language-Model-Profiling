"""
Python baseline benchmark for the unmodified NumPy transformer.
Times each phase of a training step separately and the inference forward pass.

Usage (from the repo root):
    python baseline/bench.py --config small --reps 30 --warmup 3 --out baseline/results_small.json
"""
import argparse, json, os, platform, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from transformer import (CharTokenizer, init_params, forward, cross_entropy_loss,
                         backward, AdamOptimizer, get_batch)

CONFIGS = {
    "small":  dict(d_model=64,  n_heads=4, d_ff=256,  n_layers=2, seq_len=64,  batch_size=8,  lr=3e-3),
    "medium": dict(d_model=128, n_heads=4, d_ff=512,  n_layers=4, seq_len=128, batch_size=16, lr=1e-3),
    "large":  dict(d_model=256, n_heads=8, d_ff=1024, n_layers=6, seq_len=256, batch_size=8,  lr=5e-4),
}

def peak_rss_mb():
    """Peak resident memory of this process in MB, on Linux, macOS and Windows."""
    try:
        import resource                                          # Linux and macOS
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # KB on Linux, bytes on macOS
        return r / (1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0)
    except ImportError:                                          # Windows has no resource module
        import psutil                                            # pip install psutil
        return psutil.Process().memory_info().peak_wset / (1024.0 * 1024.0)

def median(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="small", choices=CONFIGS)
    ap.add_argument("--text", default="data/sample.txt")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = CONFIGS[a.config].copy(); cfg["seed"] = a.seed
    text = open(a.text, encoding="utf-8").read()
    tok = CharTokenizer(text)
    data = np.array(tok.encode(text), dtype=np.int32)
    cfg["vocab_size"] = tok.vocab_size
    params = init_params(tok.vocab_size, cfg["seq_len"], cfg["d_model"], cfg["n_heads"],
                         cfg["d_ff"], cfg["n_layers"], seed=a.seed)
    opt = AdamOptimizer(params, lr=cfg["lr"])
    rng = np.random.default_rng(a.seed)
    batch = get_batch(data, cfg["batch_size"], cfg["seq_len"], rng)   # fixed batch: timing, not learning
    tokens_per_step = batch.shape[0] * batch.shape[1]                  # forward() sees seq_len + 1 columns

    keys = ["forward", "loss", "backward", "adam", "train_step", "inference_forward"]
    t = {k: [] for k in keys}
    for i in range(-a.warmup, a.reps):
        s0 = time.perf_counter()
        logits, caches = forward(batch, params, cfg)
        s1 = time.perf_counter()
        loss, dlogits = cross_entropy_loss(logits, batch)
        s2 = time.perf_counter()
        grads = backward(dlogits, caches, params, cfg)
        s3 = time.perf_counter()
        params = opt.step(params, grads)
        s4 = time.perf_counter()
        _ , _ = forward(batch, params, cfg)                            # inference-style forward only
        s5 = time.perf_counter()
        if i >= 0:
            for k, v in zip(keys, [s1-s0, s2-s1, s3-s2, s4-s3, s4-s0, s5-s4]):
                t[k].append(v)

    res = {
        "config": a.config, "model": {k: cfg[k] for k in ("d_model","n_heads","d_ff","n_layers","seq_len","batch_size","vocab_size")},
        "tokens_per_step": tokens_per_step, "reps": a.reps, "warmup": a.warmup,
        "python": sys.version.split()[0], "numpy": np.__version__, "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "median_ms": {k: 1e3 * median(v) for k, v in t.items()},
        "min_ms":    {k: 1e3 * min(v) for k, v in t.items()},
        "max_ms":    {k: 1e3 * max(v) for k, v in t.items()},
        "train_tokens_per_sec": tokens_per_step / median(t["train_step"]),
        "inference_tokens_per_sec": tokens_per_step / median(t["inference_forward"]),
        "peak_rss_mb": peak_rss_mb(),
    }
    ts = res["median_ms"]["train_step"]
    res["share_of_train_step_pct"] = {k: 100.0 * res["median_ms"][k] / ts for k in ("forward","loss","backward","adam")}
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f: json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()

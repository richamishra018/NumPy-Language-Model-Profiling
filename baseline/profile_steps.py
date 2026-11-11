"""
Run N training steps of the unmodified model with a fixed seed. This is the target
script for cProfile, line_profiler, py-spy, memory_profiler and VTune, so every
tool profiles exactly the same work (and not matplotlib, argument parsing or plotting).

    python baseline/profile_steps.py --config small --steps 30
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from transformer import (CharTokenizer, init_params, forward, cross_entropy_loss,
                         backward, AdamOptimizer, get_batch)
from bench import CONFIGS

def run(config="small", steps=30, text_path="data/sample.txt", seed=42):
    cfg = CONFIGS[config].copy(); cfg["seed"] = seed
    text = open(text_path, encoding="utf-8").read()
    tok = CharTokenizer(text)
    data = np.array(tok.encode(text), dtype=np.int32)
    cfg["vocab_size"] = tok.vocab_size
    params = init_params(tok.vocab_size, cfg["seq_len"], cfg["d_model"], cfg["n_heads"],
                         cfg["d_ff"], cfg["n_layers"], seed=seed)
    opt = AdamOptimizer(params, lr=cfg["lr"])
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        batch = get_batch(data, cfg["batch_size"], cfg["seq_len"], rng)
        logits, caches = forward(batch, params, cfg)
        loss, dlogits = cross_entropy_loss(logits, batch)
        grads = backward(dlogits, caches, params, cfg)
        params = opt.step(params, grads)
    return loss

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="small", choices=CONFIGS)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--text", default="data/sample.txt")
    a = ap.parse_args()
    print("final loss", run(a.config, a.steps, a.text))

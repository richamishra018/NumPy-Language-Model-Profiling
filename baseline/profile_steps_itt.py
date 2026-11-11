"""
Same work as profile_steps.py, with Intel ITT task markers so VTune can report time per phase
(forward, loss, backward, adam). Without VTune attached the markers do nothing.

    vtune -collect hotspots -start-paused -knob sampling-mode=sw -result-dir vtune/py_itt -- python baseline/profile_steps_itt.py --steps 100
"""
import argparse, os, sys
import numpy as np
import ittapi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    ittapi.resume()                                    # with -start-paused, collection begins here
    for _ in range(steps):
        batch = get_batch(data, cfg["batch_size"], cfg["seq_len"], rng)
        with ittapi.task("forward"):
            logits, caches = forward(batch, params, cfg)
        with ittapi.task("loss"):
            loss, dlogits = cross_entropy_loss(logits, batch)
        with ittapi.task("backward"):
            grads = backward(dlogits, caches, params, cfg)
        with ittapi.task("adam"):
            params = opt.step(params, grads)
    ittapi.pause()                                     # collection stops here
    return loss

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="small", choices=CONFIGS)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--text", default="data/sample.txt")
    a = ap.parse_args()
    print("final loss", run(a.config, a.steps, a.text))

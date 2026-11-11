"""
Export a reference snapshot from the unmodified NumPy model so the C port can be
checked against it. Writes into ./c/ref/ :
  model.cfg    : vocab seq_len d_model n_heads d_ff n_layers batch steps
  params0.bin  : initial parameters, float64, in init_params() insertion order
  batches.bin  : int32 token batches, shape (steps, batch, seq_len+1)
  logits0.bin  : float64 logits for batch 0 with params0
  grads0.bin   : float64 gradients for batch 0, same layout as params0.bin
  losses.bin   : float64 loss for every step of the NumPy training loop
Usage: python baseline/export_ref.py --config small --steps 20
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from transformer import (CharTokenizer, init_params, forward, cross_entropy_loss,
                         backward, AdamOptimizer, get_batch)

CONFIGS = {
    "small":  dict(d_model=64,  n_heads=4, d_ff=256,  n_layers=2, seq_len=64,  batch_size=8,  lr=3e-3),
    "medium": dict(d_model=128, n_heads=4, d_ff=512,  n_layers=4, seq_len=128, batch_size=16, lr=1e-3),
    "large":  dict(d_model=256, n_heads=8, d_ff=1024, n_layers=6, seq_len=256, batch_size=8,  lr=5e-4),
}

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="small", choices=CONFIGS)
ap.add_argument("--text", default="data/sample.txt")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="c/ref")
a = ap.parse_args()

cfg = CONFIGS[a.config].copy(); cfg["seed"] = a.seed
text = open(a.text, encoding="utf-8").read()
tok = CharTokenizer(text)
data = np.array(tok.encode(text), dtype=np.int32)
cfg["vocab_size"] = tok.vocab_size
params = init_params(tok.vocab_size, cfg["seq_len"], cfg["d_model"], cfg["n_heads"],
                     cfg["d_ff"], cfg["n_layers"], seed=a.seed)
os.makedirs(a.out, exist_ok=True)

with open(f"{a.out}/model.cfg", "w") as f:
    f.write(f'{tok.vocab_size} {cfg["seq_len"]} {cfg["d_model"]} {cfg["n_heads"]} '
            f'{cfg["d_ff"]} {cfg["n_layers"]} {cfg["batch_size"]} {a.steps} {cfg["lr"]}\n')

np.concatenate([v.ravel() for v in params.values()]).astype(np.float64).tofile(f"{a.out}/params0.bin")

rng = np.random.default_rng(a.seed)
batches = np.stack([get_batch(data, cfg["batch_size"], cfg["seq_len"], rng) for _ in range(a.steps)]).astype(np.int32)
batches.tofile(f"{a.out}/batches.bin")

# Step-0 reference: logits and gradients with the initial parameters
logits, caches = forward(batches[0], params, cfg)
loss0, dlogits = cross_entropy_loss(logits, batches[0])
grads = backward(dlogits, caches, params, cfg)
logits.astype(np.float64).tofile(f"{a.out}/logits0.bin")
np.concatenate([grads[k].ravel() for k in params]).astype(np.float64).tofile(f"{a.out}/grads0.bin")

# Full NumPy training trajectory on the same batches
opt = AdamOptimizer(params, lr=cfg["lr"])
losses = []
for s in range(a.steps):
    lg, ca = forward(batches[s], params, cfg)
    l, dl = cross_entropy_loss(lg, batches[s])
    g = backward(dl, ca, params, cfg)
    params = opt.step(params, g)
    losses.append(l)
np.array(losses, dtype=np.float64).tofile(f"{a.out}/losses.bin")
print("exported", a.config, "vocab", tok.vocab_size, "step0 loss", float(loss0), "last loss", losses[-1])

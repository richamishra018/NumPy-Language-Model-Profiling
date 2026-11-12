"""Group per-call-site dgemm timings (from c/dgemm_sites.so) into the candidate kernels of the Stage 6 ranking.
Usage (repo root):  python3 baseline/dgemm_sites_report.py docs/logs/dgemm_sites_ref.raw c/ref c/tx
Prints a table: kernel, calls, time share of all dgemm, GFLOP/s.  Call sites are mapped to tx.c lines with addr2line."""
import subprocess, sys, collections
raw, refdir, binary = sys.argv[1:4]
V, S, D, H, F, L, B, steps, lr = open(refdir + "/model.cfg").read().split()
V, D, F = int(V), int(D), int(F)

def frames(off):
    out = subprocess.run(["addr2line", "-i", "-f", "-e", binary, hex(off - 1)], capture_output=True, text=True).stdout.splitlines()
    return [(out[i], int(out[i + 1].rsplit(":", 1)[1].split()[0])) for i in range(0, len(out) - 1, 2)]

def classify(fr, M, N, K):
    fn = [f for f, _ in fr]; line = fr[0][1]
    if "linear" in fn:                         # forward linear(x,W,b,y,N,Din,Dout): gemm M=N, N=Dout, K=Din
        if N == V: return "vocab projection (fwd)"
        if (N, K) in ((F, D), (D, F)): return "feed-forward W1/W2 (fwd)"
        return "QKV + output projection (fwd)"
    if line in (144, 152): return "attention scores/context per head (fwd)"
    if line in (196, 198): return "vocab projection (bwd)"
    if line in (204, 205, 208, 210): return "feed-forward W1/W2 (bwd)"
    if line in (215, 216, 231, 232, 233, 234, 235, 236): return "QKV + output projection (bwd)"
    if line in (221, 222, 228, 229): return "attention scores/context per head (bwd)"
    return "other (line %d)" % line

agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
for l in open(raw):
    p = l.split()
    if p[0] != "SITE": continue
    off, M, N, K, calls, ns = int(p[1], 16), int(p[2]), int(p[3]), int(p[4]), int(p[5]), float(p[6])
    c = classify(frames(off), M, N, K)
    a = agg[c]; a[0] += calls; a[1] += ns; a[2] += 2.0 * M * N * K * calls
tot = sum(a[1] for a in agg.values())
print(f"{'kernel':45s} {'calls':>7s} {'% of dgemm':>10s} {'GFLOP/s':>8s}")
fam = collections.defaultdict(lambda: [0, 0.0, 0.0])
for c, a in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print(f"{c:45s} {a[0]:7d} {100*a[1]/tot:9.1f}% {a[2]/a[1]:8.2f}")
    k = c.rsplit(" (", 1)[0]; fam[k][0] += a[0]; fam[k][1] += a[1]; fam[k][2] += a[2]
print("--- fwd+bwd combined ---")
for c, a in sorted(fam.items(), key=lambda kv: -kv[1][1]):
    print(f"{c:45s} {a[0]:7d} {100*a[1]/tot:9.1f}% {a[2]/a[1]:8.2f}")

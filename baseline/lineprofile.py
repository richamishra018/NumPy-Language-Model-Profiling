"""
Line-by-line timing of the hot functions WITHOUT editing transformer.py.
We register the function objects with LineProfiler, run a few training steps, then print.

    python baseline/lineprofile.py small 10 > baseline/lineprofile_small.txt
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from line_profiler import LineProfiler
import transformer as tx
from profile_steps import run

config = sys.argv[1] if len(sys.argv) > 1 else "small"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10

targets = [tx.mha_forward, tx.mha_backward, tx.ff_forward, tx.ff_backward,
           tx.layer_norm_forward, tx.layer_norm_backward, tx.gelu, tx.gelu_grad,
           tx.softmax, tx.cross_entropy_loss, tx.AdamOptimizer.step]
lp = LineProfiler()
for fn in targets:
    lp.add_function(fn)
lp.enable_by_count(); run(config, steps); lp.disable_by_count()
lp.print_stats(output_unit=1e-3, stripzeros=True)      # report in milliseconds

"""Run cProfile on profile_steps.run() and write a sorted text report (no plotting libraries involved)."""
import cProfile, pstats, io, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_steps import run

config = sys.argv[1] if len(sys.argv) > 1 else "small"
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
pr = cProfile.Profile(); pr.enable(); run(config, steps); pr.disable()
pr.dump_stats(f"baseline/cprofile_{config}.prof")
buf = io.StringIO()
ps = pstats.Stats(pr, stream=buf).strip_dirs()
buf.write("=== sorted by tottime (self time, the one that finds hotspots) ===\n"); ps.sort_stats("tottime").print_stats(15)
buf.write("=== sorted by cumulative (inclusive, the one that shows call structure) ===\n"); ps.sort_stats("cumulative").print_stats(15)
open(f"baseline/cprofile_{config}.txt", "w").write(buf.getvalue()); print(buf.getvalue())

# VTune log (Stage 4)

Machine: Windows PC "richanotebook", Intel i5-3210M (Ivy Bridge), Intel VTune Profiler 2026.6 (GUI at C:\Program Files (x86)\Intel\oneAPI\vtune\2026.6).
Runs are on **native Windows**, not the Linux guest used for Stages 1-3 (the guest hides hardware counters).
Result folders live in `vtune/` (git-ignored); reports/screenshots go in this folder.

## Setup done
- VTune opened; the Intel Software Improvement Program prompt was answered "I do NOT consent" (privacy-preserving choice).

## Run log (exact settings per run are added below as they are run)

## Finding: hardware analyses are not available on this CPU (checked in the VTune GUI, 2026-11-12)
- Performance Snapshot: "VTune Profiler cannot recognize the processor ... not applicable to the current machine microarchitecture".
- Microarchitecture Exploration: "This analysis type is not applicable to the current machine microarchitecture" (plus a note about the sampling driver / running as administrator).
- Hotspots: only **User-Mode Sampling** is selectable; **Hardware Event-Based Sampling** is greyed out.
- Cause: the i5-3210M is Ivy Bridge (2012), older than the microarchitectures supported by VTune 2026.6.
- Consequence per the guide (step 4.1 / 4.14): do Hotspots with `sampling-mode=sw` and Threading here; get counter-style evidence for Runs 3 and 4 (uarch, memory access) from Stage 5 tools, or run them on a newer Intel machine on the team.

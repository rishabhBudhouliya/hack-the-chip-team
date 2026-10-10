# Example-guided RHS cache transfer

The model started from reference_level4.py and received an illustrative RHS caching pattern validated at M=512,K=256,N=1024. Optimization targeted M=1024,K=256,N=1024, float32.

Round 0 implemented a real cache: outer n loop loads rhs_cache[:,k,:] once per K tile, and the inner m loop reads those retained tiles. K tiles have separate storage; PSUM remains outside the contraction loop. The RHS cache allocation is outside n, so the same cache is refilled for each column strip.

All four standard correctness shapes passed simulation. The optimization checker also verified numerical output and input preservation on the new shape and successfully compiled the kernel for trn2,LNC=1. The subsequent hardware capture completed correctness checks and device timing on the new shape; results are below.

| Metric | Seed | Round 0 |
|---|---:|---:|
| Predicted latency | 131.872 us | 110.512 us |
| HBM traffic | 14680064 bytes | 7340032 bytes |

Predicted latency reduction: 16.20%. HBM traffic reduction: 50%. Later rounds did not improve these metrics. This is an example-guided model-generated optimization, not evidence that roofline feedback alone suffices or that the model weights learned permanently. A control run without the example and hardware timing would strengthen the experiment.

## Hardware capture

The hardware benchmark completed its correctness check for both kernels before timing. The exported profile contains 62 executions, consistent with one validation, ten warmups, and twenty timed runs per kernel.

| Metric | Original reference | Agent-generated cached kernel |
|---|---:|---:|
| Profiled device median | 248.577 us | 216.534 us |
| Timed samples | 20 | 20 |

Device median reduction: 12.89%; candidate faster in all 20/20 pairs. All 62 device durations agree with runtime nc_exec_running durations within 1 ns, and model IDs match. Mapping follows reference-then-candidate initial benchmark execution order.

This single hardware capture supports example-guided transfer to the new M=1024 shape, with 50% lower simulator HBM traffic and approximately 12.9% lower profiled device median latency. Repetition on the new shape remains needed to establish reproducibility. A run without the example remains needed to isolate the example's effect; this result does not demonstrate optimization from roofline feedback alone.

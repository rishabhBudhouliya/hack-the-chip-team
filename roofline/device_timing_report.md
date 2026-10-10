# Trainium2 kernel comparison

Shape: M=512, K=256, N=1024; float32; target trn2; LNC=1. Both kernels passed hardware correctness.

| Capture | Reference device median | Candidate device median | Reduction |
|---|---:|---:|---:|
| 1 | 129.452 us | 126.923 us | 1.95% |
| 2 | 129.100 us | 126.876 us | 1.72% |
| 3 | 129.381 us | 126.929 us | 1.89% |

The candidate was faster in all 60 corresponding timed iterations across three fresh processes. Each capture excludes one validation execution and ten warmups per kernel, leaving 20 timed executions per kernel. Device execution intervals and runtime nc_exec_running durations agree within 1 ns for all 186 captured executions; model IDs also match. Mapping follows the benchmark first executing reference, then candidate.

Compiler predictions were 65.936 us (reference) and 60.774 us (candidate), a 7.83% predicted reduction. Profiled device median reductions were 1.72–1.95%. Earlier unprofiled execution-call medians were 1079.0845 us and 1071.7205 us; these include runtime and transfer overhead.

HBM traffic remained 7,340,032 bytes per execution, twice the assumed ideal 3,670,016 bytes. The code change hoisted rhs_tile allocation while retaining the repeated DMA loads; it did not implement data reuse.

Conclusion: the tested candidate preserved hardware correctness and consistently lowered profiled device median latency by about 1.8–2.0% under these conditions. These captures do not establish unprofiled device-only latency or generalize to other shapes/dtypes. Profiling and concurrent workloads may affect absolute timing.

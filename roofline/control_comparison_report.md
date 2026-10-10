# Worked-example agent comparison: final results

Qwen/Qwen3-8B was asked to optimize the original matmul at M=1024,K=256,N=1024,float32. Each run started from the same correct reference, with four rounds and one sample per round. There were three fresh agent processes per condition (six runs total). The example condition received a worked RHS-cache pattern validated at M=512; the control retained the same verbal reuse guidance but omitted the worked code example.

| Outcome | Worked example | Verbal guidance control |
|---|---:|---:|
| Runs finding a correct, compiled improvement over the seed | 3/3 | 0/3 |
| Valid generated candidates | 12/12 | 0/12 |
| First improvement | Round 0 in all three runs | None within four rounds |
| Winning predicted latency | 110.512 us | 131.872 us, preserved seed |
| Winning simulator HBM traffic | 7 MiB | 14 MiB, preserved seed |

Success means a generated candidate passes all standard correctness shapes, additionally passes numerical/input-preservation checks on the target shape, compiles successfully, and predicts lower latency than the seed. The reference seed does not count as a generated success. All six seed performance records match.

The three example-guided runs produced identical normalized Python ASTs for their winning kernels. They cache every right-hand K tile separately, then reuse the cache across output-row tiles while retaining correct PSUM accumulation. They reduce HBM traffic by 50% and predicted latency by 16.20%. Repeated generation of the same implementation is reproducibility of this outcome, not evidence of broad exploration.

Control failures included using k before definition, loading the wrong K tile, out-of-bounds slicing, and copying a larger source into a smaller tile. No control candidate reached performance ranking, so the fastest preserved valid kernel remains the original seed.

## Hardware evidence

The example-guided winner passed hardware correctness at the target shape. One stable profiled capture, excluding validation and ten warmups per kernel, measured 20 timed executions per kernel:

| Kernel | Device median |
|---|---:|
| Original reference | 248.5775 us |
| Agent-generated cached kernel | 216.534 us |

The candidate had 12.89% lower device median latency and was faster in all 20 paired iterations. Device trace durations agree with runtime nc_exec_running within 1 ns. The earlier assistant-implemented cache experiment at M=512 also showed about 12.8% lower device latency in two stable repeat captures. That is supporting mechanism evidence, not independent repeated target-shape hardware measurement.

## Defensible claim and limits

On this task, adding a worked caching example enabled a correct improvement in all three agent trials, while the same verbal guidance without the example produced none within the matched four-round budget. The resulting kernel halved HBM traffic and showed 12.89% lower profiled device median latency in one hardware capture.

This is a small experiment on one model and one target shape, with a close transfer where only M changes. It does not establish a general success probability, permanent model learning, optimization from roofline feedback alone, or the worked example's benefit on unrelated kernels. The target-shape hardware comparison has one capture. Profiling can affect absolute timing.

## Evidence

Per-trial paths, metrics, normalized-code hashes, and success decisions are recorded in agent_comparison_results.json beside this report. The raw agent logs and terminal transcripts are retained under roofline_agent_repeats, alongside the initial two logs. Device trace tables are retained under roofline_decoded_transfer.

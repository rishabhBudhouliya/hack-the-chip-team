"""Compare NKI kernels with the installed standalone SDK's hardware executor.

Run in the seat pod, with the chip free. Timings cover the prepared execution call,
including any runtime/transfer overhead; they are not isolated device instruction times.
"""
import argparse
import inspect
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time


def timed_execute(executor, compiled, inputs, outputs):
    start = time.perf_counter_ns()
    executor(compiled, inputs, outputs)
    return (time.perf_counter_ns() - start) / 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="reference_level4.py")
    parser.add_argument("--candidate", default="roofline_reuse_best.py")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--shape", nargs=3, type=int, default=(512, 256, 1024), metavar=("M", "K", "N"))
    a = parser.parse_args()
    if a.warmup < 0 or a.iters < 1:
        parser.error("--warmup must be nonnegative and --iters must be positive")
    if any(n <= 0 or n % tile for n, tile in zip(a.shape, (128, 128, 512))):
        parser.error("M,K,N must be positive multiples of 128,128,512")

    sys.path.insert(0, str(Path.cwd()))
    import numpy as np
    import nkibench
    from nki.compiler.frontend import ParserFrontend
    from nki.framework.compiled import CompileOptions, compile_to_bir, run_from_bir

    execute = inspect.signature(run_from_bir).parameters["executor"].default
    if not callable(execute):
        raise RuntimeError("This SDK does not expose a callable default hardware executor")

    case = dict(zip(("M", "K", "N"), a.shape))
    args, _ = nkibench.make_inputs(case, 4)
    want = nkibench.LEVELS[4]["ref"](*args)
    states = []
    # Keep compile artifacts alive for both kernels until execution finishes.
    with tempfile.TemporaryDirectory(prefix="nki_hw_compare_") as tmp:
        for label, path in (("reference", a.reference), ("candidate", a.candidate)):
            print(f"Compiling and executing {label}: {path}", flush=True)
            kernel = nkibench.load_kernel(path, nkibench.LEVELS[4]["entry"])
            inputs = {"lhsT": args[0].copy(), "rhs": args[1].copy()}
            opts = CompileOptions(target="trn2", lnc=1, enable_simulation=False,
                                  artifacts_dir=str(Path(tmp) / label))
            bir = compile_to_bir(kernel, frontend=ParserFrontend(), inputs=inputs,
                                 compile_opts=opts, enable_cache=False)
            state = dict(label=label, path=path, times=[])

            def capture(compiled, prepared_inputs, outputs):
                execute(compiled, prepared_inputs, outputs)
                state.update(compiled=compiled, inputs=prepared_inputs, outputs=outputs)

            got = run_from_bir(bir, opts, inputs, executor=capture)
            error = nkibench.describe_mismatch(got, want)
            if error:
                raise RuntimeError(f"{label} failed hardware correctness: {error}")
            # Retain the returned array as well as the prepared buffers and runtime object.
            state["got"] = got
            states.append(state)
            print(f"{label}: hardware output matches reference", flush=True)

        for _ in range(a.warmup):
            for state in states:
                execute(state["compiled"], state["inputs"], state["outputs"])

        for iteration in range(a.iters):
            for state in states[::1 if iteration % 2 == 0 else -1]:
                state["times"].append(timed_execute(
                    execute, state["compiled"], state["inputs"], state["outputs"]))

        rows = []
        for state in states:
            error = nkibench.describe_mismatch(state["got"], want)
            if error:
                raise RuntimeError(f"{state['label']} failed correctness after repeated execution: {error}")
            row = dict(kernel=state["label"], path=state["path"], shape=case, dtype="float32",
                       target="trn2", lnc=1, warmup=a.warmup, iters=a.iters,
                       metric="prepared_executor_wall_time",
                       median_us=statistics.median(state["times"]),
                       p90_us=float(np.percentile(state["times"], 90)),
                       min_us=min(state["times"]))
            rows.append(row)
            print(json.dumps(row), flush=True)
        speedup = rows[0]["median_us"] / rows[1]["median_us"]
        print(f"Reference / candidate median: {speedup:.3f}x", flush=True)
        print("These are execution-call wall times, including runtime/transfer overhead; "
              "do not compare them directly to the theoretical device floor.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.exit(f"Hardware comparison stopped: {type(error).__name__}: {error}\n"
                 "If the runtime reports busy cores, release the model server's chip allocation "
                 "before retrying. No processes were stopped by this script.")

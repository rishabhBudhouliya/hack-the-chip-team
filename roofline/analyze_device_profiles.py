"""Compare profiled kernels from benchmark_hardware.py across fresh processes.

Mapping follows the benchmark's first two executions: reference, then candidate.
ExecutionInfo timestamps are in nanoseconds; validation and warmups are excluded.
"""
import argparse
import json
from pathlib import Path
import statistics


def summarize(executions, warmup=10):
    ordered = sorted(executions, key=lambda row: row["execution_index"])
    if len(ordered) < 2:
        raise ValueError("Profile must contain both kernels")
    names = [ordered[0]["neff_name"], ordered[1]["neff_name"]]
    if names[0] == names[1] or set(row["neff_name"] for row in ordered) != set(names):
        raise ValueError("Expected exactly two kernels in reference/candidate execution order")
    summaries = {}
    for label, name in zip(("reference", "candidate"), names):
        rows = [row for row in ordered if row["neff_name"] == name][1 + warmup:]
        times = [(row["execution_end_ts"] - row["execution_start_ts"]) / 1000 for row in rows]
        if not times or min(times) <= 0:
            raise ValueError("Missing timed executions or invalid timestamps")
        summaries[label] = dict(neff_name=name, samples_us=times,
                                median_us=statistics.median(times), n=len(times))
    a, b = summaries["reference"], summaries["candidate"]
    if a["n"] != b["n"]:
        raise ValueError("Incomplete profile: kernels have unequal execution counts")
    return dict(kernels=summaries,
                median_reduction_percent=100 * (1 - b["median_us"] / a["median_us"]),
                faster_pairs=sum(x > y for x, y in zip(a["samples_us"], b["samples_us"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("device_repeats.json"))
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    import pyarrow.parquet as pq

    paths = sorted(set(p for root in args.roots for p in root.rglob("ExecutionInfo.parquet")))
    if not paths:
        parser.error("No ExecutionInfo.parquet files found")
    results = []
    for path in paths:
        metadata = pq.read_table(path.with_name("Metadata.parquet")).to_pylist()
        if not metadata or metadata[0].get("is_simulation") is not False:
            raise ValueError(f"Profile is not confirmed as hardware: {path}")
        headers = pq.read_table(path.with_name("NeffHeader.parquet")).to_pylist()
        header_paths = {row["network_name"] for row in headers}
        if not any("/reference/" in name for name in header_paths) or not any(
                "/candidate/" in name for name in header_paths):
            raise ValueError(f"Missing benchmark kernel labels in NEFF headers: {path}")
        result = summarize(pq.read_table(path).to_pylist(), args.warmup)
        result["profile"] = str(path.parent)
        results.append(result)
        ref, candidate = result["kernels"]["reference"], result["kernels"]["candidate"]
        print(f"{path.parent}: reference {ref['median_us']:.3f} us, "
              f"candidate {candidate['median_us']:.3f} us; "
              f"reduction {result['median_reduction_percent']:.2f}%; "
              f"faster {result['faster_pairs']}/{ref['n']} pairs")
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {len(results)} independent capture summaries to {args.output}")
    print("Profiled device times; repeatability across captures matters more than pooled samples.")


if __name__ == "__main__":
    main()

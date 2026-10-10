#!/usr/bin/env bash
# Run inside seat-45; each process starts from the same original reference kernel.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
test -f agent_roofline.py
test -f reference_level4.py

batch="roofline_agent_repeats/$(date +%Y%m%dT%H%M%S)"
mkdir -p "$batch"
printf 'Logs and terminal transcripts: %s\n' "$batch"

# Reverse condition order for the second pair to reduce order effects.
for trial in example_1 control_1 control_2 example_2; do
  args=(agent_roofline.py --level 4 --optimize
        --model Qwen/Qwen3-8B
        --seed-kernel reference_level4.py
        --benchmark-shape 1024 256 1024
        --rounds 4 --samples 1
        --log "$batch/$trial.jsonl")
  if [[ "$trial" == example_* ]]; then
    args+=(--reuse-example)
  fi
  printf '\nRunning %s\n' "$trial"
  python "${args[@]}" 2>&1 | tee "$batch/$trial.stdout.txt"
done

printf '\nAll four trials finished. Copy roofline_agent_repeats to your laptop.\n'

"""Check trial isolation and ordering without contacting the model or importing NKI."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class RepeatComparisonTest(unittest.TestCase):
    def test_four_fresh_trials_with_matched_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "roofline").mkdir()
            (root / "bin").mkdir()
            script = root / "roofline" / "repeat_agent_comparison.sh"
            shutil.copyfile(Path(__file__).with_name(script.name), script)
            (root / "agent_roofline.py").touch()
            (root / "reference_level4.py").touch()
            spy = root / "bin" / "python"
            spy.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$REPEAT_CAPTURE"\n')
            spy.chmod(0o755)
            capture = root / "calls.txt"
            env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
                       REPEAT_CAPTURE=str(capture))
            subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
            calls = capture.read_text().splitlines()
            self.assertEqual(len(calls), 4)
            for index, name in enumerate(("example_1", "control_1", "control_2", "example_2")):
                self.assertIn("--seed-kernel reference_level4.py", calls[index])
                self.assertIn("--benchmark-shape 1024 256 1024", calls[index])
                self.assertIn("--rounds 4 --samples 1", calls[index])
                self.assertIn(name + ".jsonl", calls[index])
                self.assertEqual("--reuse-example" in calls[index], name.startswith("example"))
            self.assertEqual(len(list(root.rglob("*.stdout.txt"))), 4)


if __name__ == "__main__":
    unittest.main()

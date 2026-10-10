"""CPU-only loop checks; SDK evaluation is exercised separately inside the seat pod."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class OptimizationLoopTest(unittest.TestCase):
    def run_loop(self, batches, results, seed=None, shape=None, reuse_example=False):
        # Load only the loop functions so these checks need neither NumPy nor the Neuron SDK.
        path = Path(__file__).with_name("agent_remote.py")
        tree = ast.parse(path.read_text())
        names = {"solve_optimized", "optimization_prompt"}
        functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                     and n.name in names], type_ignores=[])
        grade = Mock(side_effect=lambda source, level: (
            0.5 if source == "wrong" else 1.0,
            {"correct": source != "wrong"}, "fix arithmetic"))
        ns = dict(json=json, os=__import__("os"), first_prompt=lambda *a: "start",
                  ask_parallel=Mock(side_effect=batches), extract_code=lambda s: s,
                  grade=grade, performance_check=Mock(side_effect=results),
                  repair_prompt=lambda level, source, feedback: "repair " + feedback)
        exec(compile(functions, str(path), "exec"), ns)
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(log=str(Path(tmp) / "attempts.jsonl"),
                                   rounds=len(batches), samples=len(batches[0]),
                                   offline=False, terse=0, seed_kernel=None,
                                   benchmark_shape=shape, reuse_example=reuse_example)
            if seed:
                args.seed_kernel = str(Path(tmp) / "seed.py")
                Path(args.seed_kernel).write_text(seed)
            log = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()):
                ns["solve_optimized"](args, log)
            winner = Path(tmp) / "attempts_best.py"
            saved = winner.read_text() if winner.exists() else None
        rows = [json.loads(line) for line in log.getvalue().splitlines()]
        return ns, rows, saved

    @staticmethod
    def performance(ns):
        return dict(predicted_ns=ns, ratio=ns / 4000, hbm_bytes=589824), "reuse data"

    def test_continues_after_correctness_and_preserves_fastest(self):
        ns, rows, saved = self.run_loop(
            [["slow"], ["fast"], ["regression"], ["compile_failure"]],
            [self.performance(15000), self.performance(6000), self.performance(9000),
             ValueError("unsupported instruction")])
        self.assertEqual(saved, "fast\n")
        self.assertEqual(len(rows), 4)
        self.assertIsNone(rows[-1]["performance"])
        self.assertIn("unsupported instruction", rows[-1]["feedback"])
        self.assertIn("passed simulation and compilation", ns["ask_parallel"].call_args_list[1].args[1])
        self.assertIn("did not beat", ns["ask_parallel"].call_args_list[3].args[1])
        self.assertNotIn("did not beat", ns["ask_parallel"].call_args_list[2].args[1])

    def test_invalid_and_compile_failed_candidates_cannot_win(self):
        ns, rows, saved = self.run_loop(
            [["wrong", "compile_failure", "valid"]],
            [ValueError("compile failed"), self.performance(7000)])
        self.assertEqual(saved, "valid\n")
        self.assertEqual(ns["performance_check"].call_count, 2)
        self.assertIsNone(rows[0]["performance"])

    def test_no_winner_when_every_compile_fails(self):
        _, rows, saved = self.run_loop([["compile_failure"]], [ValueError("compile failed")])
        self.assertIsNone(saved)
        self.assertIsNone(rows[0]["performance"])

    def test_seed_is_evaluated_and_survives_invalid_revisions(self):
        ns, rows, saved = self.run_loop([["wrong"], ["regression"]],
                                       [self.performance(5608), self.performance(9000)], seed="reference")
        self.assertEqual(saved, "reference\n")
        self.assertTrue(rows[0]["seed"])
        self.assertEqual(rows[0]["round"], -1)
        self.assertIn("passed simulation and compilation", ns["ask_parallel"].call_args_list[0].args[1])
        self.assertIn("M=512, K=256, N=1024", ns["ask_parallel"].call_args_list[0].args[1])
        self.assertIn("overwrites earlier tiles", ns["ask_parallel"].call_args_list[0].args[1])
        self.assertIn("not m", ns["ask_parallel"].call_args_list[0].args[1])

    def test_new_shape_reaches_checker_and_example_prompt(self):
        ns, rows, _ = self.run_loop([["valid"]],
                                   [self.performance(15000), self.performance(9000)],
                                   seed="reference", shape=(1024, 256, 1024), reuse_example=True)
        expected = dict(M=1024, K=256, N=1024)
        self.assertEqual(ns["performance_check"].call_args_list[0].args, (expected,))
        self.assertEqual(ns["performance_check"].call_args_list[1].args, (expected,))
        prompt = ns["ask_parallel"].call_args_list[0].args[1]
        self.assertIn("for M=1024, K=256, N=1024", prompt)
        self.assertIn("rhs_cache[:, k, :]", prompt)
        self.assertTrue(rows[0]["reuse_example"])


if __name__ == "__main__":
    unittest.main()

"""Check timing boundaries without importing the Neuron SDK."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    "benchmark_hardware", Path(__file__).with_name("benchmark_hardware.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TimingTest(unittest.TestCase):
    def test_times_executor_and_forwards_prepared_buffers(self):
        execute = Mock()
        with patch.object(module.time, "perf_counter_ns", side_effect=[1000, 61000]):
            elapsed = module.timed_execute(execute, "compiled", "inputs", "outputs")
        self.assertEqual(elapsed, 60)
        execute.assert_called_once_with("compiled", "inputs", "outputs")

    def test_execution_failure_is_not_reported_as_a_timing(self):
        execute = Mock(side_effect=RuntimeError("cores busy"))
        with self.assertRaisesRegex(RuntimeError, "cores busy"):
            module.timed_execute(execute, None, None, None)


if __name__ == "__main__":
    unittest.main()

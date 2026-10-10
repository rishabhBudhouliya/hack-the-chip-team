import unittest
from analyze_device_profiles import summarize


def event(index, name, duration):
    return dict(execution_index=index, neff_name=name,
                execution_start_ts=index * 1000000, execution_end_ts=index * 1000000 + duration)


class DeviceProfileTest(unittest.TestCase):
    def test_discards_initialization_and_maps_by_execution_order(self):
        rows = [event(0, "ref", 900000), event(1, "new", 800000),
                event(2, "ref", 200000), event(3, "new", 190000),
                event(4, "new", 100000), event(5, "ref", 120000)]
        result = summarize(list(reversed(rows)), warmup=1)
        self.assertEqual(result["kernels"]["reference"]["median_us"], 120)
        self.assertEqual(result["kernels"]["candidate"]["median_us"], 100)
        self.assertEqual(result["faster_pairs"], 1)

    def test_rejects_incomplete_capture(self):
        with self.assertRaisesRegex(ValueError, "unequal"):
            summarize([event(0, "ref", 1), event(1, "new", 1),
                       event(2, "ref", 120000), event(3, "new", 100000),
                       event(4, "ref", 120000)], warmup=0)


if __name__ == "__main__":
    unittest.main()

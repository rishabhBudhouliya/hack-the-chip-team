"""Check cache indexing and HBM traffic using NumPy; NKI compilation needs the pod."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np


class CachedMatmulTest(unittest.TestCase):
    def test_all_lab_shapes_and_traffic(self):
        nki = ModuleType("nki")
        nl = ModuleType("nki.language")
        nisa = ModuleType("nki.isa")
        nki.jit = lambda fn: fn
        nl.sbuf, nl.psum, nl.shared_hbm = "sbuf", "psum", "hbm"
        nl.float32 = np.float32
        nl.affine_range = range
        nl.ndarray = lambda shape, dtype, buffer: (
            np.zeros(shape, dtype=dtype) if buffer == nl.psum else np.empty(shape, dtype=dtype))
        traffic = [0]

        def dma_copy(dst, src):
            traffic[0] += src.nbytes
            np.copyto(dst, src)

        def matmul(dst, stationary, moving):
            dst += stationary.T @ moving

        nisa.dma_copy = dma_copy
        nisa.nc_matmul = matmul
        nisa.tensor_copy = lambda dst, src: np.copyto(dst, src)
        modules = {"nki": nki, "nki.language": nl, "nki.isa": nisa}
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location(
                "rhs_cached", Path(__file__).with_name("matmul_rhs_cached.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            rng = np.random.default_rng(45)
            for M, K, N, expected_bytes in [(128, 128, 512, 589824),
                                           (256, 256, 1024, 2621440),
                                           (128, 512, 512, 1572864),
                                           (512, 256, 1024, 4194304)]:
                with self.subTest(M=M, K=K, N=N):
                    lhs = rng.standard_normal((K, M)).astype(np.float32)
                    rhs = rng.standard_normal((K, N)).astype(np.float32)
                    before_lhs, before_rhs = lhs.copy(), rhs.copy()
                    traffic[0] = 0
                    got = module.nki_matmul_tiled_(lhs, rhs)
                    np.testing.assert_allclose(got, lhs.T @ rhs, atol=1e-4, rtol=1e-4)
                    np.testing.assert_array_equal(lhs, before_lhs)
                    np.testing.assert_array_equal(rhs, before_rhs)
                    self.assertEqual(traffic[0], expected_bytes)


if __name__ == "__main__":
    unittest.main()

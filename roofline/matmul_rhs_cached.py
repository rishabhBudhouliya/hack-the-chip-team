"""Level-4 experiment: cache a strip of RHS tiles across all output-row tiles.

Adapted from the shipped reference_level4.py (AWS Neuron NKI tutorial).
Copyright (C) 2024, Amazon.com. All Rights Reserved.

This experiment supports the lab's K=128/256/512 shapes. Cache size is bounded
to 128 * 4 * 512 elements (1 MiB for float32); larger K needs chunked caching.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_matmul_tiled_(lhsT, rhs):
    K, M = lhsT.shape
    K_rhs, N = rhs.shape
    TILE_M = 128
    TILE_K = 128
    TILE_N = 512
    assert K == K_rhs
    assert M % TILE_M == 0 and K % TILE_K == 0 and N % TILE_N == 0
    assert 0 < K <= 512, "This cache experiment supports K up to 512"
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for n in nl.affine_range(N // TILE_N):
        # Distinct storage per K tile prevents preloading from overwriting earlier data.
        rhs_cache = nl.ndarray((TILE_K, K // TILE_K, TILE_N),
                               dtype=rhs.dtype, buffer=nl.sbuf)
        for k in nl.affine_range(K // TILE_K):
            nisa.dma_copy(
                dst=rhs_cache[:, k, :],
                src=rhs[k * TILE_K:(k + 1) * TILE_K,
                        n * TILE_N:(n + 1) * TILE_N])

        for m in nl.affine_range(M // TILE_M):
            res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for k in nl.affine_range(K // TILE_K):
                lhs_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=lhs_tile,
                    src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                             m * TILE_M:(m + 1) * TILE_M])
                nisa.nc_matmul(dst=res_psum, stationary=lhs_tile, moving=rhs_cache[:, k, :])
            res_sb = nl.ndarray((TILE_M, TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res_sb, src=res_psum)
            nisa.dma_copy(
                dst=result[m * TILE_M:(m + 1) * TILE_M,
                           n * TILE_N:(n + 1) * TILE_N], src=res_sb)
    return result

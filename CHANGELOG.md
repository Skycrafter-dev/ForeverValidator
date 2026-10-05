# Changelog

## 0.2.5

- Fix CUDA/HIP segmented search returning out-of-order inputs after a winner
  is promoted or when the baseline has inputs just beyond the search horizon.
  Keep boundary restoration inputs visible to mutation; CPU behavior and GPU
  kernels are unchanged.

## 0.2.4

- Add CUDA/HIP candidate mutation restricted to randomly selected segments,
  preserving baseline controls outside the selected segments and allowing
  the changed-segment count to be updated between batches.
- Add custom polygon-prism volume entry evaluation on CUDA and generated HIP.
- Retain checkpoint-time evaluation on CUDA and HIP; CPU behavior is unchanged.

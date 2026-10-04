# Changelog

## 0.2.4

- Add CUDA/HIP candidate mutation restricted to randomly selected segments,
  preserving baseline controls outside the selected segments and allowing
  the changed-segment count to be updated between batches.
- Add custom polygon-prism volume entry evaluation on CUDA and generated HIP.
- Retain checkpoint-time evaluation on CUDA and HIP; CPU behavior is unchanged.

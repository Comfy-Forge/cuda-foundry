# pyg-lib — build notes
- **Source:** pyg-team/pyg-lib (0.5.0, commit ff87d1bb)
- **Build:** CMake, driven by upstream's setup.py (`cmake` + `cmake --build`).
  `FORCE_NINJA=1` selects the Ninja generator (upstream only does so when the
  *python* `ninja` package is importable, which conda's `ninja` is not), and
  `CMAKE_BUILD_PARALLEL_LEVEL` carries `jobs` because upstream passes no `-j`.
- **Quirks:** recursive clone (cutlass, cccl, cuCollections, parallel-hashmap,
  METIS). Patch (`patches/pyg_lib.py`): bridges `TORCH_CUDA_ARCH_LIST` into
  `CMAKE_CUDA_ARCHITECTURES`, pins RPATH to `$ORIGIN`, filters nvrtc out of
  the link (the farm's wheel vendors a 104 MB libnvrtc it never calls), and
  moves the farm's "C++20 for torch >= 2.13" switch into CMakeLists.txt so one
  tarball serves every torch cell.
- **Not ported:** the farm's patch of the *installed* torch's cmake files for
  torch 2.4–2.6 (legacy nvToolsExt). Fetch-time patching cannot reach the
  build env; those cells need a CMake-side shim before they can build here.

## Overrides

- **`arch_override.yml`** — overrides the shared arch policy's lists per CUDA
  line (`arch_list_by_cuda`), ported from cuda-wheels. Keeps Pascal/Volta
  (6.0/7.0) SASS for older cards on the CUDA 12.x lines; the 13.x rows floor
  at 7.5 because CUDA 13 nvcc removed compute_60/70 entirely. See the file's
  own comment for why the rows are the union of the old list and what
  upstream's ladder was actually building.

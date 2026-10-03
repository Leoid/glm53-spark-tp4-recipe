# exllamav3 1.4.5 — aarch64 / GB10 (sm121) build & run fix

**Problem.** `exllamav3` 1.4.5's CPU extension (`exllamav3_ext`) assumes x86:
it calls `__cpuidex` / `__builtin_cpu_supports` for AVX2/AVX512/F16C detection
and uses the x86 `pause` intrinsic (`__builtin_ia32_pause`) in its CPU spin
loops. On GB10 (ARM64, sm121) the upstream `patch_aarch64.py` did **not**
cover these, so the extension failed to compile / link. These edits make it
build and run on aarch64; the CPU SIMD paths are simply disabled (the real work
is on-GPU), and the spin loops use the ARM `yield` hint.

Patched source is preserved verbatim under `src/exllamav3_ext/`. Apply the same
edits to a fresh `exllamav3==1.4.5` checkout, then build with `build.sh`.

## Edits

### 1. Disable x86 CPU-feature detection on Linux/aarch64
`exllamav3_ext/avx2_target.cpp`, `avx512_target.cpp`
- Under `#ifdef __linux__`, force the capability to **false** instead of running
  the x86 `__cpuidex` path (which does not exist on ARM):
  - `avx2_target.cpp`: `avx2_supported = false;` (and same for `is_f16c_supported`).
  - `avx512_target.cpp`: the detector `return false;`.
- Net effect: no AVX2/AVX512/F16C CPU kernels are selected on aarch64.

### 2. Replace the `__builtin_cpu_supports` dispatch in the MoE CPU kernel
`exllamav3_ext/cpu/moe_mul1.cpp`
- The x86 `__builtin_cpu_supports(...)` runtime dispatch (which also checks OS
  state-saving) is not valid on ARM. The dispatch is made to fall through to the
  scalar path on aarch64 (see the annotated branch near the `__builtin_cpu_supports`
  comment, ~line 1136).

### 3. ARM spin-wait hint instead of x86 `pause`
`exllamav3_ext/cpu/moe_handoff.cu`, `exllamav3_ext/parallel/all_reduce_cpu.cu`
- Prepend, before any use, an aarch64 override of the pause macro:
  ```c
  #if defined(__aarch64__)
  #undef  EXL3_CPU_PAUSE
  #define EXL3_CPU_PAUSE() __asm__ __volatile__("yield" ::: "memory")
  #endif
  ```
  (x86 keeps `__builtin_ia32_pause()`.)

### 4. Remove the broken `atomic_ref` stubs in the CPU all-reduce
`exllamav3_ext/parallel/all_reduce_cpu_avx2.cpp` (and the avx512 sibling)
- The bogus `atomic_ref<uint32_t> f_(...)` stub(s) that did not compile under the
  aarch64 toolchain were removed / corrected; see the preserved file for the
  working form around the `*_stage_`/`*_tail_` counter handling.

### 5. Build sequencing — force reinstall
`build.sh`
- A plain `pip install .` **skipped** reinstalling when a same-version
  `exllamav3` was already present (runtime kept reporting `0.0.43`/stale).
  The build uses `pip install --force-reinstall` so the freshly compiled
  `exllamav3_ext` actually replaces the installed one. Verify afterwards:
  `python3 -c "import importlib.metadata as m; print(m.version('exllamav3'))"`
  → must print `1.4.5`.

## Upstream

`patch_aarch64.upstream.py` is upstream's aarch64 patcher **as shipped with
1.4.5** (kept for reference — it does not cover the above). To offer this
upstream, generate unified diffs of `src/exllamav3_ext/*` against a clean
`exllamav3==1.4.5` tree and open a PR to `turboderp-org/exllamav3`.

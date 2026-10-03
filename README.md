<div align="center">

# GLM-5.3 (full 743B) · EXL3 · TP4 on 4× DGX Spark

**Running the full, uncensored GLM-5.3 as a 3-bit EXL3 quant with tensor
parallelism + MTP speculative decoding across four NVIDIA DGX Spark (GB10).**

`743B params` · `EXL3 3bpw` · `TP4 / 4 nodes` · `MTP k=3 (accept ≈3.3)` ·
`200k ctx` · `vLLM OpenAI API`

</div>

---

## What this is (and whose work it stands on)

This is **not a new engine or a new model.** It is the small amount of glue +
**one compile fix** that let other people's excellent work actually run on the
DGX Spark's GB10 (ARM64, sm121):

- The model is **[`drowzeys/keys-GLM-5.3-EXL3-Abliterated`](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-Abliterated)**
  ("keys") — a 3bpw EXL3 **abliterated/derisked** quant of **Z.ai's GLM-5.3**.
- The quant format + kernels are **turboderp's [`exllamav3`](https://github.com/turboderp-org/exllamav3)**.
- The multi-node EXL3 serving stack (image chain + `start-tp4.sh` launcher) is
  **Mia AI Lab's**.

All we did was: **fix `exllamav3` 1.4.5 so it compiles and runs on aarch64/GB10**
(upstream's `patch_aarch64.py` didn't cover the x86-only bits), and **adapt
Mia's launcher to our 4-node switched-RoCE fabric**. That's it. Full, precise
attribution in **[docs/CREDITS.md](docs/CREDITS.md)**.

> **Live status (2026-10-03):** serving at `http://<head>:8888/v1`, model
> `GLM-5.3-EXL3`, TP4, MTP k=3, 200k context (no-DCP), thinking-off by default.

---

## The model

**[`drowzeys/keys-GLM-5.3-EXL3-Abliterated`](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-Abliterated)**

| | |
|---|---|
| Base | `zai-org/GLM-5.3` (743B, DeepSeek-sparse-attention MoE) |
| Quant | **EXL3 3-bit/weight** (routed experts 3bpw + bf16 rest), ~330 GB, 41 shards |
| Variant | **Abliterated / derisked** — safety refusals removed (red-team / security research) |
| Footprint | ~80 GB/rank across TP4 → fits 128 GB unified with headroom |
| License | MIT (weights); gated by a Responsible-Use agreement on HF |

> ⚠️ This is an uncensored model intended for authorized research. You are
> responsible for output filtering and deployment safety.

---

## Performance (measured on this stack)

4× DGX Spark (GB10), EXL3 3bpw, **TP4, MTP k=3**, 200k ctx, FP8 KV, thinking-off.
Decode = 256 output tokens; **agg** = aggregate throughput across all concurrent
streams, **per-req** = average single-stream rate within that batch.
Measured 2026-10-03 (single run per cell — treat as ±10%).

### Decode throughput — aggregate tok/s (per-req in parentheses)

| Workload | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| **Prose** | 15.0 (15.0) | 29.1 (15.2) | 36.7 (9.4) | 55.9 (7.3) |
| **Code** | **25.5** (25.5) | 35.3 (17.9) | 52.8 (13.8) | **70.7** (9.5) |
| **Structured (JSON)** | 20.3 (20.3) | 29.4 (17.4) | 37.9 (10.4) | 54.5 (7.9) |

- **Single-stream (C1):** code ≈25, structured ≈20, prose ≈15 tok/s. Code is
  fastest because MTP's draft accepts longer runs on predictable syntax.
- **Concurrency scales aggregate ~3.5–4×** from C1→C8 (the 4-slot batch does the
  work); per-request rate falls as slots share the bandwidth-bound decode.

### Prefill — tok/s (TTFT for a cold, unique prompt)

| Context | 4k | 8k | 32k |
|---|---|---|---|
| Prefill tok/s | 549 | 466 | 552 |
| TTFT | 7.3s | 17.2s | 58.0s |

> Prefill hovers ~500–550 tok/s across 4k–32k. Decode is **bandwidth-bound**
> (~273 GB/s/node, ~10 GB/rank active → ~37 ms/token floor), so the honest
> ceiling for single-stream tuning is ~1.5×, not 2× — see
> [docs/GOTCHAS.md](docs/GOTCHAS.md).

---

## Quickstart

Full build + launch in **[docs/BUILD.md](docs/BUILD.md)**. In brief:

1. **Quant engine** — check out `exllamav3==1.4.5`, apply the aarch64 fix
   (copy `exllamav3-aarch64-patch/src/exllamav3_ext/*` or follow
   [PATCH.md](exllamav3-aarch64-patch/PATCH.md)), build with
   `exllamav3-aarch64-patch/build.sh` (uses `--force-reinstall`).
2. **Serving image** — build the `serving/Dockerfile.*` chain on Mia's base
   → `glm53-exl3-tp4:keys3`.
3. **Weights** — place `drowzeys/keys-GLM-5.3-EXL3-Abliterated` on the head,
   NFS-export to workers.
4. **Launch** — fill `serving/env.tp4.example` → `.env.tp4`, then
   `start-tp4.sh` (see BUILD.md for the exact `env ... ./start-tp4.sh` line).
5. **Use** — OpenAI base URL `http://<head>:8888/v1`, model `GLM-5.3-EXL3`.

---

## The one fix that was actually ours

`exllamav3` 1.4.5's CPU extension assumes x86 (`__cpuidex`,
`__builtin_cpu_supports`, `__builtin_ia32_pause`). On GB10/aarch64 it wouldn't
build, and upstream's `patch_aarch64.py` didn't cover it. The working edits:

- force the x86 CPU-feature detectors to `false` on `__linux__`/aarch64,
- redefine `EXL3_CPU_PAUSE()` to the ARM `yield` hint under `__aarch64__`,
- drop the broken `atomic_ref` stubs in the CPU all-reduce,
- `pip install --force-reinstall` so the rebuilt `exllamav3_ext` actually lands.

Details + the preserved patched source: **[exllamav3-aarch64-patch/](exllamav3-aarch64-patch/PATCH.md)**.
This is the piece worth sharing upstream — anyone building EXL3 on a DGX Spark /
GB10 needs it.

---

## Alternate faster recipe (not used here)

knapcio / Zbigniew Majewski published a **faster** full-GLM-5.3 TP4 recipe
(`glm53-roce:v11` image + `Tech2wild/GLM-5.3-Int4-Int8Mix` weights, 32k context)
reaching **~30 tok/s prose c1** — it bundles **RoCEnante** (`GLM_ROCE_ALLREDUCE=1`)
and working **MTP K=2** on Int4/Int8. It is **stock (not abliterated)** and trades
context down to 32k. See [docs/CREDITS.md](docs/CREDITS.md) and
[docs/ALT-RECIPE.md](docs/ALT-RECIPE.md) for how it compares and the
uncensored-at-32k experiment we scoped but did not run.

---

## Hardware

4× DGX Spark (GB10, sm121/12.1a, aarch64), 128 GB unified memory each,
**~273 GB/s** memory bandwidth per node, joined by ConnectX-7 over a CRS504 RoCE
switch (10.0.0.x, MTU 9000, GID index 3). Weights live on the head node and are
NFS-exported to the three workers.

## Backup / restore

This repo is the curated preservation; the full `~/glm53-mia` tree and the two
custom Docker images also need to live off-fleet. See **[ARCHIVE.md](ARCHIVE.md)**.

## License

Our patch + glue: **MIT** ([LICENSE](LICENSE)). Everything else keeps its own
terms — **GLM-5.3** (zai-org), `exllamav3` (turboderp-org), the weights
(drowzeys), Mia AI Lab's images. **DFlash2 (CC BY-NC-ND 4.0) is not used.** Do
not assume any bundled weight is redistributable.

# GLM-5.3 (full 743B) — EXL3 TP4 on 4× DGX Spark

Serving recipe and build fixes for running the **full GLM-5.3** (not Flash) as a
3bpw **EXL3** quant with **TP4 + MTP speculative decoding** across four
**NVIDIA DGX Spark (GB10, sm121, aarch64, 128 GB unified)** nodes.

This repo preserves the hard-to-reproduce parts of that stack: one genuinely
**novel patch** (making `exllamav3` 1.4.5 compile and run on GB10/aarch64) plus
the **integration glue** that adapts Mia AI Lab's EXL3 serving chain to a
4-node switched-RoCE fabric. Most of the serving chain is *completed/adapted
upstream work*, not invented here — see **[docs/CREDITS.md](docs/CREDITS.md)**
for exactly who made what.

> **Status (as of 2026-10-03):** live at `http://<head>:8888`, model
> `GLM-5.3-EXL3`, TP4, MTP k=3 (acceptance ≈3.3), 200k context (no-DCP),
> thinking-off by default. Single-stream decode ≈20 tok/s prose / ≈23 tok/s
> code; concurrent agg ≈30–40 tok/s. Prefill ≈580 tok/s @4k → ≈700 @32k.

---

## What is actually novel here

**The `exllamav3` 1.4.5 aarch64/GB10 build fix.** Upstream `patch_aarch64.py`
was broken for 1.4.5; the extension would not compile on GB10 (sm121, ARM64).
The working fixes are in **[exllamav3-aarch64-patch/PATCH.md](exllamav3-aarch64-patch/PATCH.md)**
with the patched source preserved under `exllamav3-aarch64-patch/src/`. This is
the piece with external value — anyone building EXL3 on a DGX Spark / GB10 /
Jetson Thor-class box needs it — and it is the hardest part of the stack to
reconstruct from memory, so it lives here first.

## What is integration (valuable, not new capability)

- The **keys3 image chain** (`serving/Dockerfile.*`) — adapted from Mia AI Lab's
  vLLM-EXL3 base and the "keys" lineage.
- **`serving/start-tp4.sh`** — Mia's multi-node launcher, adapted to our fabric
  (NFS weight share, per-node users, GID3, dual-port NCCL HCA, NCCL channel
  forwarding). Driven by `serving/env.tp4.example`.
- **`serving/chat_template.jinja`** — edited so `thinking_enabled` defaults
  **false** (direct OpenAI clients get content, not an empty think block).
- **200k / no-DCP** context config; **MTP k=3** spec-decode config.

See **[docs/BUILD.md](docs/BUILD.md)** to rebuild the chain and
**[docs/GOTCHAS.md](docs/GOTCHAS.md)** for every trap we hit.

## Hardware context

4× DGX Spark (GB10, sm121/12.1a, aarch64), 128 GB unified memory each,
**~273 GB/s** memory bandwidth per node (decode is bandwidth-bound — see
GOTCHAS), joined by ConnectX-7 over a CRS504 RoCE switch (10.0.0.x, MTU 9000,
GID index 3). Weights live on the head node and are NFS-exported to the workers.

## Alternate faster recipe (not used here)

knapcio / Zbigniew Majewski published a **faster** full-GLM-5.3 TP4 recipe
(`glm53-roce:v11` image + `Tech2wild/GLM-5.3-Int4-Int8Mix` weights, 32k context)
reaching ~30 tok/s prose c1 — it bundles **RoCEnante** (`GLM_ROCE_ALLREDUCE=1`)
and working **MTP K=2** on Int4/Int8. It is **stock (not abliterated)** and
trades context down to 32k. See [docs/CREDITS.md](docs/CREDITS.md) and
[docs/ALT-RECIPE.md](docs/ALT-RECIPE.md) for how it compares and the
uncensored-at-32k experiment we scoped but did not run.

## Restore / full backup

This repo is the *curated* preservation. The complete `~/glm53-mia` tree
(2.6 G) is archived separately; the two custom Docker images
(`glm53-exl3-quant:1.4.5`, `glm53-exl3-tp4:keys3`, ~21 GB each) should be
`docker save`d off-fleet too. See **[ARCHIVE.md](ARCHIVE.md)**.

## License

Our patch and glue: **MIT** (see [LICENSE](LICENSE)). Third-party code and model
weights keep their own terms: **GLM-5.3** under its own license (zai-org),
`exllamav3` under its license (turboderp-org), **DFlash2 is CC BY-NC-ND 4.0 and
is deliberately not used**. Do not assume any bundled weight is redistributable.

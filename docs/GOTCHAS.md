# Gotchas — hard-won fixes (GB10 / DGX Spark / GLM-5.3 EXL3 TP4)

## Build chain
- **exllamav3 reports stale version after build** → use `pip install --force-reinstall`
  (plain install skips same-version). Verify `importlib.metadata.version('exllamav3') == 1.4.5`.
- **aarch64 compile** → apply `exllamav3-aarch64-patch/PATCH.md` (x86 cpuid/pause not on ARM).
- **flashinfer-cubin `No module named filelock`** under `uv build --no-build-isolation`:
  the temp venv can't see system site-packages. Fix: `uv venv --system-site-packages /opt/fienv`
  then `--python /opt/fienv/bin/python` (plain `python3 -m venv` fails — no ensurepip in the container).
- **cutlass-dsl conflict (4.7.0 vs 4.5.2)** → pin `nvidia-cutlass-dsl==4.6.2` in the wheel override.
- **start-tp4 missing dirs / overlay verify** → supply `overlay/`, `ablit/` (`ABLIT=0`), `files/`,
  `tests/`; `SKIP_OVERLAY_VERIFY=1` (the glm-tp3 overlay test expects a fat-MoE diag the mul1 build lacks).

## Launch / runtime
- **Unified-memory swap death-spiral on first inference** (Triton JIT compile spike): the node
  thrashes, SSH banner-times-out, inference dies. Needs a hard power-cycle. Keep memory headroom;
  set `vm.compaction_proactiveness=0` (persisted in `/etc/sysctl.d/`). **Never run heavy I/O /
  installs on a node that is actively serving** — stream backups when it's idle.
- **1M context + DCP cudagraph hang** → workers freeze at warmup, head spins on shm_broadcast.
  Drop DCP; run **200k no-DCP**.
- **Empty chat content** → GLM's thinking burned the token budget. Fix: `thinking_enabled`
  defaults false in `chat_template.jinja`; or send `chat_template_kwargs.enable_thinking:false`.

## Thinking / reasoning (verified on this stack, 2026-10)
- Default (no params) → **thinking OFF** (`<think></think>`).
- **Top-level `reasoning_effort: low|high`** (what OpenAI-style clients like zcode send) →
  thinking **ON** (vLLM serving layer maps it to enable). `reasoning_effort` inside
  `chat_template_kwargs` does **not** auto-enable.
- `chat_template_kwargs.enable_thinking:true` → ON.
- Only `low` and `high` are valid efforts; **anything else → `max`** (silently). No "medium".
- Reading a client's actual request: vLLM logs no bodies. The endpoint is plain HTTP, so
  `tcpdump -A 'tcp port 8888 and src host <client>'` reveals the exact params on the wire.

## Spec decode
- **MTP k=3** works on the EXL3 abliterated weights (acceptance ≈3.3) — this is the big decode lever.
- **DFlash2-for-code is a dead end on this model**: the draft is dense qwen3 attention; GLM uses
  **DSA** (sparse) with an `indexer.k_cache` whose page size won't unify with the draft's →
  `NotImplementedError: unify_kv_cache_spec_page_size` at KV init. Never produces a token.
  (Also CC BY-NC-ND licensed.) Stay on MTP.

## Collectives / speed
- Decode is **bandwidth-bound**: ~10 GB/rank weights ÷ ~273 GB/s ≈ 37 ms/token forward; the
  TP collective is the *smaller* term (~11 ms of a ~79 ms step). So no "2×" is on the table;
  ~1.5× is the realistic ceiling. **273 GB/s, not 2 TB/s** — don't repeat that error.
- **RoCEnante** (`b12x comm.roce`) is the real collective lever: fp32-accumulate all-reduce
  (better than NCCL bf16-per-hop), bit-exact all-gather, only sub-2MB/16MB intercepted. On our
  own keys3 image it was a **NO-GO to hand-port** (the image lacks the whole `vllm/utils/b12x.py`
  preparation framework + warmup driver). The clean path is the knapcio `glm53-roce:v11` image,
  which already bundles it (`GLM_ROCE_ALLREDUCE=1`).

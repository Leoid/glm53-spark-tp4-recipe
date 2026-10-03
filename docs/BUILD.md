# Build & run

## 0. Prereqs
4× DGX Spark (GB10, sm121, aarch64), 128 GB unified each. Docker. A RoCE fabric
(we use a CRS504 switch, 10.0.0.x, MTU 9000, GID index 3). ~308 GB free for the
EXL3 weights on the head node, NFS-exported to the three workers.

## 1. exllamav3 1.4.5 quant engine (image `glm53-exl3-quant:1.4.5`)
1. Check out `exllamav3==1.4.5`.
2. Apply the aarch64 fixes — copy the patched files from
   `exllamav3-aarch64-patch/src/exllamav3_ext/` over the checkout (or re-apply
   the edits in `exllamav3-aarch64-patch/PATCH.md`).
3. Build with `exllamav3-aarch64-patch/build.sh` (uses `pip install --force-reinstall`).
4. Verify: `python3 -c "import importlib.metadata as m; print(m.version('exllamav3'))"` → `1.4.5`.

## 2. Serving image chain (image `glm53-exl3-tp4:keys3`)
The `serving/Dockerfile.*` chain layers on top of Mia AI Lab's vLLM-EXL3 base:
`Dockerfile.v145` → `Dockerfile.mul1` → `Dockerfile.keys` → `Dockerfile.keys3`
(`Dockerfile.dcp` is the optional DCP variant — not used; we run no-DCP).
Build in that order; the final tag is `glm53-exl3-tp4:keys3`.

## 3. Weights
`drowzeys/keys-GLM-5.3-EXL3-Abliterated` (3bpw routed experts + bf16 rest,
~308 GB, ~80 GB/rank). Place on the head node and NFS-export to workers at a
shared mount (we use `/mnt/glm-exl3`).

## 4. Launch (TP4)
1. Copy `serving/env.tp4.example` → `.env.tp4` and fill in: `HEAD_IP/WORKER*_IP`
   (10.0.0.1–4), per-node users, `NFS_SHARE=1`, `GID_INDEX=3`, NCCL HCA vars,
   `MAX_MODEL_LEN=200000`, `GPU_MEM_UTIL≈0.82`, `MAX_NUM_BATCHED_TOKENS=4096`,
   `KV_CACHE_DTYPE=fp8`, `SPEC_METHOD=mtp`, `MTP_TOKENS=3`,
   `EXTRA_ARGS="--kv-cache-memory-bytes 12000000000"`.
2. On the head:
   ```bash
   env ABLIT=0 SKIP_PULL=1 SKIP_BUILD=1 SKIP_DOWNLOAD=1 SKIP_OVERLAY_VERIFY=1 \
       ENFORCE_EAGER=0 setsid nohup bash ./start-tp4.sh > launch.log 2>&1 < /dev/null &
   ```
3. Wait for health: `curl -s -o /dev/null -w '%{http_code}' http://<head>:8888/health` → 200
   (≈10–15 min: 41 weight shards + KV + cudagraph capture).

## 5. Smoke test
```bash
curl -s http://<head>:8888/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-EXL3","prompt":"def is_prime(n):\n","max_tokens":128,"temperature":0.2}'
```
Expect coherent code. For reasoning, send top-level `reasoning_effort:"low"` (see GOTCHAS).

## Clients
- OpenAI-compatible base URL `http://<head>:8888/v1`, model `GLM-5.3-EXL3`, no key.
- Hermes: provider `dgx-glm` (thinking off) / `dgx-glm-think` (thinking on).
- Android: any custom-base-URL app (Chatbox/ChatterUI) — allow cleartext HTTP.

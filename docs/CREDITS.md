# Credits & provenance

Being precise about "consumed vs. made," because it changes what is reusable
and what is ours to publish.

## Made here (novel)
- **exllamav3 1.4.5 aarch64/GB10 build fix** — the `__cpuidex`/`__builtin_cpu_supports`
  disable, the `EXL3_CPU_PAUSE` → ARM `yield` override, the `atomic_ref` stub fix,
  and the `--force-reinstall` sequencing. Upstream `patch_aarch64.py` was broken
  for 1.4.5 and nobody had fixed it. See `exllamav3-aarch64-patch/PATCH.md`.
  Authored in a Claude Code session (model: Claude Opus 4.8) with the operator.

## Adapted / integrated here (valuable, not new capability)
- The **keys3 Docker chain** and **`start-tp4.sh`** launcher — adapted to our
  4-node switched-RoCE fabric (NFS weight share, per-node users, GID3, dual-port
  NCCL HCA, NCCL channel forwarding), 200k/no-DCP context, MTP k=3.
- **`chat_template.jinja`** thinking-off default edit.

## Upstream — who made what
| Component | Author / project | What it provides |
|---|---|---|
| **GLM-5.3** base model | **Z.ai / zai-org** | the 743B model + GLM-5.3 license |
| **EXL3 quant engine** | **turboderp — `turboderp-org/exllamav3`** | the EXL3 format & `exllamav3_ext` (the thing we patched for aarch64) |
| **vLLM-EXL3 base image + keys3 serving chain + `start-tp4.sh`** | **Mia AI Lab (`MiaAI-Lab`)** | the multi-node EXL3 serving stack we adapted |
| **GLM-5.3-EXL3-Abliterated 3bpw weights** | **`drowzeys` ("keys")** | the uncensored/derisked EXL3 weights we serve |
| **vllm-glm53-flash image base** | **tonyd2wild** | sm121 vLLM base used by the alt recipe |
| **GLM-5.3-Int4-Int8Mix weights** | **Tech2wild** | compressed-tensors W4A16/W8A16 (alt recipe; stock, not abliterated) |
| **GLM-5.3-DERISKED-Int4-Int8Mix** | **Blackfrost-AI** | uncensored compressed-tensors quant (held locally; alt-recipe experiment) |
| **sparse MLA / deep_gemm bypass** | **CosmicRaisins** | sm12x MoE/MLA kernels & fallbacks |
| **RoCEnante (`comm.roce`) + b12x** | **`local-inference-lab` / b12x; Luke Alonso, Jason Cook, tonyd2wild, rhys101** | one-shot RDMA all-reduce/all-gather for multi-node TP |
| **DSA (sparse attention)** | **DeepSeek** | the DeepSeek sparse-attention architecture GLM-5.3 uses |
| **faster full-GLM TP4 recipe** | **knapcio / Zbigniew Majewski — `knapcio/GLM-5.3-4x-DGX-Spark-TP4`** | prefill switch, DSA indexer shortcut, dirty-L2 fix, launcher; the 30 tok/s Int4/Int8 recipe |
| **native MTP K2 path** | **Z.ai + vLLM** | the speculative-decode method |

## License boundaries
- Our patch + glue: **MIT**.
- `exllamav3`: its own license (turboderp-org). GLM-5.3 weights: **GLM-5.3 license** (zai-org).
- **DFlash2 is CC BY-NC-ND 4.0 — not used** (also incompatible with GLM's DSA KV layout; see GOTCHAS).
- Model weights (abliterated / Int4 / derisked variants) retain their own terms; do not assume redistributable.

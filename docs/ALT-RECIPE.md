# Alternate recipe — knapcio `glm53-roce:v11` (faster, stock, 32k)

Source: **`knapcio/GLM-5.3-4x-DGX-Spark-TP4`** (Zbigniew Majewski). Published,
measured recipe for the *same* hardware. Kept here as a comparison + a scoped
experiment we did **not** run (it would replace the live stack).

## Their measured numbers (vs ours)
| | knapcio recipe | ours (EXL3) |
|---|---|---|
| Prose decode c1 | **30.0** (off) / 24.9 (think) | ~19.6 prose / ~22.6 code (off) |
| Prose agg c4 | **61.1** | ~35 |
| Prefill 4k / 32k | **~1.0k / 840** | 579 / 704 |
| Context | **32k** | 200k |
| Uncensored | ❌ stock (`Tech2wild/GLM-5.3-Int4-Int8Mix`, from `zai-org/GLM-5.3`) | ✅ abliterated (`drowzeys`) |
| Quality gate | 72.7/75 | — |

## What makes it fast (all in the image `glm53-roce:v11-b58f34ea`)
- `GLM_ROCE_ALLREDUCE=1` — **RoCEnante** bundled (tonyd2wild v11 base + the Apache-2.0 b12x subset).
- `GLM_MTP_FIX=1` — working native **MTP K=2** on Int4/Int8 (the fix that was blocked on our image).
- `GLM_INDEXER_SHORTCUT=1` — DSA sparse-attention indexer shortcut.
- `GLM_DIRTY_L2=discard` — dirty-L2 fix.
- `GLM_W2_PREFILL_CHUNK=2048`, `NCCL_BUFFSIZE=1048576`, `NCCL_MAX_NCHANNELS=8`.
- MTP K=2, FP8 target/draft KV, block size 64, 2 GiB target KV/rank, prefix caching off.
- **DFlash2 not used** (CC BY-NC-ND; also the DSA KV-page dead end).

Model: `Tech2wild/GLM-5.3-Int4-Int8Mix` rev `206507bb`, W4A16/W8A16 compressed-tensors,
Marlin MoE, 282 shards (~405 GB). Launch: `./start.sh serve`.

## The experiment we scoped but did NOT run
Run `glm53-roce:v11` with our **uncensored** `Blackfrost-AI/GLM-5.3-DERISKED-Int4-Int8Mix`
(same compressed-tensors format, already on disk at `/home/dgx2/models/`) at 32k,
to get **RoCEnante + MTP-fix speed AND uncensored**.
- Upside: plausibly ~30 tok/s while keeping uncensored.
- Risk: `GLM_MTP_FIX`/indexer patches may be tuned to Tech2wild's exact layout
  and might not transfer to Blackfrost's → could land ~24 (barely above ours).
- Cost: both models need all 4 Sparks, so testing means ~30–40 min of GLM
  downtime (launch alt → benchmark → keep or relaunch EXL3). Decision gated on
  whether the operator routinely needs >32k context. **Unverified until run.**

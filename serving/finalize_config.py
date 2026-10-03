#!/usr/bin/env python3
"""Post-compile fixups so the exllamav3 output is a Mia-native vLLM checkpoint.

* config.json / quantization_config.json: quant_method=exl3, bits=<int>, codebook=mcg,
  scope=glm53_routed_experts_only, head_bits=16, mtp_bits=16 (Mia's Exl3Config reads bits/codebook/scope;
  everything else is carried as raw config). The FP8 source block is preserved as original_quantization_config.
* sanity: every routed expert has trellis/suh/svh/mcg; every non-expert linear is a plain .weight; report dtypes.
Usage: finalize_config.py <out_dir> [expert_bits]
"""
import json, os, sys, re, struct, collections

out = sys.argv[1]
bits = int(sys.argv[2]) if len(sys.argv) > 2 else 3
CODEBOOK = sys.argv[3] if len(sys.argv) > 3 else "mcg"

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

# --- tensor inventory
suffix_by_kind = collections.Counter()
dtypes = collections.Counter()
experts_ok, experts_bad = 0, []
nonexp_quant = []
shards = sorted(p for p in os.listdir(out) if p.endswith(".safetensors"))
seen = {}
for sh in shards:
    for k, v in read_header(os.path.join(out, sh)).items():
        if k == "__metadata__":
            continue
        seen[k] = v["dtype"]
exp = collections.defaultdict(set)
for k in seen:
    m = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(\w+)$", k)
    if m:
        exp[(int(m.group(1)), int(m.group(2)), m.group(3))].add(m.group(4))
        continue
    m2 = re.match(r"(model\.layers\.\d+\.(self_attn|mlp\.shared_experts|mlp)\.[\w.]+?)\.(trellis|suh|svh|mcg|mul1)$", k)
    if m2:
        nonexp_quant.append(k)
for key, sfx in exp.items():
    if {"trellis", "suh", "svh"} <= sfx and ({"mcg"} <= sfx or {"mul1"} <= sfx):
        experts_ok += 1
    else:
        experts_bad.append((key, sorted(sfx)))
for k, d in seen.items():
    dtypes[(k.rsplit(".", 1)[-1], d)] += 1
print(f"shards={len(shards)} tensors={len(seen)} experts_ok={experts_ok} experts_bad={len(experts_bad)} nonexpert_quantized={len(nonexp_quant)}")
if experts_bad:
    print("  bad:", experts_bad[:5])
if nonexp_quant:
    print("  !! non-expert EXL3 tensors (Mia overlay expects native):", nonexp_quant[:5])
for (sfx, d), n in sorted(dtypes.items(), key=lambda x: -x[1])[:12]:
    print(f"  {sfx:14s} {d:8s} {n}")

# --- config fixups
def fix(path, is_hf_config):
    d = json.load(open(path))
    q = d.get("quantization_config", d if not is_hf_config else {}) if is_hf_config else d
    if is_hf_config:
        q = d.setdefault("quantization_config", {})
    orig = None
    if q.get("quant_method") not in (None, "exl3"):
        orig = dict(q)
        q.clear()
    if orig is not None:
        q["original_quantization_config"] = orig
    q.update({"quant_method": "exl3", "bits": bits, "codebook": CODEBOOK, "scope": "glm53_routed_experts_only",
              "head_bits": 16, "mtp_bits": 16, "keys_profile": f"routed-experts-{bits}bpw-{CODEBOOK}; attn/dense/shared/head bf16",
              "keys_source": "zai-org/GLM-5.3 (FP8) via exllamav3 1.4.5 + keys encode farm"})
    q.pop("tensor_storage", None) if is_hf_config else None
    json.dump(d, open(path, "w"), indent=2)
    print(f"wrote {path}: quant_method={q['quant_method']} bits={q['bits']} codebook={q['codebook']}")

fix(os.path.join(out, "config.json"), True)
qc = os.path.join(out, "quantization_config.json")
if os.path.exists(qc):
    fix(qc, False)
for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "model.safetensors.index.json"):
    print(f"  {fn}: {'ok' if os.path.exists(os.path.join(out, fn)) else 'MISSING'}")

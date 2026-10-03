import sys
p=sys.argv[1]; s=open(p).read()
old="""    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        if isinstance(layer, RoutedExperts):
            return Exl3MoEMethod(layer.moe_config, self)"""
new="""    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        # KEYS PATCH: the MTP block (model.layers.<num_hidden_layers>+) is stored bf16
        # (-mb 16) -> leave it to the stock unquantized MoE path so SPEC_METHOD=mtp works.
        if isinstance(layer, RoutedExperts) and _is_mtp_prefix(prefix):
            logger.info_once("EXL3: MTP block %s left unquantized (bf16 experts)", prefix)
            return None
        if isinstance(layer, RoutedExperts):
            return Exl3MoEMethod(layer.moe_config, self)"""
assert s.count(old)==1, s.count(old); s=s.replace(old,new)
helper='''

def _is_mtp_prefix(prefix: str) -> bool:
    """True for layers at index >= num_hidden_layers (DeepSeek/GLM MTP blocks)."""
    import re as _re
    m = _re.search(r"\\.layers\\.(\\d+)\\.", prefix + ".")
    if not m:
        return "mtp" in prefix.lower()
    idx = int(m.group(1))
    n = None
    try:
        from vllm.config import get_current_vllm_config
        n = int(get_current_vllm_config().model_config.hf_config.num_hidden_layers)
    except Exception:  # noqa: BLE001
        n = None
    if n is None:
        n = int(os.environ.get("EXL3_MTP_LAYER_START", "78"))
    return idx >= n
'''
anchor='\n@register_quantization_config("exl3")'
assert s.count(anchor)==1, s.count(anchor); s=s.replace(anchor, helper+anchor)
open(p,"w").write(s); print("mtp exemption patched")

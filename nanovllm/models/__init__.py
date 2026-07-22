from transformers import PretrainedConfig

from nanovllm.models.llama import LlamaForCausalLM
from nanovllm.models.qwen3 import Qwen3ForCausalLM

_MODEL_REGISTRY: dict[str, type] = {
    "qwen3": Qwen3ForCausalLM,
    "llama": LlamaForCausalLM,
}


def create_causal_lm(config: PretrainedConfig):
    model_type = getattr(config, "model_type", None)
    if model_type not in _MODEL_REGISTRY:
        supported = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model_type={model_type!r}. "
            f"Supported types: {supported}. "
            "Add a model module under nanovllm/models/ and register it in _MODEL_REGISTRY."
        )
    return _MODEL_REGISTRY[model_type](config)


def supported_model_types() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_REGISTRY))

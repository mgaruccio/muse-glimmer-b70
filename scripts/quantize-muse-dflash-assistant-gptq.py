#!/usr/bin/env python3
"""GPTQ W4A16 quantization for Muse Glimmer's DFlash assistant.

The assistant has no token embedding or lm_head of its own: vLLM's DFlash
runner shares both with the target model. We quantize exactly 35 decoder
linears (q/k/v/o + gate/up/down in five layers) and deliberately keep the
context encoder and all norms BF16.

Calibration is shape-correct deterministic DFlash activation data. A later
real-hidden-state calibration pass can replace this artifact, but must never
overwrite the BF16 source or this output path.
"""

import gc
import json
import shutil
import time
from pathlib import Path

import torch
from gptqmodel import QuantizeConfig
from gptqmodel.models._const import DEVICE
from gptqmodel.models.base import BaseQModel
from gptqmodel.utils.backend import BACKEND
from safetensors import safe_open
from transformers import AutoConfig, AutoModel

SOURCE = Path("/home/mike/inference-models/Muse-Glimmer-30B-assistant-BF16-source")
FINAL = Path("/home/mike/inference-models/Muse-Glimmer-30B-assistant-GPTQ-Int4-sym-G128")
PARTIAL = FINAL.with_name(FINAL.name + ".partial")
OFFLOAD = Path("/tmp/muse-dflash-assistant-gptq-offload")

CALIBRATION_COUNT = 256
CONTEXT_TOKENS = 32
NOISE_EMBED_STD = 0.02
SEED = 20260826
QUANTIZED_PER_LAYER = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
# vLLM DFlash fuses Q/K/V and gate/up and offsets layer indices by the target
# depth. AutoGPTQ matches substrings, so record the runner-side fused names
# rather than source checkpoint layer indices in the artifact metadata.
VLLM_FUSED_QUANT_MODULES = (
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "mlp.gate_up_proj",
    "mlp.down_proj",
)


class MuseGlimmerAssistantQModel(BaseQModel):
    """Explicit module tree for the root-level DFlash assistant architecture."""

    loader = AutoModel
    module_tree = [
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
                "q_norm:!",
                "k_norm:!",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]
    pre_lm_head_norm_module = "norm"

    @classmethod
    def extract_layers_node(cls):
        return ["layers"]

    @classmethod
    def get_base_modules(cls, model):
        # Root-level encoder/norm are outside the repeating layer container.
        # Materialize/offload them with the stages, but do not quantize them.
        return ["encoder", "norm", "rotary_emb"]

    def prepare_dataset(self, calibration_dataset, **_):
        """DFlash takes hidden-state tensors, not token IDs.

        GPTQModel's loop requires an `input_ids` key for progress accounting.
        Each batch provides a same-length zero placeholder; AssistantModel
        accepts and ignores it. The DFlash tensors remain the actual inputs.
        """
        return calibration_dataset


def make_calibration(config):
    """Create deterministic, shape-correct inputs for AssistantModel.forward.

    `noise_embeds` follows the normal token-embedding scale. Target hidden
    states are unit-RMS features concatenated from the five configured target
    layers; the DFlash context projection consumes this tensor.
    """
    target_layers = len(config.target_layer_ids)
    block_size = config.block_size
    hidden_size = config.hidden_size
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    sequence_length = CONTEXT_TOKENS + block_size
    samples = []
    for _ in range(CALIBRATION_COUNT):
        noise_embeds = torch.randn(
            (1, block_size, hidden_size), generator=generator, dtype=torch.float32
        ).mul_(NOISE_EMBED_STD).to(torch.bfloat16)
        context_hidden_states = torch.randn(
            (1, CONTEXT_TOKENS, hidden_size * target_layers),
            generator=generator,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        samples.append(
            {
                # GPTQModel requires this for token-count accounting only;
                # AssistantModel accepts and ignores it via **kwargs.
                "input_ids": torch.zeros((1, sequence_length), dtype=torch.long),
                "noise_embeds": noise_embeds,
                "context_hidden_states": context_hidden_states,
                "attention_mask": torch.ones((1, sequence_length), dtype=torch.long),
                "position_ids": torch.arange(sequence_length, dtype=torch.long).unsqueeze(0),
            }
        )
    return samples


def set_vllm_dflash_quant_modules(path):
    for name in ("config.json", "quantize_config.json"):
        config_path = path / name
        config = json.loads(config_path.read_text())
        qconfig = config["quantization_config"] if name == "config.json" else config
        qconfig["modules_in_block_to_quantize"] = list(VLLM_FUSED_QUANT_MODULES)
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def validate_artifact(path, expected_linears):
    safetensors = sorted(path.glob("*.safetensors"))
    if not safetensors:
        raise RuntimeError("save_quantized produced no safetensors files")
    keys = []
    for shard in safetensors:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            keys.extend(handle.keys())
    qweights = sorted(key for key in keys if key.endswith(".qweight"))
    if len(qweights) != expected_linears:
        raise RuntimeError(f"expected {expected_linears} quantized linears, got {len(qweights)}")
    if any(key.startswith("encoder.") for key in qweights):
        raise RuntimeError("encoder must remain BF16")
    required_bf16 = {"encoder.fc.weight", "encoder.output_norm_enc.weight", "norm.weight"}
    missing = required_bf16 - set(keys)
    if missing:
        raise RuntimeError(f"missing required BF16 tensors: {sorted(missing)}")
    qconfig = json.loads((path / "quantize_config.json").read_text())
    contract = {
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
        "format": "gptq",
    }
    actual = {key: qconfig.get(key) for key in contract}
    if actual != contract:
        raise RuntimeError(f"quantize contract mismatch: {actual}")
    if qconfig.get("modules_in_block_to_quantize") != list(VLLM_FUSED_QUANT_MODULES):
        raise RuntimeError("vLLM DFlash fused-module metadata mismatch")
    return {
        "qweight_count": len(qweights),
        "bf16_required": sorted(required_bf16),
        "vllm_fused_quant_modules": list(VLLM_FUSED_QUANT_MODULES),
    }


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if FINAL.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact: {FINAL}")
    for path in (PARTIAL, OFFLOAD):
        if path.exists():
            shutil.rmtree(path)
    OFFLOAD.mkdir(parents=True, exist_ok=True)
    FINAL.parent.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(str(SOURCE), local_files_only=True)
    expected_linears = config.num_hidden_layers * len(QUANTIZED_PER_LAYER)
    calibration = make_calibration(config)
    manifest = {
        "source": str(SOURCE),
        "output": str(FINAL),
        "architecture": config.architectures,
        "layers": config.num_hidden_layers,
        "expected_quantized_linears": expected_linears,
        "calibration": {
            "kind": "deterministic_shape_correct_synthetic_dflash",
            "count": CALIBRATION_COUNT,
            "context_tokens": CONTEXT_TOKENS,
            "block_size": config.block_size,
            "noise_embed_std": NOISE_EMBED_STD,
            "seed": SEED,
            "vllm_fused_quant_modules": list(VLLM_FUSED_QUANT_MODULES),
        },
    }
    print(json.dumps(manifest, sort_keys=True), flush=True)
    print("cuda", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
    print("torch", torch.__version__, "cuda_runtime", torch.version.cuda, flush=True)

    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        lm_head=False,
        method="gptq",
        format="gptq",
        pack_dtype=torch.int32,
        pack_impl="cpu",
        device=DEVICE.CUDA,
        offload_to_disk=True,
        offload_to_disk_path=str(OFFLOAD),
        calibration_data_device="cpu",
        act_group_aware=True,
    )
    print("quantize_config", qcfg.to_dict(), flush=True)
    model = MuseGlimmerAssistantQModel.from_pretrained(
        str(SOURCE),
        quantize_config=qcfg,
        backend=BACKEND.GPTQ_TORCH,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        tokenizer_trust_remote_code=False,
    )
    print("loaded assistant; starting GPTQ", flush=True)
    started = time.time()
    result = model.quantize(calibration=calibration, batch_size=1, backend=BACKEND.GPTQ_TORCH)
    print("quantize_result", json.dumps(result, default=str, sort_keys=True), flush=True)
    print("elapsed_sec", round(time.time() - started, 1), flush=True)

    model.save_quantized(
        str(PARTIAL),
        max_shard_size="4GB",
        safetensors_metadata={
            "quantizer": "gptqmodel:7.3.2",
            "format": "gptq",
            "calibration_kind": "deterministic_shape_correct_synthetic_dflash",
            "calibration_count": str(CALIBRATION_COUNT),
            "calibration_context_tokens": str(CONTEXT_TOKENS),
        },
    )
    set_vllm_dflash_quant_modules(PARTIAL)
    for name in ("README.md", "LICENSE", "USAGE_POLICY.md", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        source_file = SOURCE / name
        if source_file.is_file():
            shutil.copy2(source_file, PARTIAL / name)
    validation = validate_artifact(PARTIAL, expected_linears)
    manifest["validation"] = validation
    (PARTIAL / "quantization_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    PARTIAL.rename(FINAL)
    print("final_output", FINAL, flush=True)
    print("validation", json.dumps(validation, sort_keys=True), flush=True)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

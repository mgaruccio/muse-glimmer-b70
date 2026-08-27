#!/usr/bin/env python3
"""GPTQ W4A16 for the Muse Glimmer 30B *target*. CUDA, GPTQModel 7.3.2.

Not bit-identical to the Hub trees unless you match seed, dataset slice,
and GPU numerics. Prefer downloading the published GPTQ.
"""
import gc
import json
import os
import shutil
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoTokenizer
from gptqmodel import QuantizeConfig
from gptqmodel.models._const import DEVICE
from gptqmodel.models.base import BaseQModel
from gptqmodel.utils.backend import BACKEND

SOURCE = Path(os.environ.get("MUSE_TARGET_BF16", "models/Muse-Glimmer-30B"))
FINAL = Path(os.environ.get("MUSE_TARGET_GPTQ", "models/Muse-Glimmer-30B-GPTQ-Int4-sym-G128"))
PARTIAL = FINAL.with_name(FINAL.name + '.partial')
OFFLOAD = Path('/tmp/muse-glimmer-gptq-offload')
CALIBRATION_COUNT = 512
BODY_TOKENS = 1850

class MuseGlimmerQModel(BaseQModel):
    # Muse Glimmer is multimodal, but only this decoder layer tree is quantized.
    loader = AutoModelForImageTextToText
    module_tree = [
        'model',
        'language_model',
        'layers',
        '#',
        {
            'input_layernorm': ('input_layernorm:!',),
            'self_attn': (
                'q_proj:0', 'k_proj:0', 'v_proj:0', 'o_proj:1',
                'gate_proj:0', 'q_norm:!', 'k_norm:!',
            ),
            'post_attention_layernorm': ('post_attention_layernorm:!',),
            'pre_feedforward_layernorm': ('pre_feedforward_layernorm:!',),
            'mlp': ('gate_proj:0', 'up_proj:0', 'down_proj:1'),
            'post_feedforward_layernorm': ('post_feedforward_layernorm:!',),
        },
    ]
    pre_lm_head_norm_module = 'model.language_model.norm'

    @classmethod
    def extract_layers_node(cls):
        return ['model.language_model.layers']


def make_calibration(tokenizer):
    # Wikitext-2 is used only as plain text; each sample is rendered through the
    # Muse chat template before token ids are handed to GPTQModel.
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    raw_tokens = []
    for row in ds:
        text = row['text'].strip()
        if not text:
            continue
        raw_tokens.extend(tokenizer(text, add_special_tokens=False)['input_ids'])
        if len(raw_tokens) >= CALIBRATION_COUNT * BODY_TOKENS:
            break
    if len(raw_tokens) < CALIBRATION_COUNT * BODY_TOKENS:
        raise RuntimeError(f'not enough calibration tokens: {len(raw_tokens)}')

    samples = []
    cursor = 0
    for _ in range(CALIBRATION_COUNT):
        body_ids = raw_tokens[cursor:cursor + BODY_TOKENS]
        cursor += BODY_TOKENS
        body = tokenizer.decode(body_ids, skip_special_tokens=True)
        encoded = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': body}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt',
        )
        ids = encoded['input_ids'][0].to(dtype=torch.long).tolist()
        if len(ids) < 256:
            raise RuntimeError(f'calibration sample too short: {len(ids)}')
        samples.append({'input_ids': ids, 'attention_mask': [1] * len(ids)})
    lengths = [len(x['input_ids']) for x in samples]
    print(json.dumps({
        'calibration_count': len(samples),
        'token_lengths': {'min': min(lengths), 'max': max(lengths), 'avg': sum(lengths) / len(lengths)},
        'template': 'Muse Glimmer chat_template.jinja via tokenizer.apply_chat_template',
        'dataset': 'Salesforce/wikitext:wikitext-2-raw-v1:train',
    }, sort_keys=True), flush=True)
    return samples


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if PARTIAL.exists():
        shutil.rmtree(PARTIAL)
    if OFFLOAD.exists():
        shutil.rmtree(OFFLOAD)
    OFFLOAD.mkdir(parents=True, exist_ok=True)
    FINAL.parent.mkdir(parents=True, exist_ok=True)

    print('cuda', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
    print('torch', torch.__version__, 'cuda_runtime', torch.version.cuda, flush=True)
    print('source', SOURCE, flush=True)
    print('partial_output', PARTIAL, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True)
    calibration = make_calibration(tokenizer)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        lm_head=False,
        method='gptq',
        format='gptq',
        pack_dtype=torch.int32,
        pack_impl='cpu',
        device=DEVICE.CUDA,
        offload_to_disk=True,
        offload_to_disk_path=str(OFFLOAD),
        calibration_data_device='cpu',
        # Keep the 7.3.2 quality path while retaining the requested desc_act=False contract.
        act_group_aware=True,
    )
    print('quantize_config', qcfg.to_dict(), flush=True)
    model = MuseGlimmerQModel.from_pretrained(
        str(SOURCE),
        quantize_config=qcfg,
        backend=BACKEND.GPTQ_TORCH,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        tokenizer_trust_remote_code=False,
    )
    print('loaded model; starting GPTQ', flush=True)
    t0 = time.time()
    result = model.quantize(
        calibration=calibration,
        batch_size=1,
        tokenizer=tokenizer,
        backend=BACKEND.GPTQ_TORCH,
    )
    print('quantize_result_layers', len(result), 'elapsed_sec', round(time.time() - t0, 1), flush=True)
    model.save_quantized(
        str(PARTIAL),
        max_shard_size='4GB',
        safetensors_metadata={
            'quantizer': 'gptqmodel:7.3.2',
            'format': 'gptq',
            'calibration_examples': str(CALIBRATION_COUNT),
            'calibration_tokens_target': str(BODY_TOKENS),
            'calibration_template': 'Muse Glimmer chat_template.jinja',
        },
    )

    # BaseQModel does not load a processor for text-only calibration. Preserve the
    # multimodal processor and the standalone Muse chat template verbatim.
    for name in ('processor_config.json', 'chat_template.jinja'):
        src = SOURCE / name
        if src.is_file():
            shutil.copy2(src, PARTIAL / name)
    # Keep the original model license/policy metadata when present.
    for name in ('README.md', 'LICENSE', 'USAGE_POLICY.md'):
        src = SOURCE / name
        if src.is_file():
            shutil.copy2(src, PARTIAL / name)

    # Rename only after the writer completed; validation is performed by the
    # follow-up validation command against the final directory.
    PARTIAL.rename(FINAL)
    print('final_output', FINAL, flush=True)
    print('output_files', sorted(p.name for p in FINAL.iterdir()), flush=True)
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

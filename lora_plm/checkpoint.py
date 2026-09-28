"""One checked model-loading path for prediction and beam search."""
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel

try:
    from .model import load_encoder, SeqRegressor, MultiSiteAttentionRegressor, SixSiteAttentionRegressor
    from .utils import load_pos_map
except ImportError:
    from model import load_encoder, SeqRegressor, MultiSiteAttentionRegressor, SixSiteAttentionRegressor
    from utils import load_pos_map


@dataclass(frozen=True)
class CheckpointSpec:
    head: str
    indexing: str
    ref_positions: tuple
    enzyme_name: str
    attn_heads: int = 4
    attn_layers: int = 1
    attn_dropout: float = 0.1
    attn_ff_mult: int = 2


def read_checkpoint_spec(directory, head='auto', indexing='auto'):
    path = Path(directory) / 'meta.txt'
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint metadata is required: {path}')
    meta = dict(line.strip().split('=', 1) for line in path.read_text(encoding='utf-8').splitlines()
                if '=' in line and not line.lstrip().startswith('#'))
    required = {'head', 'ref_positions', 'enzyme_name'}
    missing = required - meta.keys()
    if missing:
        raise ValueError(f'Checkpoint metadata is missing {sorted(missing)}')
    if meta['head'] not in ('meanpool', 'site_attn', 'sixsite_attn'):
        raise ValueError(f"Unsupported checkpoint head: {meta['head']}")
    if head != 'auto' and head != meta['head']:
        raise ValueError(f"Requested head {head} differs from saved head {meta['head']}")
    saved_indexing = meta.get('indexing')
    if saved_indexing is None:
        if indexing == 'auto':
            raise ValueError('Checkpoint has no indexing metadata; specify its verified convention explicitly')
        saved_indexing = indexing
    if saved_indexing not in ('one_based', 'legacy_shifted'):
        raise ValueError(f'Unsupported checkpoint indexing: {saved_indexing}')
    if indexing != 'auto' and indexing != saved_indexing:
        raise ValueError('Requested indexing differs from checkpoint metadata; use a matching checkpoint')
    if meta['head'] != 'meanpool':
        arch = {'attn_heads', 'attn_layers', 'attn_dropout', 'attn_ff_mult'}
        if arch - meta.keys():
            raise ValueError(f'Missing attention architecture: {sorted(arch - meta.keys())}')
    spec = CheckpointSpec(
        head=meta['head'], indexing=saved_indexing,
        ref_positions=tuple(int(x) for x in meta['ref_positions'].split(',')),
        enzyme_name=meta['enzyme_name'], attn_heads=int(meta.get('attn_heads', 4)),
        attn_layers=int(meta.get('attn_layers', 1)),
        attn_dropout=float(meta.get('attn_dropout', 0.1)),
        attn_ff_mult=int(meta.get('attn_ff_mult', 2)))
    if not spec.ref_positions or len(set(spec.ref_positions)) != len(spec.ref_positions):
        raise ValueError('Checkpoint reference positions must be nonempty and unique')
    if min(spec.attn_heads, spec.attn_layers, spec.attn_ff_mult) < 1:
        raise ValueError('Invalid attention dimensions')
    return spec


def load_state(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required checkpoint component is missing: {path}')
    with path.open('rb') as stream:
        if stream.read(42).startswith(b'version https://git-lfs.github.com/spec'):
            raise RuntimeError(f'{path} is a Git LFS pointer. Run git lfs pull first.')
    return torch.load(path, map_location='cpu', weights_only=True)


def load_regressor(model_path, checkpoint_dir, crossmap, enzyme_name, ref_positions,
                   *, head='auto', indexing='auto', local_files_only=False,
                   trust_remote_code=False, device='cpu'):
    directory = Path(checkpoint_dir)
    spec = read_checkpoint_spec(directory, head=head, indexing=indexing)
    if tuple(ref_positions) != spec.ref_positions or enzyme_name != spec.enzyme_name:
        raise ValueError('Requested protein or ordered sites differ from checkpoint metadata')
    mapping = load_pos_map(crossmap, enzyme_name, ref_positions, indexing=spec.indexing)
    # Validate all small/required components before loading the language model.
    regression = load_state(directory / 'reg_head.pt')
    attention = load_state(directory / 'site_encoder.pt') if spec.head != 'meanpool' else None
    config_path = directory / 'adapter_config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing LoRA configuration: {config_path}')
    raw = json.loads(config_path.read_text(encoding='utf-8'))
    allowed = set(inspect.signature(LoraConfig.__init__).parameters) - {'self'}
    lora_config = LoraConfig(**{k: v for k, v in raw.items() if k in allowed})
    loaded = load_encoder(model_path, local_files_only=local_files_only,
                          trust_remote_code=trust_remote_code)
    encoder = PeftModel.from_pretrained(loaded.encoder, str(directory), config=lora_config,
                                        local_files_only=local_files_only)
    if spec.head == 'meanpool':
        model = SeqRegressor(encoder, loaded.hidden_size)
    else:
        cls = SixSiteAttentionRegressor if spec.head == 'sixsite_attn' else MultiSiteAttentionRegressor
        model = cls(encoder, loaded.hidden_size, [mapping[p] for p in ref_positions],
                    n_heads=spec.attn_heads, n_layers=spec.attn_layers,
                    ff_mult=spec.attn_ff_mult, dropout=spec.attn_dropout)
        model.site_encoder.load_state_dict(attention, strict=True)
    model.reg_head.load_state_dict(regression, strict=True)
    model.to(device).eval()
    return loaded.tokenizer, model, mapping, spec

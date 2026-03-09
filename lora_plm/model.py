from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model


@dataclass
class EncoderWithTokenizer:
    encoder: nn.Module
    tokenizer: AutoTokenizer
    hidden_size: int
    is_mlm_wrapper: bool
    parent_for_peft: nn.Module


def load_encoder(model_path: str,
                 local_files_only: bool = False,
                 trust_remote_code: bool = False) -> EncoderWithTokenizer:
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only, trust_remote_code=trust_remote_code)
    hidden_size = None
    is_mlm = False
    parent = None
    try:
        enc = AutoModel.from_pretrained(model_path, local_files_only=local_files_only, trust_remote_code=trust_remote_code)
        parent = enc
        hidden_size = getattr(enc.config, 'hidden_size', None)
    except Exception:
        mlm = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=local_files_only, trust_remote_code=trust_remote_code)
        # ESM masked LM wrapper has .esm as the encoder
        enc = mlm.esm
        parent = mlm
        hidden_size = getattr(enc.config, 'hidden_size', None) or getattr(mlm.config, 'hidden_size', None)
        is_mlm = True
    if hidden_size is None:
        hidden_size = 768
    return EncoderWithTokenizer(encoder=enc, tokenizer=tok, hidden_size=hidden_size, is_mlm_wrapper=is_mlm, parent_for_peft=parent)


class SeqRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.reg_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids=None, attention_mask=None, labels: Optional[torch.Tensor] = None):
        # Some PEFT versions set PeftModel.config as a dict, which breaks
        # PeftModel.forward("return_dict" resolution). Call the wrapped base
        # model directly when available to avoid relying on PeftModel.forward.
        core = getattr(self.encoder, "base_model", self.encoder)
        outputs = core(input_ids=input_ids,
                       attention_mask=attention_mask,
                       output_hidden_states=False,
                       return_dict=True)
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1e-6)
            pooled = summed / denom
        else:
            pooled = last_hidden[:, 0, :]
        pred = self.reg_head(pooled).squeeze(-1)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(pred, labels)
        return {"loss": loss, "logits": pred}


class MultiSiteAttentionRegressor(nn.Module):
    """Regressor that attends over arbitrary site tokens to learn cross-position interactions."""

    def __init__(self, encoder: nn.Module, hidden_size: int,
                 site_seq_positions: List[int], n_heads: int = 4,
                 n_layers: int = 1, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        if len(site_seq_positions) == 0:
            raise ValueError("site_seq_positions must be non-empty")
        self.encoder = encoder
        self.site_pos = site_seq_positions  # list of ints (1-based)
        d_model = hidden_size
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               dim_feedforward=d_model * ff_mult,
                                               dropout=dropout, batch_first=True)
        self.site_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids=None, attention_mask=None):
        core = getattr(self.encoder, "base_model", self.encoder)
        outputs = core(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=False, return_dict=True)
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        idx = torch.tensor([p + 1 for p in self.site_pos], device=last_hidden.device)
        idx = idx.view(1, -1).expand(last_hidden.size(0), -1)
        B, L, H = last_hidden.size()
        idx = idx.clamp(min=0, max=L - 1)
        gathered = last_hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))
        z = self.site_encoder(gathered)
        pooled = z.mean(dim=1)
        pred = self.reg_head(pooled).squeeze(-1)
        return {"logits": pred}


class SixSiteAttentionRegressor(MultiSiteAttentionRegressor):
    """Backward compatible wrapper that enforces exactly six sites."""

    def __init__(self, encoder: nn.Module, hidden_size: int,
                 site_seq_positions: List[int], n_heads: int = 4,
                 n_layers: int = 1, ff_mult: int = 2, dropout: float = 0.1):
        if len(site_seq_positions) != 6:
            raise ValueError("SixSiteAttentionRegressor requires exactly 6 site positions.")
        super().__init__(encoder, hidden_size, site_seq_positions, n_heads, n_layers, ff_mult, dropout)


def _auto_detect_lora_targets(model: nn.Module) -> List[str]:
    """Find likely attention projection module names for LoRA.
    Strategy:
      1) Collect names of all Linear submodules
      2) Prefer names containing ('attn' or 'attention') and any of
         ['q_proj','k_proj','v_proj','o_proj','out_proj','query','key','value','proj','projection','dense']
      3) Fallback: if still empty, return a small set of linear names to avoid failure
    Returns a list of unique short names (leaf module names) compatible with PEFT.
    """
    import torch.nn as nn
    linear_names = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            linear_names.append(name)
    # reduce to leaf names
    leaf = [n.split('.')[-1] for n in linear_names]
    # score by patterns
    preferred = []
    for full in linear_names:
        low = full.lower()
        if ("attn" in low or "attention" in low) and any(p in low for p in [
            'q_proj','k_proj','v_proj','o_proj','out_proj','query','key','value','proj','projection','dense'
        ]):
            preferred.append(full.split('.')[-1])
    preferred = list(dict.fromkeys(preferred))
    if preferred:
        return preferred
    # fallback: take frequent leaf names among linear layers
    if leaf:
        # keep unique but limited set to avoid covering every Linear
        uniq = list(dict.fromkeys(leaf))
        return uniq[:8]
    # last resort
    return ["out_proj"]


def attach_lora(model: nn.Module,
                target_modules: Optional[List[str]] = None,
                r: int = 8,
                lora_alpha: int = 16,
                lora_dropout: float = 0.05,
                auto: bool = True) -> nn.Module:
    # choose targets
    if target_modules is None and auto:
        target_modules = _auto_detect_lora_targets(model)
        if not target_modules:
            target_modules = ["out_proj"]
    elif target_modules is None:
        # default ESM2 naming
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="FEATURE_EXTRACTION"
    )
    try:
        lora_model = get_peft_model(model, config)
        return lora_model
    except ValueError as e:
        # retry with auto-detected names if first attempt failed
        auto_targets = _auto_detect_lora_targets(model)
        if auto_targets:
            config.target_modules = auto_targets
            lora_model = get_peft_model(model, config)
            return lora_model
        raise

import os
from typing import Dict, List, Tuple

import pandas as pd


def read_fasta_first_seq(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"WT FASTA not found: {path}")
    seq = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if seq:
                    break
                continue
            seq.append(line)
    s = ''.join(seq).strip()
    if not s:
        raise RuntimeError(f"No sequence parsed from {path}")
    return s


def load_pos_map(crossmap_csv: str, enzyme_name: str, ref_positions: List[int]) -> Dict[int, int]:
    df = pd.read_csv(crossmap_csv)
    if 'enzyme_name' not in df.columns:
        raise ValueError("crossmap CSV missing 'enzyme_name' column. Regenerate with enzyme_map.")
    df = df[(df['enzyme_name'] == enzyme_name) & df['seq_pos'].notna()]
    r2s = {int(r.ref_pos): int(r.seq_pos) for _, r in df.iterrows()}
    missing = [rp for rp in ref_positions if rp not in r2s]
    if missing:
        raise ValueError(f"Crossmap missing positions {missing} for enzyme {enzyme_name}")
    return r2s


def apply_combo_to_wt(wt_seq: str, ref_positions: List[int], r2s: Dict[int, int], combo: str) -> str:
    if len(combo) != len(ref_positions):
        raise ValueError(f"Combo length {len(combo)} != num positions {len(ref_positions)}")
    s_list = list(wt_seq)
    for i, rp in enumerate(ref_positions):
        si = r2s[rp]
        s_list[si] = combo[i]
    return ''.join(s_list)


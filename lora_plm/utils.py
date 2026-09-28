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


def load_pos_map(crossmap_csv: str, enzyme_name: str, ref_positions: List[int],
                 indexing: str = 'one_based') -> Dict[int, int]:
    """Convert one-based crossmap positions to zero-based Python indices.

    ``legacy_shifted`` is reserved for explicitly labelled historical checkpoints.
    It preserves their original one-residue offset; it is not the default for
    new training or biological sequence construction.
    """
    if indexing not in ('one_based', 'legacy_shifted'):
        raise ValueError(f"Unknown indexing convention: {indexing}")
    if len(set(ref_positions)) != len(ref_positions):
        raise ValueError('Reference positions must be unique')
    df = pd.read_csv(crossmap_csv)
    required = {'enzyme_name', 'ref_pos', 'seq_pos'}
    if not required.issubset(df.columns):
        raise ValueError(f"Crossmap missing columns: {sorted(required - set(df.columns))}")
    df = df[(df['enzyme_name'] == enzyme_name) & df['seq_pos'].notna()]
    df = df[df['ref_pos'].isin(ref_positions)]
    if df['ref_pos'].duplicated().any():
        raise ValueError(f'Duplicate reference positions for {enzyme_name}')
    positions = pd.to_numeric(df['seq_pos'], errors='raise')
    if ((positions < 1) | (positions % 1 != 0)).any():
        raise ValueError('Crossmap seq_pos must contain positive one-based integers')
    offset = 1 if indexing == 'one_based' else 0
    r2s = {int(r.ref_pos): int(r.seq_pos) - offset for _, r in df.iterrows()}
    missing = [rp for rp in ref_positions if rp not in r2s]
    if missing:
        raise ValueError(f"Crossmap missing positions {missing} for enzyme {enzyme_name}")
    return r2s


def apply_partial_to_wt(wt_seq: str, ref_positions: List[int],
                        r2s: Dict[int, int], letters) -> str:
    """Apply a full or partial assignment using zero-based sequence indices."""
    if len(letters) != len(ref_positions):
        raise ValueError(f"Combo length {len(letters)} != num positions {len(ref_positions)}")
    s_list = list(wt_seq)
    for rp, letter in zip(ref_positions, letters):
        index = r2s[rp]
        if index < 0 or index >= len(s_list):
            raise ValueError(f'Mapped position {rp} is outside the input sequence')
        if letter is not None:
            if letter not in 'ACDEFGHIKLMNPQRSTVWY' or len(letter) != 1:
                raise ValueError(f'Invalid amino-acid assignment: {letter!r}')
            s_list[index] = letter
    return ''.join(s_list)


def apply_combo_to_wt(wt_seq: str, ref_positions: List[int], r2s: Dict[int, int], combo: str) -> str:
    return apply_partial_to_wt(wt_seq, ref_positions, r2s, combo)


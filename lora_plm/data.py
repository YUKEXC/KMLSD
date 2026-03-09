from typing import List, Dict
import pandas as pd
from torch.utils.data import Dataset

try:
    from lora_plm.utils import apply_combo_to_wt
except Exception:
    from utils import apply_combo_to_wt


class ComboRegressionDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 wt_seq: str,
                 ref_positions: List[int],
                 r2s: Dict[int, int],
                 obj_col: str,
                 weight_col: str | None = None):
        if 'Combo' not in df.columns:
            raise ValueError("DataFrame must contain 'Combo' column")
        if obj_col not in df.columns:
            raise ValueError(f"DataFrame missing target column {obj_col}")
        self.combos = df['Combo'].astype(str).tolist()
        self.y = pd.to_numeric(df[obj_col], errors='coerce').fillna(0.0).astype(float).tolist()
        # optional sample weights
        if weight_col and weight_col in df.columns:
            self.w = pd.to_numeric(df[weight_col], errors='coerce').fillna(1.0).astype(float).tolist()
        else:
            self.w = [1.0] * len(self.combos)
        self.wt_seq = wt_seq
        self.ref_positions = ref_positions
        self.r2s = r2s

    def __len__(self):
        return len(self.combos)

    def __getitem__(self, idx):
        combo = self.combos[idx]
        y = float(self.y[idx])
        seq = apply_combo_to_wt(self.wt_seq, self.ref_positions, self.r2s, combo)
        w = float(self.w[idx])
        return {"seq": seq, "label": y, "weight": w}

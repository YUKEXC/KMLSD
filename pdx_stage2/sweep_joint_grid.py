#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid builder/runner for PDX Stage-II:
- objective ratios between yield and selectivity
- attention heads in {2,4}
- attention layers in {1,2}

By default:
- prepares per-ratio data (fast)
- writes all train/beam commands to a txt file
- does NOT execute heavy training unless --run_train is provided
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def parse_float_list(s: str) -> List[float]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def parse_int_list(s: str) -> List[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def ratio_tag(wy: float, ws: float) -> str:
    return f"y{int(round(wy * 100)):02d}_s{int(round(ws * 100)):02d}"


def cmd_to_str(cmd: List[str]) -> str:
    return " ".join(cmd)


def run_cmd(cmd: List[str], run: bool) -> None:
    print(cmd_to_str(cmd))
    if run:
        env = os.environ.copy()
        cwd = os.getcwd()
        cur_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{cwd}{os.pathsep}{cur_pp}" if cur_pp else cwd
        subprocess.run(cmd, check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="pdx_stage2/raw/ML.csv")
    ap.add_argument("--prepare_script", default="pdx_stage2/prepare_pdx_stage2_data.py")
    ap.add_argument("--enzyme_name", default="PDX")
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--yield_weights", default="0.2,0.4,0.5,0.6,0.8")
    ap.add_argument("--attn_heads", default="2,4")
    ap.add_argument("--attn_layers", default="1,2")

    ap.add_argument("--model_path", default="model/esm2_650M")
    ap.add_argument("--obj_col", default="ObjJointZ")
    ap.add_argument("--weight_col", default="Weight")
    ap.add_argument("--rank_loss_weight", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn_dropout", type=float, default=0.1)
    ap.add_argument("--attn_ff_mult", type=int, default=2)

    ap.add_argument("--base_work_dir", default="pdx_stage2/sweeps")
    ap.add_argument("--results_prefix", default="results/lora_plm/pdx_stage2_grid")
    ap.add_argument("--run_train", action="store_true")
    ap.add_argument("--run_beam", action="store_true")
    args = ap.parse_args()

    y_weights = parse_float_list(args.yield_weights)
    heads = parse_int_list(args.attn_heads)
    layers = parse_int_list(args.attn_layers)

    base_work_dir = Path(args.base_work_dir)
    base_work_dir.mkdir(parents=True, exist_ok=True)
    cmd_file = base_work_dir / "grid_commands.txt"

    lines: List[str] = []
    lines.append("# Auto-generated grid commands")
    lines.append(
        f"# yield_weights={y_weights}, attn_heads={heads}, attn_layers={layers}, run_train={args.run_train}, run_beam={args.run_beam}"
    )
    lines.append("")

    for wy in y_weights:
        ws = 1.0 - wy
        tag = ratio_tag(wy, ws)
        ratio_dir = base_work_dir / tag
        artifacts_dir = ratio_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Always prepare data for this ratio (fast and needed to get selected positions)
        prep_cmd = [
            "python",
            args.prepare_script,
            "--input_csv",
            args.input_csv,
            "--out_dir",
            str(artifacts_dir),
            "--enzyme_name",
            args.enzyme_name,
            "--top_k",
            str(args.top_k),
            "--w_yield",
            str(wy),
            "--w_selectivity",
            str(ws),
        ]
        run_cmd(prep_cmd, run=True)

        # IMPORTANT: use crossmap order (same order used by stage2_train Combo encoding)
        crossmap_path = artifacts_dir / "refpos_crossmap.csv"
        crossmap_df = pd.read_csv(crossmap_path)
        ref_positions = [int(x) for x in crossmap_df["ref_pos"].tolist()]
        ref_positions_str = ",".join(map(str, ref_positions))
        head_mode = "sixsite_attn" if len(ref_positions) == 6 else "site_attn"

        wt_fasta = artifacts_dir / "PDX_WT.fasta"
        crossmap = crossmap_path
        train_csv = artifacts_dir / "stage2_train.csv"

        lines.append(f"# ===== Ratio {tag} (yield={wy:.2f}, selectivity={ws:.2f}) =====")
        lines.append(f"# selected_ref_positions={ref_positions_str}")
        lines.append("")

        for h in heads:
            for l in layers:
                run_name = f"{args.results_prefix}_{tag}_h{h}_l{l}"

                train_cmd = [
                    "python",
                    "lora_plm/train.py",
                    "--model_path",
                    args.model_path,
                    "--wt_fasta",
                    str(wt_fasta),
                    "--crossmap",
                    str(crossmap),
                    "--enzyme_name",
                    args.enzyme_name,
                    "--ref_positions",
                    ref_positions_str,
                    "--train_csv",
                    str(train_csv),
                    "--obj_col",
                    args.obj_col,
                    "--weight_col",
                    args.weight_col,
                    "--head",
                    head_mode,
                    "--attn_heads",
                    str(h),
                    "--attn_layers",
                    str(l),
                    "--attn_dropout",
                    str(args.attn_dropout),
                    "--attn_ff_mult",
                    str(args.attn_ff_mult),
                    "--rank_loss_weight",
                    str(args.rank_loss_weight),
                    "--epochs",
                    str(args.epochs),
                    "--batch_size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--out_dir",
                    run_name,
                    "--device",
                    args.device,
                    "--local_files_only",
                    "--trust_remote_code",
                ]
                lines.append(cmd_to_str(train_cmd))
                run_cmd(train_cmd, run=args.run_train)

                if args.run_beam:
                    beam_cmd = [
                        "python",
                        "beam/beam_search_lora.py",
                        "--model_path",
                        args.model_path,
                        "--peft_dir",
                        run_name,
                        "--wt_fasta",
                        str(wt_fasta),
                        "--crossmap",
                        str(crossmap),
                        "--enzyme_name",
                        args.enzyme_name,
                        "--ref_positions",
                        ref_positions_str,
                        "--out_dir",
                        f"{run_name}/beam",
                        "--beam",
                        "256",
                        "--epsilon",
                        "0.05",
                        "--diversity_dmin",
                        "0",
                        "--seeds_from_singles",
                        "400",
                        "--batch_size",
                        "128",
                        "--device",
                        args.device,
                        "--local_files_only",
                        "--trust_remote_code",
                        "--head",
                        "auto",
                    ]
                    lines.append(cmd_to_str(beam_cmd))
                    run_cmd(beam_cmd, run=True)

                lines.append("")

    cmd_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[DONE] Grid commands written to: {cmd_file}")
    if not args.run_train:
        print("[INFO] Training was not executed. Use --run_train to start actual training.")


if __name__ == "__main__":
    main()

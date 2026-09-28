"""Recompute the manuscript's GB1 top-10 metrics from a frozen candidate pool."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def evaluate(predictions, truth, training, k=10):
    if k < 1:
        raise ValueError('k must be positive')
    truth = truth[['Variants', 'Fitness']].rename(columns={'Variants': 'Combo'})
    if truth.Combo.duplicated().any() or predictions.Combo.duplicated().any():
        raise ValueError('Duplicate combinations would change the evaluation population')
    ranked = truth.sort_values('Fitness', ascending=False).reset_index(drop=True)
    ranked['true_rank'] = np.arange(1, len(ranked) + 1)
    selected = predictions[['Combo', 'y_pred']].merge(ranked, on='Combo', how='inner', validate='1:1')
    selected = selected[~selected.Combo.isin(set(training.Combo))]
    selected = selected.sort_values('y_pred', ascending=False).head(k).reset_index(drop=True)
    if len(selected) != k:
        raise ValueError(f'Only {len(selected)} eligible candidates; expected {k}')
    y = selected.Fitness.to_numpy(float)
    pred = selected.y_pred.to_numpy(float)
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Non-finite fitness or predictions')
    relevance = y - y.min()
    discount = np.log2(np.arange(2, k + 2))
    ideal = np.sum(np.sort(relevance)[::-1] / discount)
    gain = np.sum(relevance[np.argsort(pred)[::-1]] / discount)
    best = selected.loc[selected.Fitness.idxmax()]
    metrics = {
        'mse': float(np.mean((y - pred) ** 2)),
        'spearman': float(spearmanr(y, pred).correlation),
        f'ndcg@{k}': float(gain / ideal) if ideal > 0 else None,
        'n': len(selected),
        f'avg_top{k}_true_rank': float(selected.true_rank.mean()),
        f'best_top{k}_true_rank': int(best.true_rank),
        f'best_top{k}_true_combo': best.Combo,
        f'best_top{k}_true_fitness': float(best.Fitness),
    }
    selected.insert(0, 'predicted_slot', np.arange(1, k + 1))
    return metrics, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=ROOT / 'results/lora_plm/gb1_beam/beam_all_final.csv')
    parser.add_argument('--truth', type=Path, default=ROOT / 'data/GB1/GB1.CSV')
    parser.add_argument('--training', type=Path, default=ROOT / 'data/GB1/gb1_stage2_train.csv')
    parser.add_argument('--out_dir', type=Path, default=ROOT / 'outputs/gb1_evaluation')
    parser.add_argument('--verify-reference', action='store_true')
    args = parser.parse_args()
    metrics, selected = evaluate(pd.read_csv(args.predictions), pd.read_csv(args.truth), pd.read_csv(args.training))
    if args.verify_reference:
        reference = json.loads((ROOT / 'results/lora_plm/gb1_beam/metrics.json').read_text())
        for key, value in reference.items():
            if isinstance(value, (int, float)):
                if not np.isclose(metrics[key], value, rtol=1e-10, atol=1e-10):
                    raise AssertionError(f'{key}: {metrics[key]} != archived {value}')
            elif metrics[key] != value:
                raise AssertionError(f'{key}: {metrics[key]} != archived {value}')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    selected.to_csv(args.out_dir / 'evaluated_top10.csv', index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

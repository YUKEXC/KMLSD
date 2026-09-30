"""Offline regressions for indexing, checkpoint loading and manuscript evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import EsmConfig, EsmModel, EsmTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lora_plm.utils import read_fasta_first_seq, load_pos_map, apply_combo_to_wt
from lora_plm.model import MultiSiteAttentionRegressor, attach_lora
from lora_plm.checkpoint import load_regressor, load_state, read_checkpoint_spec
from beam.beam_search_lora import batch_predict, apply_letters_to_wt
from scripts.evaluate_gb1 import evaluate

torch.set_num_threads(2)


class MappingTests(unittest.TestCase):
    def test_wild_type_is_unchanged_and_single_mutation_is_at_requested_site(self):
        for enzyme, positions, wt_path, crossmap, combo in [
            ('GB1', [39, 40, 41, 54], 'data/GB1/GB1_WT.fasta', 'data/GB1/gb1_refpos_crossmap.csv', 'VDGV'),
            ('CYP107D1', [68, 96, 173, 192, 294, 296], 'WT.fasta', 'stage1_data/p450/refpos_crossmap.csv', 'SQTVGF'),
        ]:
            with self.subTest(enzyme=enzyme):
                wt = read_fasta_first_seq(ROOT / wt_path)
                mapping = load_pos_map(ROOT / crossmap, enzyme, positions)
                self.assertEqual(apply_combo_to_wt(wt, positions, mapping, combo), wt)
                mutant = apply_combo_to_wt(wt, positions, mapping, 'A' + combo[1:])
                self.assertEqual([i for i, (a, b) in enumerate(zip(wt, mutant)) if a != b], [mapping[positions[0]]])
                self.assertEqual(apply_letters_to_wt(wt, positions, mapping, [None] * len(positions)), wt)
                legacy = load_pos_map(ROOT / crossmap, enzyme, positions, indexing='legacy_shifted')
                self.assertNotEqual(apply_combo_to_wt(wt, positions, legacy, combo), wt)
        with self.assertRaises(ValueError):
            apply_combo_to_wt('ACD', [1], {1: 0}, 'AA')
        with self.assertRaises(ValueError):
            apply_combo_to_wt('ACD', [4], {4: 3}, 'A')

    def test_attention_uses_residue_tokens_and_rejects_truncation(self):
        class Encoder(nn.Module):
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1))
        model = MultiSiteAttentionRegressor(Encoder(), 1, [0, 2], n_heads=1, dropout=0)
        model.site_encoder = nn.Identity()
        model.reg_head = nn.Identity()
        ids = torch.tensor([[100, 2, 4, 6, 200]])  # BOS, residues, EOS
        self.assertEqual(model(input_ids=ids, attention_mask=torch.ones_like(ids))['logits'].item(), 4)
        with self.assertRaises(ValueError):
            model(input_ids=ids[:, :4], attention_mask=torch.ones_like(ids[:, :4]))


class CheckpointTests(unittest.TestCase):
    def test_saved_two_layer_head_is_loaded_by_both_prediction_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            base = folder / 'base'; base.mkdir()
            checkpoint = folder / 'checkpoint'; checkpoint.mkdir()
            vocab = ROOT / 'results/lora_plm/gb1_siteattn/vocab.txt'
            tokenizer = EsmTokenizer(vocab_file=str(vocab))
            tokenizer.save_pretrained(base)
            config = EsmConfig(vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=16, pad_token_id=1,
                               mask_token_id=32, max_position_embeddings=128, token_dropout=False)
            encoder = EsmModel(config)
            encoder.save_pretrained(base)
            adapted = attach_lora(encoder, target_modules=['query', 'value'], r=2)
            model = MultiSiteAttentionRegressor(adapted, 8, [38, 39, 40, 53], n_heads=2, n_layers=2).eval()
            adapted.save_pretrained(checkpoint)
            torch.save(model.site_encoder.state_dict(), checkpoint / 'site_encoder.pt')
            torch.save(model.reg_head.state_dict(), checkpoint / 'reg_head.pt')
            (checkpoint / 'meta.txt').write_text('ref_positions=39,40,41,54\nenzyme_name=GB1\nhead=site_attn\nindexing=one_based\nattn_heads=2\nattn_layers=2\nattn_dropout=0.1\nattn_ff_mult=2\n')
            tok, loaded, mapping, spec = load_regressor(str(base), str(checkpoint), ROOT / 'data/GB1/gb1_refpos_crossmap.csv',
                                                       'GB1', [39, 40, 41, 54], head='site_attn', local_files_only=True)
            self.assertEqual(spec.attn_layers, 2)
            self.assertEqual(len(loaded.site_encoder.layers), 2)
            wt = read_fasta_first_seq(ROOT / 'data/GB1/GB1_WT.fasta')
            combos = ['VDGV', 'ADGV', 'LYGV']
            seqs = [apply_combo_to_wt(wt, [39, 40, 41, 54], mapping, c) for c in combos]
            with torch.no_grad():
                expected = model(**tokenizer(seqs, return_tensors='pt', padding=True))['logits'].numpy()
            np.testing.assert_allclose(batch_predict(seqs, tok, loaded, torch.device('cpu'), 2), expected, atol=1e-6)
            candidates = folder / 'candidates.csv'
            pd.DataFrame({'Combo': combos}).to_csv(candidates, index=False)
            output = folder / 'predictions.csv'
            command = [sys.executable, str(ROOT / 'lora_plm/predict.py'), '--model_path', str(base),
                       '--peft_dir', str(checkpoint), '--wt_fasta', str(ROOT / 'data/GB1/GB1_WT.fasta'),
                       '--crossmap', str(ROOT / 'data/GB1/gb1_refpos_crossmap.csv'), '--enzyme_name', 'GB1',
                       '--ref_positions', '39,40,41,54', '--candidates_csv', str(candidates),
                       '--out_csv', str(output), '--device', 'cpu', '--local_files_only']
            subprocess.run(command, check=True, capture_output=True, text=True)
            np.testing.assert_allclose(pd.read_csv(output).y_pred, expected, atol=1e-6)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            # Exercise the actual training writer, not only hand-written metadata.
            training = folder / 'training.csv'
            pd.DataFrame({'Combo': combos, 'Fitness': [1., 0.5, 2.]}).to_csv(training, index=False)
            trained = folder / 'trained'
            subprocess.run([sys.executable, str(ROOT / 'lora_plm/train.py'), '--model_path', str(base),
                            '--wt_fasta', str(ROOT / 'data/GB1/GB1_WT.fasta'),
                            '--crossmap', str(ROOT / 'data/GB1/gb1_refpos_crossmap.csv'),
                            '--enzyme_name', 'GB1', '--ref_positions', '39,40,41,54',
                            '--train_csv', str(training), '--obj_col', 'Fitness', '--out_dir', str(trained),
                            '--head', 'site_attn', '--attn_heads', '2', '--attn_layers', '2',
                            '--epochs', '1', '--device', 'cpu', '--local_files_only'],
                           check=True, capture_output=True, text=True)
            _, trained_model, _, trained_spec = load_regressor(str(base), trained,
                ROOT / 'data/GB1/gb1_refpos_crossmap.csv', 'GB1', [39, 40, 41, 54], local_files_only=True)
            self.assertEqual(trained_spec.indexing, 'one_based')
            self.assertEqual(len(trained_model.site_encoder.layers), 2)
            for kwargs in ({'head': 'meanpool'}, {'indexing': 'legacy_shifted'}):
                with self.assertRaises(ValueError):
                    read_checkpoint_spec(checkpoint, **kwargs)
            # A truncated saved head must fail, not silently discard/miss layers.
            torch.save({k: v for k, v in model.site_encoder.state_dict().items() if k.startswith('layers.0.')}, checkpoint / 'site_encoder.pt')
            with self.assertRaises(RuntimeError):
                load_regressor(str(base), checkpoint, ROOT / 'data/GB1/gb1_refpos_crossmap.csv',
                               'GB1', [39, 40, 41, 54], local_files_only=True)
            pointer = folder / 'pointer.pt'
            pointer.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n')
            with self.assertRaisesRegex(RuntimeError, 'git lfs pull'):
                load_state(pointer)
            with self.assertRaises(FileNotFoundError):
                load_state(folder / 'missing.pt')


class PaperResultTests(unittest.TestCase):
    def test_gb1_metrics_match_frozen_reference(self):
        result_dir = ROOT / 'results/lora_plm/gb1_beam'
        metrics, selected = evaluate(pd.read_csv(result_dir / 'beam_all_final.csv'),
                                     pd.read_csv(ROOT / 'data/GB1/GB1.CSV'),
                                     pd.read_csv(ROOT / 'data/GB1/gb1_stage2_train.csv'))
        reference = json.loads((result_dir / 'metrics.json').read_text())
        for key, value in reference.items():
            if isinstance(value, (float, int)):
                self.assertAlmostEqual(metrics[key], value, places=10)
            else:
                self.assertEqual(metrics[key], value)
        self.assertEqual(selected.Combo.tolist(), ['IYGV', 'IWGV', 'LYGV', 'LWGV', 'IYGC', 'IYGA', 'VYGC', 'IYGI', 'VYGI', 'VYGA'])



if __name__ == '__main__':
    unittest.main()

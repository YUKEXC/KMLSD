"""CPU-only checks for the current experimental inputs and Stage-I reader."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage1.score_hotspots import load_alanine_labels, infer_srs_region


class ExperimentalDataTests(unittest.TestCase):
    def test_srs_boundary_matches_the_manuscript_and_public_tables(self):
        ranges = [(68, 96), (173, 181), (186, 195), (233, 256), (287, 300), (390, 401)]
        expected = {p for lo, hi in ranges for p in range(lo, hi + 1)}
        self.assertEqual(len(expected), 98)
        self.assertEqual(infer_srs_region(300), 5)
        self.assertEqual(infer_srs_region(301), 0)
        for filename in ['stage1_scores.csv', 'ddg_srs_site_summary.csv', 'plm_srs_site_summary.csv']:
            table = pd.read_csv(ROOT / 'stage1_data/p450' / filename)
            self.assertEqual(len(table), 98)
            self.assertEqual(set(table.ref_pos), expected)

    def test_current_alanine_measurements_are_not_rescaled_or_dropped(self):
        path = ROOT / 'stage1_data/p450/alanine_labels.csv'
        source = pd.read_csv(path)
        labels = load_alanine_labels(path)
        self.assertEqual(len(labels), 85)
        self.assertEqual(labels.ref_pos.nunique(), 85)
        self.assertEqual(labels.Variant.tolist(), source.Variant.tolist())
        np.testing.assert_array_equal(labels.y, source.YUDCA)
        self.assertAlmostEqual(labels.loc[labels.Variant == 'S68A', 'y'].item(), 0.3727)
        self.assertIn('S295A', labels.Variant.tolist())

    def test_saturation_combinations_match_the_reported_single_mutations(self):
        table = pd.read_csv(ROOT / 'data/P450/fitness_round1_training_six_with_aux.csv')
        sequence = ''.join(line.strip() for line in (ROOT / 'WT.fasta').read_text().splitlines()
                           if not line.startswith('>'))
        positions = [68, 96, 173, 192, 294, 296]
        parent = ''.join(sequence[p - 1] for p in positions)
        self.assertEqual(parent, 'SQTVGF')
        self.assertEqual(len(table), 114)
        self.assertEqual(table.Combo.nunique(), 114)
        observed = {p: set() for p in positions}
        for row in table.itertuples():
            position = int(row.Variant[1:-1])
            self.assertEqual(row.Variant[0], sequence[position - 1])
            changes = [i for i, (a, b) in enumerate(zip(parent, row.Combo)) if a != b]
            self.assertEqual(len(row.Combo), 6)
            self.assertEqual(changes, [positions.index(position)])
            self.assertEqual(row.Combo[changes[0]], row.Variant[-1])
            observed[position].add(row.Variant[-1])
        for position, substitutions in observed.items():
            self.assertEqual(substitutions, set('ACDEFGHIKLMNPQRSTVWY') - {sequence[position - 1]})
        values = table[['YUDCA', 'SUDCA', 'YMDCA', 'SMDCA', 'Conversion']].to_numpy()
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(((values >= 0) & (values <= 1)).all())

    def test_invalid_measurement_records_are_rejected(self):
        invalid = [
            pd.DataFrame({'Variant': ['OM3'], 'YUDCA': [0.4]}),
            pd.DataFrame({'Variant': ['S68A', 'S68A'], 'YUDCA': [0.3, 0.4]}),
            pd.DataFrame({'Variant': ['S68A'], 'YUDCA': [np.inf]}),
            pd.DataFrame({'Variant': ['S68A'], 'YUDCA': [np.nan]}),
            pd.DataFrame({'Variant': ['S68A'], 'YUDCA': [37.27]}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.csv'
            for table in invalid:
                with self.subTest(table=table.to_dict()):
                    table.to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        load_alanine_labels(path)

    def test_legacy_label_schema_is_preserved(self):
        expected = pd.DataFrame({'ref_pos': [68, 69], 'y': [1.5, -0.5], 'risk': [0, 1]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.csv'
            expected.to_csv(path, index=False)
            pd.testing.assert_frame_equal(load_alanine_labels(path), expected)

    def test_stage1_uses_current_data_and_records_the_actual_input(self):
        source = ROOT / 'stage1_data/p450'
        with tempfile.TemporaryDirectory() as directory:
            for protocol, training, missing in [('paper', 98, 13), ('corrected', 85, 0)]:
                with self.subTest(protocol=protocol):
                    output = Path(directory) / protocol
                    subprocess.run([
                        sys.executable, str(ROOT / 'stage1/score_hotspots.py'),
                        '--in_dir', str(source), '--out_dir', str(output),
                        '--protocol', protocol, '--topk', '6', '--srs_only',
                        '--plm_csv', str(source / 'plm_srs_site_summary.csv'),
                        '--ddg_csv', str(source / 'ddg_srs_site_summary.csv'),
                    ], check=True, capture_output=True, text=True)
                    metadata = json.loads((output / 'score_metadata.json').read_text())
                    self.assertEqual(metadata['n_candidates'], 98)
                    self.assertIn([287, 300], metadata['srs_ranges'])
                    self.assertEqual(metadata['n_observed_labels'], 85)
                    self.assertEqual(metadata['n_training_labels'], training)
                    self.assertEqual(metadata['missing_labels_in_training'], missing)
                    self.assertEqual(metadata['nonzero_explicit_risk_penalties'], 0)
                    self.assertEqual(metadata['label_column'], 'YUDCA')
                    self.assertIn('alanine_labels.csv', metadata['input_sha256'])
                    scored = pd.read_csv(output / 'site_features_stage1.csv')
                    self.assertNotIn(301, scored.ref_pos.tolist())
                    measured = pd.read_csv(source / 'alanine_labels.csv')
                    joined = measured.merge(scored.dropna(subset=['Variant']), on='Variant', validate='one_to_one')
                    self.assertEqual(len(joined), 85)
                    np.testing.assert_allclose(joined.y, joined.YUDCA_x, rtol=0, atol=1e-15)
                    self.assertEqual(len(pd.read_csv(output / 'top6.csv')), 6)

    def test_positions_missing_from_features_are_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            pd.DataFrame({'ref_pos': [68], 'entropy': [1.]}).to_csv(source / 'msa_site_features.csv', index=False)
            pd.DataFrame({'Variant': ['R69A'], 'YUDCA': [0.3]}).to_csv(source / 'alanine_labels.csv', index=False)
            result = subprocess.run([
                sys.executable, str(ROOT / 'stage1/score_hotspots.py'),
                '--in_dir', str(source), '--out_dir', str(source / 'output'),
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('absent from MSA features', result.stderr)


if __name__ == '__main__':
    unittest.main()

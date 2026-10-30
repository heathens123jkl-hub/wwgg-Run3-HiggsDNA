"""Focused wiring regression tests; no ROOT, NanoAOD or network needed.

Extract the actual functions with AST to avoid importing every optional
workflow dependency. The map is a controlled event filter: these tests
validate WWgg integration, not official payload values or jet-ID physics.
"""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import awkward as ak
import numpy

SOURCE = Path(__file__).resolve().parents[1] / 'higgs_dna/workflows/WWgg.py'
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))


class Events:
    def __init__(self, array):
        self.array = array
        self.metadata = {'dataset': 'WWgg_test'}

    def __len__(self):
        return len(self.array)

    def __getattr__(self, key):
        return getattr(self.array, key)

    def __getitem__(self, mask):
        return Events(self.array[mask])


class TestWWggCorrectionWiring(unittest.TestCase):
    def setUp(self):
        self.events = Events(ak.Array([
            {'event': 10, 'genWeight': 2., 'GenPart': [], 'Jet': [{'pt': 40.}], 'Electron': [{'pt': 25.}]},
            {'event': 11, 'genWeight': -1., 'GenPart': [], 'Jet': [{'pt': 60.}], 'Electron': []},
            {'event': 12, 'genWeight': 3., 'GenPart': [], 'Jet': [], 'Electron': [{'pt': 15.}]},
        ]))
        self.map = Mock(side_effect=lambda processor, **kw: kw['events'][kw['events'].event != 11])
        self.ns = {'jetvetomap': self.map, 'logger': logging.getLogger('wwgg_test'), 'numpy': numpy}
        self.ns.update(available_object_corrections={}, available_weight_corrections={})
        validator = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'wwgg_validate_corrections')
        exec(compile(ast.Module(body=[validator], type_ignores=[]), str(SOURCE), 'exec'), self.ns)
        fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'wwgg_apply_jet_veto_map')
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), 'exec'), self.ns)
        self.apply = self.ns['wwgg_apply_jet_veto_map']
        from higgs_dna.tools.wwgg_corrections import options
        self.processor = SimpleNamespace(skipJetVetoMap=False, year={'WWgg_test': ['2024']}, corrections={}, systematics={}, wwgg_corrections=options())

    def test_event_filter_preserves_alignment_and_all_jets(self):
        events, counts = self.apply(self.processor, self.events, 'WWgg_test', '2024')
        self.assertEqual(ak.to_list(events.event), [10, 12])
        self.assertEqual(ak.to_list(events.Electron.pt), [[25.], [15.]])
        self.assertEqual(ak.to_list(events.Jet.pt), [[40.], []])
        self.assertEqual(counts['cf_rejected_jetvetomap'], 1)
        args, kw = self.map.call_args
        self.assertIs(args[0], self.processor)
        self.assertIs(kw['events'], self.events)
        self.assertEqual(kw['year'], '2024')
        self.assertEqual(kw['dataset_name'], 'WWgg_test')
        self.assertTrue(hasattr(kw['logger'], 'debug'))

    def test_explicit_skip_keeps_identical_events(self):
        self.processor.skipJetVetoMap = True
        events, counts = self.apply(self.processor, self.events, 'WWgg_test', '2024')
        self.assertIs(events, self.events)
        self.assertEqual(counts['cf_rejected_jetvetomap'], 0)
        self.map.assert_not_called()

    def test_failure_is_not_silently_skipped(self):
        self.map.side_effect = FileNotFoundError('missing payload')
        with self.assertRaisesRegex(FileNotFoundError, 'missing payload'):
            self.apply(self.processor, self.events, 'WWgg_test', '2024')

    def test_all_events_rejected(self):
        self.map.side_effect = lambda processor, **kw: kw['events'][kw['events'].event < 0]
        events, counts = self.apply(self.processor, self.events, 'WWgg_test', '2024')
        self.assertEqual(len(events), 0)
        self.assertEqual(counts['cf_rejected_jetvetomap'], 3)

    def test_empty_chunk_does_not_call_map(self):
        events, counts = self.apply(self.processor, self.events[:0], 'WWgg_test', '2024')
        self.assertEqual(len(events), 0)
        self.assertEqual(counts['cf_before_jetvetomap'], 0)
        self.map.assert_not_called()

    def prefix(self):
        cls = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'WWggProcessor')
        process = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'process')
        stop = next(i for i, n in enumerate(process.body) if isinstance(n, ast.Assign)
                    and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                    and n.value.func.attr == 'apply_filters_and_triggers')
        body = process.body[:stop] + ast.parse('return events, sum_genw_presel, early_cutflow').body
        fn = ast.FunctionDef(name='prefix', args=process.args, body=body, decorator_list=[])
        tree = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
        exec(compile(tree, str(SOURCE), 'exec'), self.ns)
        return self.ns['prefix']

    def test_generator_denominator_includes_rejected_events(self):
        events, denominator, counts = self.prefix()(self.processor, self.events)
        self.assertEqual(denominator, 4.)  # 2 - 1 + 3 before map; selected sum = 5.
        self.assertEqual(float(ak.sum(events.genWeight)), 5.)
        self.assertEqual(counts['n_input'], 3)
        self.assertEqual(counts['cf_after_jetvetomap'], 2)

    def test_empty_selection_still_returns_cutflow(self):
        self.map.side_effect = lambda processor, **kw: kw['events'][:0]
        result = self.prefix()(self.processor, self.events)['WWgg']
        self.assertEqual(result['n_input'], 3)
        self.assertEqual(result['cf_rejected_jetvetomap'], 3)
        self.assertEqual(result['n_events'], 0)

    def test_only_one_shared_map_call(self):
        calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'jetvetomap']
        self.assertEqual(len(calls), 1)

    def test_raw_pt_only_for_scale_or_smearing(self):
        cond = next(n.test for n in ast.walk(TREE) if isinstance(n, ast.If)
                    and 'correction.lower()' in ast.unparse(n.test))
        code = compile(ast.Expression(cond), str(SOURCE), 'eval')
        for name in ('Pileup', 'TriggerSF', 'LoosePhoIDSF', 'jec_jet'):
            self.assertFalse(eval(code, {'correction': name}))
        for name in ('Scale2G_IJazZ', 'Smearing2G_IJazZ', 'Electron_Scale_Trad'):
            self.assertTrue(eval(code, {'correction': name}))


if __name__ == '__main__':
    unittest.main(verbosity=2)

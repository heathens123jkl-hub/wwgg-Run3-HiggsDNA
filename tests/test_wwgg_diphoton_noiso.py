"""Exercise the actual selection functions without the full Coffea runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings
import awkward as ak
import numpy

ROOT = Path(__file__).resolve().parents[1]
ns = dict(ak=ak, numpy=numpy, warnings=warnings)
for path, name in [
    ('higgs_dna/selections/diphoton_selections.py', 'apply_fiducial_cut_det_level'),
    ('higgs_dna/workflows/WWgg.py', 'wwgg_select_diphotons'),
]:
    tree = ast.parse((ROOT/path).read_text(encoding='utf-8'))
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, 'exec'), ns)


class Tests(unittest.TestCase):
    def processor(self, mode='apply'):
        return SimpleNamespace(fiducialCuts='classical_noIso', validate_with_electrons=False,
                               wwgg_selection={'diphoton_pt_over_mass': mode})

    def candidates(self):
        # No isolation branches at all: noIso must not need them.
        return ak.Array([[dict(mass=m, pho_lead={'pt': p1}, pho_sublead={'pt': p2})
                          for m, p1, p2 in [(120., 50., 35.), (120., 40., 35.),
                                           (120., 50., 30.), (100., 50., 35.),
                                           (180., 80., 60.), (125., 60., 40.)]], []])

    def test_no_iso_branches_and_strict_boundaries(self):
        out = ns['wwgg_select_diphotons'](self.processor(), self.candidates())
        self.assertEqual(ak.to_list(out.mass), [[120., 125.], []])
        self.assertTrue(ak.all(out.pass_fiducial_classical_noIso))

    def test_high_isolation_is_retained(self):
        x = self.candidates()
        for name in ('pho_lead', 'pho_sublead'):
            x[name, 'pfRelIso03_all_quadratic'] = ak.ones_like(x[name].pt)*100
        out = ns['wwgg_select_diphotons'](self.processor(), x)
        self.assertEqual(ak.to_list(out.mass), [[120., 125.], []])

    def test_store_only_preserves_mass_window_and_flags(self):
        out = ns['wwgg_select_diphotons'](self.processor('store_only'), self.candidates())
        self.assertEqual(ak.to_list(out.pass_fiducial_classical_noIso), [[True, False, False, True], []])

    def test_classical_still_uses_isolation(self):
        x = self.candidates()
        for name in ('pho_lead', 'pho_sublead'):
            x[name, 'eta'] = ak.zeros_like(x[name].pt)
            x[name, 'pfRelIso03_all_quadratic'] = ak.ones_like(x[name].pt)*100
        p = self.processor(); p.fiducialCuts = 'classical'
        out = ns['apply_fiducial_cut_det_level'](p, x)
        self.assertEqual(ak.to_list(ak.num(out)), [0, 0])


if __name__ == '__main__': unittest.main(verbosity=2)

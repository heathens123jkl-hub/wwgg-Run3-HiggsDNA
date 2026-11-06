"""Configuration guards and nominal Hgg registration, without Coffea."""
import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT/'higgs_dna/workflows/WWgg.py').read_text(encoding='utf-8'))
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'wwgg_validate_corrections')
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<guard>', 'exec'), ns)


class Tests(unittest.TestCase):
    def test_double_calibration_rejected(self):
        for names in [('Smearing','Smearing2G_IJazZ'), ('Scale','Scale_Trad'), ('Scale','Smearing')]:
            with self.assertRaises(ValueError): ns['wwgg_validate_corrections'](names, dict.fromkeys(names), {})

    def test_single_calibration_allowed(self):
        ns['wwgg_validate_corrections'](['Smearing','Pileup'], {'Smearing':None}, {'Pileup':None})

    def test_new_configs_replace_old_names(self):
        for kind, name in [('signal','Smearing'), ('data_CDEI','Scale')]:
            cfg = json.loads((ROOT/f'production/wwgg/production_{kind}_20261106.json').read_text())
            for corrections in cfg['corrections'].values():
                self.assertIn(name, corrections)
                self.assertFalse(any('IJazZ' in x for x in corrections))

    def test_registered_scale_includes_zmmg(self):
        t = ast.parse((ROOT/'higgs_dna/systematics/registry.py').read_text())
        node = next(n.value for n in t.body if isinstance(n,ast.Assign)
                    and any(isinstance(x,ast.Name) and x.id=='object_corrections' for x in n.targets))
        calls = {k.value:v for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant)}
        scale = calls['Scale']
        self.assertEqual(scale.args[0].id, 'Scale_ZeeZmmg')
        self.assertIs(next(k.value.value for k in scale.keywords if k.arg=='is_Zee'), False)
        for name in ['Scale','Smearing']:
            self.assertEqual(next(k.value.value for k in calls[name].keywords if k.arg=='gaussians'),'2G')


if __name__=='__main__':unittest.main(verbosity=2)

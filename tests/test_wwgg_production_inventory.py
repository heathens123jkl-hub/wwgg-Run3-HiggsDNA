import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'production/wwgg/prepare_all_20261031.py'
spec = importlib.util.spec_from_file_location('prepare', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
WGG_FILE = '/store/mc/RunIII2024Summer24NanoAODv15/WGG-4FS_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/00000/example.root'


class Tests(unittest.TestCase):
    def test_missing_wgg_stops(self):
        with self.assertRaises(ValueError):
            module.inventory([])

    def test_full_inventory_and_duplicate_lfn(self):
        groups, removed = module.inventory([WGG_FILE, 'root://example.org/'+WGG_FILE])
        self.assertEqual(len(groups['signal']), 3)
        self.assertEqual(len(groups['singleH']), 6)
        self.assertEqual(len(groups['data']), 16)
        self.assertEqual(len(groups['continuum']), 4)
        self.assertEqual(removed['WGG_4FS_2024'], 1)
        self.assertEqual(sum(map(len, groups['signal'].values())), 276)
        self.assertEqual(sum(map(len, groups['singleH'].values())), 453)

    def test_wrong_campaign_rejected(self):
        with self.assertRaises(ValueError):
            module.inventory([WGG_FILE.replace('Summer24', 'Summer23')])


if __name__ == '__main__':
    unittest.main()

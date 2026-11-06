import unittest
from unittest.mock import patch
import awkward as ak
from higgs_dna.tools import wwgg_corrections as c


class Tests(unittest.TestCase):
    def run_flags(self, mask):
        events=ak.Array([{'event':1},{'event':2}])
        jets=ak.Array([[{'pt':487.,'eta':1.6,'btagUParTAK4B':-1.}],
                       [{'pt':50.,'eta':.2,'btagUParTAK4B':.8}]])
        cfg=dict(branch='btagUParTAK4B',threshold=.4648,pt_min=30.,abs_eta_max=2.5)
        with patch('higgs_dna.tools.wwgg_b_veto.load_b_config',return_value=cfg):
            return c.veto_flags(events,None,None,None,jets,c.options({'b_veto':'apply'}),mask)

    def test_rejected_event_keeps_invalid_counter(self):
        out=self.run_flags(ak.Array([False,True]))
        self.assertEqual(ak.to_list(out['b_n_invalid_kin']),[1,0])
        self.assertEqual(ak.to_list(out['b_veto']),[False,True])

    def test_selected_invalid_still_fails(self):
        with self.assertRaisesRegex(ValueError,'Invalid UParT'):
            self.run_flags(ak.Array([True,True]))

    def test_default_stays_strict(self):
        with self.assertRaisesRegex(ValueError,'Invalid UParT'):
            self.run_flags(None)

    def test_misaligned_mask(self):
        with self.assertRaisesRegex(ValueError,'Unaligned'):
            self.run_flags(ak.Array([False]))


if __name__=='__main__':
    unittest.main(verbosity=2)

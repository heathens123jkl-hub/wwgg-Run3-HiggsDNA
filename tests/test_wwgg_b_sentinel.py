import unittest
import awkward as ak
import numpy as np
from higgs_dna.tools.wwgg_b_veto import evaluate_b
from higgs_dna.tools.wwgg_b_efficiency import collect

class Tests(unittest.TestCase):
    cfg=dict(branch='btagUParTAK4B',threshold=.4648,pt_min=30.,abs_eta_max=2.5)
    def jets(self,scores):
        return ak.Array([[dict(pt=100.,eta=.2,btagUParTAK4B=s,hadronFlavour=5) for s in scores]])
    def test_sentinel_is_untagged_and_preserved(self):
        j=self.jets([-1.]);r=evaluate_b(j,self.cfg)
        self.assertEqual(ak.to_list(r['b_trial_veto']),[False])
        self.assertEqual(ak.to_list(r['b_trial_n_invalid_kin']),[0])
        self.assertEqual(ak.to_list(r['b_trial_n_sentinel_kin']),[1])
        self.assertEqual(ak.to_list(j.btagUParTAK4B),[[-1.]])
    def test_other_tagged_jet_still_vetoes(self):
        self.assertEqual(ak.to_list(evaluate_b(self.jets([-1.,.4648]),self.cfg)['b_trial_veto']),[True])
    def test_other_invalid_scores_still_invalid(self):
        for score in [-2.,-.5,1.1,float('nan'),float('inf')]:
            j=self.jets([score])
            self.assertEqual(ak.to_list(evaluate_b(j,self.cfg)['b_trial_n_invalid_kin']),[1])
            with self.assertRaises(ValueError):collect(j,ak.Array([1.]),ak.Array([0]))
    def test_efficiency_denominator_includes_sentinel(self):
        r=collect(self.jets([-1.,.8]),ak.Array([-2.]),ak.Array([0]))
        m=r['channels']['0']['5']
        self.assertEqual(np.sum(m['all']['rows']),2)
        self.assertEqual(np.sum(m['tagged']['rows']),1)
        self.assertEqual(np.sum(m['all']['sumw']),-4)
        self.assertEqual(r['sentinel_jets'],1)

if __name__=='__main__':unittest.main(verbosity=2)

import unittest
from types import SimpleNamespace
import awkward as ak
import numpy as np
from higgs_dna.tools.wwgg_jec_type1 import calculate, BRANCHES, SOURCES


class Correction:
    inputs = [SimpleNamespace(name='JetPt', type='real')]
    def __init__(self, factor): self.factor = factor
    def evaluate(self, pt): return np.full_like(pt, self.factor, dtype=float)


class ToyPayload:
    def __init__(self):
        self.compound = {f'Summer24Prompt24_V5_{era}_L1L2L3Res_AK4PFPuppi': Correction(1.25)
                         for era in ('MC', 'DATA')}
        self.calls = []
    def __getitem__(self, name):
        self.calls.append(name)
        return Correction(1. if '_L1FastJet_' in name else .1)


def fixture():
    # Event 0: 100 GeV jet -> 80 raw -> 100 calibrated; MET shifts by -20.
    # Event 1: low-pT collection alone, 14 raw -> 17.5; contribution is -3.5.
    # Event 2: no jets, raw MET stays unchanged.
    d = {x: [[0.], [], []] for x in BRANCHES if x.startswith('Jet_')}
    d.update({x: [[], [0.], []] for x in BRANCHES if x.startswith('CorrT1METJet_')})
    d.update(run=[386509]*3, luminosityBlock=[157]*3, event=[1,2,3],
             Rho_fixedGridRhoFastjetAll=[30.]*3, RawPuppiMET_pt=[40.]*3,
             RawPuppiMET_phi=[0.]*3, PuppiMET_pt=[999.]*3, PuppiMET_phi=[1.]*3)
    d.update(Jet_pt=[[100.],[],[]], Jet_mass=[[10.],[],[]],
             Jet_rawFactor=[[.2],[],[]], Jet_area=[[.5],[],[]],
             CorrT1METJet_rawPt=[[],[14.],[]], CorrT1METJet_area=[[],[.5],[]])
    return ak.Array(d)


class CandidateTests(unittest.TestCase):
    def test_float32_input_scalar_reference(self):
        a=fixture()
        a=ak.with_field(a,np.array([42.348,67.123,23.8],dtype=np.float32),'RawPuppiMET_pt')
        a=ak.with_field(a,np.array([.435,-2.156,1.234],dtype=np.float32),'RawPuppiMET_phi')
        r=calculate(a,ToyPayload(),False,False)
        import math
        expected_x=[float(p)*math.cos(float(phi))-dx for p,phi,dx in zip(a.RawPuppiMET_pt,a.RawPuppiMET_phi,[20.,3.5,0.])]
        np.testing.assert_allclose(r['met']['nominal']['px'],expected_x,rtol=1e-12,atol=1e-12)

    def test_invalid_jes_diagnostic_preserves_values(self):
        class InvalidPayload(ToyPayload):
            def __getitem__(self,name):
                return Correction(1. if '_L1FastJet_' in name else -999.)
        a=fixture();p=InvalidPayload();r=calculate(a,p,False,False)
        with self.assertRaises(ValueError):calculate(a,p,False)

    def test_signed_jes_and_met_preserve_direction(self):
        class SignedCorrection(Correction):
            def evaluate(self,pt): return np.where(pt>50., -.1, .1)
        class SignedPayload(ToyPayload):
            def __getitem__(self,name):
                return Correction(1.) if '_L1FastJet_' in name else SignedCorrection(0.)
        a=fixture();p=SignedPayload();r=calculate(a,p,False)
        # Same nuisance moves the regular jet down and the low-pT jet up.
        np.testing.assert_allclose(ak.flatten(r['jets']['Absolute_up']['pt']),[90.])
        np.testing.assert_allclose(ak.flatten(r['jets']['Absolute_down']['pt']),[110.])
        np.testing.assert_allclose(r['met']['Absolute_up']['px'],[30.,34.75,40.])
        np.testing.assert_allclose(r['met']['Absolute_down']['px'],[10.,38.25,40.])

    def test_raw_once_mass_and_raw_met_baseline(self):
        a = fixture(); before = ak.to_list(a)
        r = calculate(a, ToyPayload(), False)
        np.testing.assert_allclose(ak.flatten(r['jets']['nominal']['pt']), [100.])
        np.testing.assert_allclose(ak.flatten(r['jets']['nominal']['mass']), [10.])
        np.testing.assert_allclose(r['met']['nominal']['px'], [20.,36.5,40.])
        self.assertEqual(before, ak.to_list(a))

    def test_jes_sources_and_shared_baseline(self):
        payload = ToyPayload(); r = calculate(fixture(), payload, False)
        self.assertEqual(len(r['met']),23)
        np.testing.assert_allclose(r['met']['Absolute_up']['px'], [10.,34.75,40.])
        np.testing.assert_allclose(r['met']['Absolute_down']['px'], [30.,38.25,40.])
        self.assertTrue(any('_Regrouped_RelativeBal_' in k for k in payload.calls))
        for s in SOURCES: self.assertIn(s+'_down', r['jets'])

    def test_muon_subtraction_and_direction(self):
        a=fixture()
        a=ak.with_field(a, ak.Array([[.5],[],[]]), 'Jet_muonSubtrFactor')
        a=ak.with_field(a, ak.Array([[np.pi/2],[],[]]), 'Jet_muonSubtrDeltaPhi')
        r=calculate(a,ToyPayload(),False,False)
        np.testing.assert_allclose(r['met']['nominal']['px'][0],40.)
        np.testing.assert_allclose(r['met']['nominal']['py'][0],-10.)
        np.testing.assert_allclose(ak.flatten(r['jets']['nominal']['pt']),[100.])

    def test_em_cut_and_eta_cut(self):
        a=fixture()
        a=ak.with_field(a,ak.Array([[.9],[],[]]),'Jet_chEmEF')
        a=ak.with_field(a,ak.Array([[],[5.2],[]]),'CorrT1METJet_eta')
        r=calculate(a,ToyPayload(),False,False)
        np.testing.assert_allclose(r['met']['nominal']['px'],[40.,40.,40.])

    def test_threshold_migration(self):
        a=ak.with_field(fixture(),ak.Array([[],[12.],[]]),'CorrT1METJet_rawPt')
        r=calculate(a,ToyPayload(),False)
        self.assertEqual(r['met']['nominal']['n_type1_low'][1],0)
        self.assertEqual(r['met']['Absolute_up']['n_type1_low'][1],1)
        self.assertEqual(r['met']['Absolute_down']['n_type1_low'][1],0)

    def test_data_no_mc_jes(self):
        r=calculate(fixture(),ToyPayload(),True)
        self.assertEqual(list(r['met']),['nominal'])

    def test_empty_chunk(self):
        r=calculate(fixture()[:0],ToyPayload(),False)
        self.assertEqual(len(r['met']['nominal']['pt']),0)

    def test_missing_or_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            calculate(ak.without_field(fixture(),'CorrT1METJet_EmEF'),ToyPayload(),False)
        with self.assertRaises(ValueError):
            calculate(ak.with_field(fixture(),[float('nan')]*3,'RawPuppiMET_pt'),ToyPayload(),False)


if __name__=='__main__': unittest.main(verbosity=2)

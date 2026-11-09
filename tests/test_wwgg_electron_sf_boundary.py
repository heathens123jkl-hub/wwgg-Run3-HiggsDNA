import unittest
from unittest.mock import patch
import awkward as ak
import numpy as np
from higgs_dna.tools import wwgg_corrections as c
class Tests(unittest.TestCase):
    def test_boundary(self):
        raw=np.array([2.5001373291015625,-2.5001373291015625,2.5,-2.5,2.49]);before=raw.copy()
        eta,flag=c.electron_sf_eta(raw)
        np.testing.assert_array_equal(raw,before)
        np.testing.assert_array_equal(flag,[True,True,True,True,False])
        self.assertTrue(np.all(abs(eta)<2.5));self.assertEqual(eta[-1],2.49)
        self.assertEqual(eta[0],-eta[1])
    def test_invalid(self):
        for v in [2.502,-2.502,np.nan,np.inf]:
            with self.assertRaises(ValueError):c.electron_sf_eta([v])
    def test_empty(self):
        self.assertEqual(len(c.electron_sf_eta([])[0]),0)
    def test_components(self):
        ele=ak.Array([[dict(pt=61.9,eta=2.49560546875,deltaEtaSC=.0045318603515625)],[]])
        calls=[]
        class SF:
            def evaluate(self,year,var,wp,eta,pt):
                calls.append(eta.copy());return np.ones(len(pt))*1.02
        diag={}
        with patch.object(c,'payload',return_value={'Electron-ID-SF':SF()}):
            r=c.lepton_components(ele,ele[:,:0],diagnostics=diag)
        self.assertEqual(len(calls),6)
        self.assertTrue(all(np.all(abs(x)<2.5) for x in calls))
        np.testing.assert_array_equal(diag['electron_sf_eta_boundary_count'],[1,0])
        np.testing.assert_allclose(r['electron_id_nominal'],[1.02,1.])
        self.assertGreater(float(ele.eta[0,0]+ele.deltaEtaSC[0,0]),2.5)
if __name__=='__main__':unittest.main(verbosity=2)

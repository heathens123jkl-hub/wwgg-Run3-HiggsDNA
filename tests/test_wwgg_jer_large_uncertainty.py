import unittest
import numpy as np
from higgs_dna.tools.wwgg_corrections import jer_sf_variation
class Tests(unittest.TestCase):
    def test_real_points_nominal_and_up(self):
        sf=np.array([1.6394801856359322]*2)
        unc=np.array([1.218732970823184,1.127563632936449])
        res=np.array([.0896280619784625,.09142456212398681])
        for mode in ['nominal','mu_id_up','Absolute_up']:
            np.testing.assert_array_equal(jer_sf_variation(res,sf,unc,mode),sf)
        np.testing.assert_allclose(jer_sf_variation(res,sf,unc,'JER_up'),sf*(1+unc))
        with self.assertRaisesRegex(ValueError,'Nonphysical varied'):
            jer_sf_variation(res,sf,unc,'JER_down')
    def test_normal_down(self):
        np.testing.assert_allclose(jer_sf_variation([.1],[1.2],[.2],'JER_down'),[.96])
    def test_invalid_stays_invalid(self):
        for values in [([-.1],[1.2],[.1]),([.1],[0],[.1]),([.1],[1.2],[-.1]),([.1],[1.2],[np.inf])]:
            with self.assertRaises(ValueError):jer_sf_variation(*values,'nominal')
    def test_empty(self):
        self.assertEqual(len(jer_sf_variation([],[],[],'nominal')),0)
if __name__=='__main__':unittest.main(verbosity=2)

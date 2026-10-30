import ast
from pathlib import Path
import unittest
import awkward as ak
import numpy as np
from higgs_dna.tools.wwgg_b_efficiency import collect


class Tests(unittest.TestCase):
    def jets(self):
        return ak.Array([[{'pt':40.,'eta':.3,'btagUParTAK4B':.8,'hadronFlavour':5},
                          {'pt':45.,'eta':.4,'btagUParTAK4B':.1,'hadronFlavour':5}],
                         [{'pt':60.,'eta':1.,'btagUParTAK4B':.4648,'hadronFlavour':4}],[]])

    def test_signed_and_tagged(self):
        r=collect(self.jets(),ak.Array([-2.,3.,1.]),ak.Array([0,1,2]))
        b=r['channels']['0']['5'];c=r['channels']['1']['4']
        self.assertEqual(np.sum(b['all']['rows']),2)
        self.assertEqual(np.sum(b['tagged']['rows']),1)
        self.assertEqual(np.sum(b['all']['sumw']),-4)
        self.assertEqual(np.sum(b['all']['sumw2']),8)
        self.assertEqual(np.sum(b['all']['sumabsw']),4)
        self.assertEqual(np.sum(c['tagged']['rows']),1)

    def test_empty_and_no_mutation(self):
        jets=self.jets();before=ak.to_list(jets)
        r=collect(jets[:0],ak.Array([1.])[:0],ak.Array([0])[:0])
        self.assertEqual(r['events_before_b_veto'],0)
        collect(jets,ak.Array([1.,1.,1.]),ak.Array([0,1,2]))
        self.assertEqual(ak.to_list(jets),before)

    def test_boundaries(self):
        j=ak.Array([[{'pt':30.,'eta':0.,'btagUParTAK4B':.8,'hadronFlavour':5},
                     {'pt':40.,'eta':2.5,'btagUParTAK4B':.8,'hadronFlavour':5},
                     {'pt':1500.,'eta':2.4,'btagUParTAK4B':.4648,'hadronFlavour':5}]])
        r=collect(j,ak.Array([1.]),ak.Array([0]))
        self.assertEqual(np.sum(r['channels']['0']['5']['tagged']['rows']),1)

    def test_chunk_additivity(self):
        j=self.jets();w=ak.Array([-2.,3.,1.]);c=ak.Array([0,1,2])
        full=collect(j,w,c);parts=[collect(j[:1],w[:1],c[:1]),collect(j[1:],w[1:],c[1:])]
        for cat in ('0','1','2'):
            for f in ('0','4','5'):
                for mode in ('all','tagged'):
                    for moment in ('rows','sumw','sumw2','sumabsw','negative_rows'):
                        vals=[np.asarray(x['channels'][cat][f][mode][moment]) for x in [full]+parts]
                        np.testing.assert_allclose(vals[0],vals[1]+vals[2])

    def test_invalid_flavour(self):
        j=self.jets();j['hadronFlavour']=ak.ones_like(j.pt)
        with self.assertRaises(ValueError):collect(j,ak.Array([1.,1.,1.]),ak.Array([0,1,2]))

    def test_workflow_order(self):
        path=Path(__file__).resolve().parents[1]/'higgs_dna/workflows/WWgg.py'
        source=path.read_text(encoding='utf-8');ast.parse(source)
        start=source.index('b_efficiency_record = None')
        self.assertLess(start,source.index("if cfg['b_veto'] == 'apply': presel"))
        self.assertGreater(start,source.index("if cfg['tau_veto'] == 'apply': presel"))
        self.assertIn("metadata['wwgg_b_efficiency']",source)


if __name__=='__main__':unittest.main(verbosity=2)

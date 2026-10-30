import json
from pathlib import Path
import unittest
from unittest.mock import patch
import awkward as ak
import numpy as np
from higgs_dna.tools import wwgg_corrections as c
from higgs_dna.tools.wwgg_b_veto import evaluate_b
from higgs_dna.tools.wwgg_tau_veto import evaluate
from higgs_dna.tools.wwgg_jer_math import factors


class Tests(unittest.TestCase):
    def test_flow_flat_scores_when_total_photons_equals_events(self):
        # Two photons + zero photons: len(flat_scores) == len(events) must not
        # be mistaken for an already-jagged array.
        import ast
        path=c.ROOT/'higgs_dna/tools/flow_corrections.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_flow_corrections_to_photons')
        ns={'ak':ak,'np':np,'calculate_flow_corrections':lambda *a,**k:(np.array([[.91],[.92]]),['r9'])}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
        ph=ak.Array([[{'r9':.8,'mvaID':.1},{'r9':.9,'mvaID':.2}],[]])
        meta={'flashggPhotons':{'flow_inputs':['r9'],'Isolation_transform_order':[]}}
        out=ns['apply_flow_corrections_to_photons'](ph,None,meta,'2024',lambda *a:np.array([.3,.4]),precision='float64')
        self.assertEqual(ak.to_list(out.mvaID),[[.3,.4],[]])
        self.assertEqual(ak.to_list(out.raw_r9),[[.8,.9],[]])

    def test_jme_adapter_replaces_met_and_preserves_raw_jet(self):
        from test_wwgg_jec_type1 import fixture,ToyPayload
        flat=fixture()
        nested={}
        for name in flat.fields:
            if '_' in name:
                collection,field=name.split('_',1)
                nested.setdefault(collection,{})[field]=flat[name]
            else:nested[name]=flat[name]
        for key,value in list(nested.items()):
            if isinstance(value,dict):nested[key]=ak.zip(value)
        events=ak.zip(nested,depth_limit=1)
        with patch.object(c,'payload',return_value=ToyPayload()), patch.object(c,'jet_id',side_effect=lambda j:ak.ones_like(j.pt,dtype=np.int32)*6):
            out=c.apply_jme(events,c.options({'jme':True}),True)
        np.testing.assert_allclose(out.PuppiMET.pt,[20.,36.5,40.])
        np.testing.assert_allclose(out.PuppiMET.pt_nano,[999.,999.,999.])
        np.testing.assert_allclose(ak.flatten(out.Jet.pt_raw),[80.])
        np.testing.assert_allclose(ak.flatten(out.Jet.pt),[100.])

    def test_invalid_configuration_fails(self):
        for cfg in ({'jer':True},{'jme':'yes'},{'variation':'JER_up'},
                    {'variation':'Absolute_up'},{'variation':'mu_id_up'},
                    {'b_veto':'yes'},{'unknown':True}):
            with self.assertRaises(ValueError): c.options(cfg)

    def test_veto_off_needs_no_tau_or_tag_branch(self):
        events=ak.Array([{'event':1},{'event':2}])
        out=c.veto_flags(events,None,None,None,None,c.options())
        self.assertEqual(ak.to_list(out['tau_veto']),[False,False])
        self.assertEqual(ak.to_list(out['b_veto']),[False,False])

    def test_b_veto_kinematics_and_invalid_scores(self):
        jets=ak.Array([[{'pt':31.,'eta':.2,'btagUParTAK4B':.5}],
                       [{'pt':29.,'eta':.2,'btagUParTAK4B':.9}],
                       [{'pt':31.,'eta':2.6,'btagUParTAK4B':.9}],
                       [{'pt':31.,'eta':.2,'btagUParTAK4B':-1.}]])
        cfg=dict(branch='btagUParTAK4B',threshold=.4648,pt_min=30.,abs_eta_max=2.5)
        out=evaluate_b(jets,cfg)
        self.assertEqual(ak.to_list(out['b_trial_veto']),[True,False,False,False])
        self.assertEqual(ak.to_list(out['b_trial_n_invalid_kin']),[0,0,0,1])

    def test_tau_requires_separated_pair_not_one_tau(self):
        tau=dict(pt=30.,eta=.5,phi=1.,dz=.01,charge=1,decayMode=0,
                 idDeepTau2018v2p5VSjet=4,idDeepTau2018v2p5VSe=6,idDeepTau2018v2p5VSmu=4)
        taus=ak.Array([[tau],[tau],[tau,dict(tau,phi=2.)]])
        empty=ak.Array([[dict(pt=20.,eta=0.,phi=0.,charge=-1)]]*3)[:,:0]
        electrons=ak.Array([[],[dict(pt=20.,eta=0.,phi=-1.,charge=-1)],[]])
        dipho=ak.Array([dict(pho_lead=dict(eta=-1.,phi=-2.),pho_sublead=dict(eta=-1.,phi=-1.))]*3)
        cfg=json.loads((c.ROOT/'production/wwgg/tau_veto_2024.json').read_text())
        out=evaluate(taus,dipho,electrons,empty,cfg)
        self.assertEqual(ak.to_list(out['tau_trial_veto']),[False,True,True])
        self.assertEqual(ak.to_list(out['tau_trial_etau']),[False,True,False])
        self.assertEqual(ak.to_list(out['tau_trial_tautau']),[False,False,True])

    def test_lepton_components_empty_events_and_reco_boundaries(self):
        calls=[]
        class Fake:
            def evaluate(self,*args):
                calls.append(args)
                return np.full(len(args[-1]),.98)
        def load(name): return {'Electron-ID-SF':Fake()}
        ele=ak.Array([[{'pt':12.,'eta':.1,'deltaEtaSC':.02},{'pt':75.,'eta':.3,'deltaEtaSC':0.}],[]])
        mu=ak.Array([[{'pt':20.,'eta':.2}],[]])[:,:0]
        with patch.object(c,'payload',load): out=c.lepton_components(ele,mu)
        np.testing.assert_allclose(out['electron_reco_nominal'],[.98**2,1.])
        np.testing.assert_allclose(out['mu_id_nominal'],[1.,1.])
        self.assertEqual({x[2] for x in calls},{'Loose','RecoBelow20','RecoAbove75'})
        self.assertTrue(all(x[0]=='2024Prompt' for x in calls))

    def test_jer_shared_draw_matched_and_unmatched(self):
        pt=np.array([100.,100.]); gp=np.array([80.,-1.]); matched=np.array([True,False]); z=np.array([.5,.5])
        nom=factors(pt,gp,matched,np.array([.1,.1]),np.array([1.2,1.2]),z)
        np.testing.assert_allclose(nom,[1.04,1+.05*np.sqrt(1.2**2-1)])
        np.testing.assert_array_equal(nom,factors(pt,gp,matched,np.array([.1,.1]),np.array([1.2,1.2]),z))


if __name__=='__main__': unittest.main(verbosity=2)

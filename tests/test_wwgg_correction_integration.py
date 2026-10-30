"""Dependency-light tests of shared correction routing (not payload validation)."""
import ast
import logging
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1]/'higgs_dna/workflows/WWgg.py'
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))


class Tests(unittest.TestCase):
    def setUp(self):
        self.ns = {'logger':logging.getLogger('test'), 'select_jets':Mock(return_value='baseline'),
                   'select_jets_eta_dependent':Mock(return_value='eta')}
        names = {'wwgg_selection_options','wwgg_validate_corrections','wwgg_apply_photon_flow','wwgg_select_jets'}
        nodes = [n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),self.ns)

    def test_old_selection_and_explicit_opt_in(self):
        options=self.ns['wwgg_selection_options']
        self.assertEqual(options()['jet_pt_scheme'],'baseline')
        self.assertEqual(options({'jet_pt_scheme':'hhbbgg_2024'})['jet_pt_scheme'],'hhbbgg_2024')
        for config in ({'jet_pt_scheme':'guess'},{'z_veto':'guess'},{'typo':'apply'}):
            with self.assertRaises(ValueError): options(config)

    def test_correction_errors(self):
        fn=self.ns['wwgg_validate_corrections']
        fn(['energy','weight'],{'energy':None},{'weight':None})
        for names in (['unknown'],['weight','weight'],'weight',[None]):
            with self.assertRaises(ValueError): fn(names,{'energy':None},{'weight':None})

    def test_exact_shared_selection(self):
        fn=self.ns['wwgg_select_jets']
        p=SimpleNamespace(wwgg_selection={'jet_pt_scheme':'baseline'})
        self.assertEqual(fn(p,1,2,3,4,'2024'),'baseline')
        p.wwgg_selection['jet_pt_scheme']='hhbbgg_2024'
        self.assertEqual(fn(p,1,2,3,4,'2024'),'eta')
        self.assertEqual(p.jet_pt_thresholds,[20,50,20])
        self.assertEqual(p.jet_eta_thresholds,[2.5,3.,4.7])
        self.ns['select_jets_eta_dependent'].assert_called_once_with(p,1,2,3,4)
        with self.assertRaises(ValueError): fn(p,1,2,3,4,'2023postBPix')

    def test_flow_disabled_and_data_unchanged(self):
        fn=self.ns['wwgg_apply_photon_flow']; ev=object()
        for kind,enabled in [('mc',False),('data',True)]:
            self.assertIs(fn(SimpleNamespace(data_kind=kind,doFlow_corrections=enabled),ev,'2024'),ev)

    def test_flow_reuses_helper_and_propagates_failure(self):
        fn=self.ns['wwgg_apply_photon_flow']
        class Events(dict):
            @property
            def Photon(self): return self['Photon']
        ev=Events(Photon='original')
        p=SimpleNamespace(data_kind='mc',doFlow_corrections=True,meta={'test':1},
                          resolve_flows_photonid_mva=Mock(return_value='mva'))
        module=ModuleType('higgs_dna.tools.flow_corrections')
        helper=Mock(return_value='corrected');module.apply_flow_corrections_to_photons=helper
        with patch.dict(sys.modules,{'higgs_dna.tools.flow_corrections':module}):
            self.assertIs(fn(p,ev,'2024'),ev)
            self.assertEqual(ev.Photon,'corrected')
            helper.assert_called_once_with('original',ev,p.meta,'2024','mva',self.ns['logger'],precision='float64')
            helper.side_effect=FileNotFoundError('missing model')
            with self.assertRaises(FileNotFoundError):fn(p,ev,'2024')


if __name__=='__main__':unittest.main(verbosity=2)

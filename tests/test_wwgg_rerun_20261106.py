import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare',ROOT/'production/wwgg/prepare_all_20261106.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Tests(unittest.TestCase):
    def records(self):
        records=[]
        for group,names in {'signal':['WWgg_FH_2024','WWgg_SL_2024','WWgg_FL_2024'],
                            'singleH':['H'],'continuum':['GG'],'data':['Run2024C_EG0']}.items():
            for name in names:
                path='/store/data/NANOv15/'+name+'.root' if group=='data' else '/store/mc/RunIII2024Summer24NanoAODv15/'+name+'.root'
                records.append({'group':group,'sample':name,'file':'root://host/'+path})
        return records

    def test_inventory_exact_and_duplicates_rejected(self):
        records=self.records();groups=module.previous_inventory(records)
        self.assertEqual(sorted(r['file'] for r in records),sorted(f for samples in groups.values() for files in samples.values() for f in files))
        with self.assertRaises(ValueError):module.previous_inventory(records+[records[0]])
        with self.assertRaises(ValueError):module.previous_inventory(records[1:])

    def test_all_mc_types_flow_no_trigger_and_new_calibration(self):
        template=json.loads((ROOT/'production/wwgg/production_signal_20261106.json').read_text())
        for name in ['WWgg_FH_2024','WWgg_SL_2024','WWgg_FL_2024','GGJets','WGG','TTGG','singleH']:
            config=module.sample_config(template,name,Path('sample.json'),False)
            self.assertEqual(config['corrections'][name][0],'Smearing')
            self.assertEqual(set(config['corrections']),{name})
            self.assertTrue(config['wwgg_corrections']['b_efficiency'])
            cmd=module.analysis_command('python',Path('config.json'),Path('output'),False)
            self.assertIn('--no-trigger',cmd);self.assertIn('--doFlow-corrections',cmd)
            self.assertEqual(cmd[cmd.index('--chunk')+1],'10000')
        self.assertEqual(set(template['corrections']),{'WWgg_FH_2024','WWgg_SL_2024','WWgg_FL_2024'})

    def test_data_scale_trigger_and_no_efficiency_collection(self):
        template=json.loads((ROOT/'production/wwgg/production_data_CDEI_20261106.json').read_text())
        config=module.sample_config(template,'Run2024G_EG1_v15',Path('sample.json'),True)
        self.assertEqual(config['corrections'],{'Run2024G_EG1_v15':['Scale']})
        self.assertFalse(config['wwgg_corrections']['b_efficiency'])
        cmd=module.analysis_command('python',Path('config.json'),Path('output'),True)
        self.assertNotIn('--no-trigger',cmd);self.assertNotIn('--doFlow-corrections',cmd)
        self.assertEqual(config['wwgg_corrections']['tau_veto'],'apply')
        self.assertEqual(config['wwgg_corrections']['b_veto'],'apply')


if __name__=='__main__':unittest.main(verbosity=2)

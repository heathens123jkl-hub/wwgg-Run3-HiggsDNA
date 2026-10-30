"""Same input cohort: baseline / corrections + veto flags / corrections + veto cuts."""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import tarfile
import traceback
from denominator_audit import snapshot, compare


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--entries',type=int,default=200)
    p.add_argument('--samples',type=Path)
    p.add_argument('--variations',nargs='*',default=[])
    a=p.parse_args(); repo=a.repo.resolve(); out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(repo));os.chdir(repo)
    import awkward as ak
    import numpy as np
    import torch
    from coffea.nanoevents import NanoEventsFactory,NanoAODSchema
    import higgs_dna.workflows.WWgg as wf
    cfg=json.loads((repo/'production/wwgg/corrections_2024.json').read_text())
    samples=json.loads((a.samples or repo/'production/wwgg/validation_samples_2024.json').read_text())
    meta=json.loads((repo/'higgs_dna/metaconditions/Era2023_postBPix_v1.json').read_text())
    report={'scope':'Integrated small-cohort validation, not production physics closure.', 'samples':{}}
    for sample,files in samples.items():
        record={};report['samples'][sample]=record
        try:
            cohort=None; stored=None; original_denominator=None
            modes=['baseline','integrated_store','integrated_apply']+a.variations
            for mode in modes:
                print('INTEGRATION_RUNNING',sample,mode,flush=True)
                captures=[]
                def capture(processor,array,filename,location,metadata,subdirs):captures.append((ak.to_packed(array),metadata))
                wf.dump_ak_array=capture
                kw=dict(schemaclass=NanoAODSchema,entry_start=0,entry_stop=a.entries,
                        metadata={'dataset':sample,'filename':files[0]})
                if 'delayed' in inspect.signature(NanoEventsFactory.from_root).parameters:kw['delayed']=False
                ev=NanoEventsFactory.from_root({files[0]:'Events'},**kw).events()
                keys=list(zip(ak.to_list(ev.run),ak.to_list(ev.luminosityBlock),ak.to_list(ev.event)))
                if cohort is None:cohort=keys
                elif cohort!=keys:raise ValueError('Input cohort changed')
                is_mc='GenPart' in ev.fields
                denominator=snapshot(ak.to_numpy(ev.genWeight)) if is_mc else None
                if is_mc:
                    if original_denominator is None:original_denominator=denominator
                    elif denominator!=original_denominator:raise ValueError('Input generator weights changed between modes')
                if not is_mc and mode in a.variations:continue
                icfg=dict(cfg)
                if mode=='integrated_store':icfg.update(tau_veto='store_only',b_veto='store_only')
                if mode in a.variations:icfg['variation']=mode
                corrections=['Smearing2G_IJazZ','TriggerSF','Pileup','LoosePhoIDSF','ElectronVetoSF','PreselSF'] if is_mc else ['Scale2G_IJazZ']
                np.random.seed(29102026);torch.manual_seed(29102026)
                processor=wf.WWggProcessor(metaconditions=meta,nano_version=15,
                    corrections={sample:corrections},systematics={},year={sample:['2024']},
                    apply_trigger=not is_mc, doFlow_corrections=is_mc and mode!='baseline',
                    output_location=str(out),output_format='parquet',
                    wwgg_corrections=None if mode=='baseline' else icfg)
                cutflow=processor.process(ev)
                sub=out/sample/mode;sub.mkdir(parents=True,exist_ok=True)
                data={'cutflow':cutflow,'input_rows':len(keys),'tables':len(captures)}
                if len(captures)>1:raise ValueError('Unexpected multiple dumps')
                if captures:
                    t,metadata=captures[0]
                    data['metadata']=metadata
                    if is_mc:
                        data['denominator_audit']=compare(metadata['sum_genw_presel'],denominator)
                        print('DENOMINATOR_AUDIT',sample,mode,json.dumps(data['denominator_audit']),flush=True)
                        if data['denominator_audit']['status']!='PASS':
                            (sub/'result.json').write_text(json.dumps(data,indent=2,default=str))
                            raise ValueError('Generator denominator differs from exact pre-selection numpy/serialization reference')
                    if not bool(ak.all(np.isfinite(t.weight))):raise ValueError('Nonfinite output weight')
                    ak.to_parquet(t,sub/'events.parquet')
                    data['categories']={label:{'rows':int(ak.sum(t.category==cat)),
                        'sumw':float(ak.sum(t.weight[t.category==cat]))} for cat,label in enumerate(('FH','SL','FL'))}
                    if mode=='integrated_store':stored=t
                    if mode=='integrated_apply' and stored is not None:
                        expected=stored[~stored.tau_veto & ~stored.b_veto]
                        for f in ('run','luminosityBlock','event','category','weight'):
                            np.testing.assert_array_equal(ak.to_numpy(t[f]),ak.to_numpy(expected[f]))
                        data['veto_filter_identity']='PASS'
                elif mode=='integrated_apply' and stored is not None:
                    assert int(ak.sum(~stored.tau_veto & ~stored.b_veto))==0
                (sub/'result.json').write_text(json.dumps(data,indent=2,default=str))
                record[mode]=data
                print('INTEGRATION_READY',sample,mode,json.dumps(data.get('categories',{})),flush=True)
            record['status']='COMPLETE'
        except Exception:
            record['status']='FAILED';record['error']=traceback.format_exc();print(record['error'],flush=True)
    report['status']='COMPLETE' if all(x['status']=='COMPLETE' for x in report['samples'].values()) else 'INCOMPLETE'
    (out/'summary.json').write_text(json.dumps(report,indent=2,default=str))
    archive=out.parent/(out.name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as handle:
        for f in sorted(out.rglob('*')):
            if f.is_file():handle.add(f,arcname=str(f.relative_to(out.parent)),recursive=False)
    print('INTEGRATION_STATUS',report['status']);print('DOWNLOAD',archive)
    if report['status']!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()

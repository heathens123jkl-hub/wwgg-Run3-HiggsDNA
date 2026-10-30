"""2024 UParT Tight b-veto flags on the existing selected WWgg jets."""
import gzip
import hashlib
import json
from pathlib import Path
import awkward as ak
import numpy as np


def load_b_config(repo):
    payload=Path(repo)/'higgs_dna/systematics/JSONs/POG/BTV/2024_Summer24/btagging.json.gz'
    if not payload.is_file():
        payload=Path(__file__).parent/'BTV_2024_Summer24_btagging.json.gz'
    content=json.loads(gzip.decompress(payload.read_bytes()))
    correction=next(c for c in content['corrections'] if c['name']=='UParTAK4_wp_values')
    node=correction['data']
    if node['nodetype']!='category' or node['input']!='working_point':
        raise ValueError('Unexpected BTV working-point schema')
    threshold=float(next(c['value'] for c in node['content'] if c['key']=='T'))
    if not np.isfinite(threshold) or abs(threshold-0.4648)>1e-10:
        raise ValueError('2024 Tight UParT payload differs from reviewed value 0.4648')
    return dict(year='2024',tagger='UParT',branch='btagUParTAK4B',wp='T',threshold=threshold,
                pt_min=30.,abs_eta_max=2.5,payload=str(payload),
                payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
                jets='Existing WWgg sel_jets; no new object cleaning',
                note='Not the partner-analysis PNet definition; no b-tag SF applied')


def evaluate_b(jets,cfg):
    branch=cfg['branch']
    missing={'pt','eta',branch}-set(jets.fields)
    if missing: raise ValueError('Missing b-veto jet fields: '+repr(missing))
    threshold=cfg['threshold']
    if not np.isfinite(threshold) or not 0<threshold<1:
        raise ValueError('Invalid b-tag threshold')
    score=jets[branch]
    kin=ak.fill_none(np.isfinite(jets.pt)&np.isfinite(jets.eta)&
                     (jets.pt>cfg['pt_min'])&(abs(jets.eta)<cfg['abs_eta_max']),False)
    valid=ak.fill_none(np.isfinite(score)&(score>=0)&(score<=1),False)
    tagged=ak.fill_none(kin&valid&(score>=threshold),False)
    out={'b_trial_n_kin':ak.sum(kin,axis=1),
         'b_trial_n_tagged':ak.sum(tagged,axis=1),
         'b_trial_n_invalid_kin':ak.sum(kin&~valid,axis=1),
         'b_trial_veto':ak.any(tagged,axis=1),
         'b_trial_jet_pt':jets.pt,'b_trial_jet_eta':jets.eta,
         'b_trial_jet_score':score,'b_trial_jet_pass_kin':kin,
         'b_trial_jet_tagged':tagged}
    for field in ('hadronFlavour','partonFlavour'):
        if field in jets.fields: out['b_trial_jet_'+field]=jets[field]
    return out

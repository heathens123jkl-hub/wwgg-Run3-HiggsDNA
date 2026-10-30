"""2024 V5 JEC/JES + Type-1 PUPPI MET candidate; JER is not implemented.

Flat NanoAOD branch arrays in, fresh results out. Never accepts a previously
calibrated output as input. This numerical core is wired by wwgg_corrections.apply_jme.
Type-1 follows the supplied April 2026 PDF component equations (minus sign).
Muon-subtracted eta/phi use the NanoAOD delta branches. Agreement with the
production CMSSW recipe remains a separate validation requirement.
"""
import awkward as ak
import numpy as np

SOURCES = ('Absolute', 'Absolute_2024', 'BBEC1', 'BBEC1_2024', 'EC2',
           'EC2_2024', 'FlavorQCD', 'HF', 'HF_2024', 'RelativeBal', 'RelativeSample_2024')
COMMON = ('run', 'luminosityBlock', 'event', 'Rho_fixedGridRhoFastjetAll',
          'RawPuppiMET_pt', 'RawPuppiMET_phi', 'PuppiMET_pt', 'PuppiMET_phi')
JET = ('pt', 'mass', 'rawFactor', 'eta', 'phi', 'area', 'chEmEF', 'neEmEF',
       'muonSubtrFactor', 'muonSubtrDeltaEta', 'muonSubtrDeltaPhi')
LOW = ('rawPt', 'eta', 'phi', 'area', 'EmEF', 'muonSubtrFactor',
       'muonSubtrDeltaEta', 'muonSubtrDeltaPhi')
BRANCHES = list(COMMON) + ['Jet_' + x for x in JET] + ['CorrT1METJet_' + x for x in LOW]


def flat(x):
    return np.asarray(ak.to_numpy(ak.flatten(x, axis=1)), dtype=np.float64)


def finite(x, name):
    if not np.isfinite(x).all():
        raise ValueError('Non-finite ' + name)


def evaluate(correction, values):
    return np.asarray(correction.evaluate(*[
        np.asarray(values[i.name], dtype=np.float64) if i.type == 'real' else values[i.name]
        for i in correction.inputs]), dtype=np.float64)


class Calibrator:
    def __init__(self, cset, is_data):
        self.cset = cset
        self.tag = 'Summer24Prompt24_V5_' + ('DATA' if is_data else 'MC')

    def levels(self, values):
        """Both L1 and full JEC are evaluated from the same raw inputs."""
        raw = values['JetPt']
        if (raw < 0).any():
            raise ValueError('Negative raw pt')
        active = raw > 0
        full_factor = np.ones(len(raw))
        l1_factor = np.ones(len(raw))
        if active.any():
            v = {k: x[active] for k, x in values.items()}
            full_factor[active] = evaluate(self.cset.compound[self.tag + '_L1L2L3Res_AK4PFPuppi'], v)
            l1_factor[active] = evaluate(self.cset[self.tag + '_L1FastJet_AK4PFPuppi'], v)
        for f in (full_factor, l1_factor):
            finite(f, 'JEC factor')
            if (f <= 0).any():
                raise ValueError('Nonpositive JEC factor')
        return raw * full_factor, raw * l1_factor, full_factor

    def uncertainty(self, values, full_pt, source):
        if source not in SOURCES:
            raise ValueError('Unknown JES source: ' + source)
        result = np.zeros(len(full_pt))
        active = full_pt > 0
        if active.any():
            v = {k: x[active] for k, x in values.items()}
            v['JetPt'] = full_pt[active]
            key = 'Summer24Prompt24_V5_MC_Regrouped_' + source + '_AK4PFPuppi'
            result[active] = evaluate(self.cset[key], v)
        finite(result, 'JES uncertainty')
        # JES sources are signed nuisance responses, not nonnegative errors.
        # Preserve signs between jets; reject sentinels and nonpositive scales.
        if (np.abs(result) >= 1).any():
            raise ValueError('JES signed response outside (-1,1); inspect payload domain before extrapolating')
        return result


def ingredients(a, prefix, muon_subtracted):
    p = prefix + '_'
    ref = a[p + ('pt' if prefix == 'Jet' else 'rawPt')]
    counts = np.asarray(ak.to_numpy(ak.num(ref)), dtype=np.int64)
    raw = flat(ref)
    if prefix == 'Jet':
        raw = raw * (1 - flat(a[p + 'rawFactor']))
    eta, phi = flat(a[p + 'eta']), flat(a[p + 'phi'])
    if muon_subtracted:
        raw = raw * (1 - flat(a[p + 'muonSubtrFactor']))
        eta = eta + flat(a[p + 'muonSubtrDeltaEta'])
        phi = phi + flat(a[p + 'muonSubtrDeltaPhi'])
    phi = np.arctan2(np.sin(phi), np.cos(phi))
    values = {'JetPt': raw, 'JetEta': eta, 'JetPhi': phi,
              'JetA': flat(a[p + 'area']),
              'Rho': np.repeat(np.asarray(a['Rho_fixedGridRhoFastjetAll'], dtype=float), counts),
              'run': np.repeat(np.asarray(a['run'], dtype=float), counts)}
    for k, v in values.items():
        finite(v, prefix + '/' + k)
    em = flat(a[p + 'chEmEF']) + flat(a[p + 'neEmEF']) if prefix == 'Jet' else flat(a[p + 'EmEF'])
    finite(em, prefix + '/EM fraction')
    return {'values': values, 'counts': counts, 'em': em}


def contribution(block, full_pt):
    v = block['values']
    eligible = (full_pt > 15.) & (np.abs(v['JetEta']) < 5.2) & (block['em'] < .9)
    delta = np.where(eligible, full_pt - block['l1'], 0.)
    def event_sum(x):
        return np.asarray(ak.to_numpy(ak.sum(ak.unflatten(x, block['counts']), axis=1)), dtype=float)
    return (event_sum(delta * np.cos(v['JetPhi'])),
            event_sum(delta * np.sin(v['JetPhi'])), event_sum(eligible.astype(float)))


def calculate(a, cset, is_data, include_jes=True):
    missing = sorted(set(BRANCHES) - set(a.fields))
    if missing:
        raise ValueError('Missing fields: ' + ', '.join(missing))
    calibrator = Calibrator(cset, is_data)
    regular = ingredients(a, 'Jet', False)
    jet_pt, _, jet_factor = calibrator.levels(regular['values'])
    jet_mass = flat(a['Jet_mass']) * (1 - flat(a['Jet_rawFactor'])) * jet_factor
    finite(jet_mass, 'corrected mass')
    jets = {'nominal': {'pt': ak.unflatten(jet_pt, regular['counts']),
                        'mass': ak.unflatten(jet_mass, regular['counts'])}}
    blocks = []
    for prefix in ('Jet', 'CorrT1METJet'):
        block = ingredients(a, prefix, True)
        block['full'], block['l1'], _ = calibrator.levels(block['values'])
        blocks.append(block)
    raw_pt = np.asarray(a['RawPuppiMET_pt'], dtype=float)
    raw_phi = np.asarray(a['RawPuppiMET_phi'], dtype=float)
    finite(raw_pt, 'RawPuppiMET_pt'); finite(raw_phi, 'RawPuppiMET_phi')
    if (raw_pt < 0).any():
        raise ValueError('Negative raw MET pt')
    raw_x, raw_y = raw_pt * np.cos(raw_phi), raw_pt * np.sin(raw_phi)

    def rebuild(fulls):
        parts = [contribution(b, p) for b, p in zip(blocks, fulls)]
        x = raw_x - sum(p[0] for p in parts)
        y = raw_y - sum(p[1] for p in parts)
        return {'px': x, 'py': y, 'pt': np.hypot(x, y), 'phi': np.arctan2(y, x),
                'n_type1_jet': parts[0][2], 'n_type1_low': parts[1][2],
                'low_dx': parts[1][0], 'low_dy': parts[1][1]}
    mets = {'nominal': rebuild([b['full'] for b in blocks])}
    if include_jes and not is_data:
        for source in SOURCES:
            delta_jet = calibrator.uncertainty(regular['values'], jet_pt, source)
            delta_met = [calibrator.uncertainty(b['values'], b['full'], source) for b in blocks]
            for direction, sign in [('up', 1), ('down', -1)]:
                key = source + '_' + direction
                jets[key] = {'pt': ak.unflatten(jet_pt * (1 + sign * delta_jet), regular['counts']),
                             'mass': ak.unflatten(jet_mass * (1 + sign * delta_jet), regular['counts'])}
                mets[key] = rebuild([b['full'] * (1 + sign * d) for b, d in zip(blocks, delta_met)])
    return {'jets': jets, 'met': mets, 'blocks': blocks,
            'scope': 'JEC/JES numerical core; JER is applied by the WWgg adapter.'}

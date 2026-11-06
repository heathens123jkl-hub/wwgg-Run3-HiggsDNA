"""Prepare, then explicitly submit the 2024 nominal production on lxplus."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WGG = '/WGG-4FS_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM'
ASSETS = ('higgs_dna/systematics/JSONs', 'higgs_dna/systematics/data',
          'higgs_dna/tools/flows', 'higgs_dna/metaconditions')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def lfn(path):
    if '/store/' not in path:
        raise ValueError('Expected a CMS /store/ file: ' + path)
    return '/store/' + path.split('/store/', 1)[1]


def inventory(wgg_files):
    groups = {
        'signal': read(ROOT/'production/samples/samples_2024_signal.json'),
        'singleH': read(ROOT/'production/samples/samples_singleH_2024.json'),
        'data': read(ROOT/'production/samples/samples_data_2024_full.json'),
    }
    backgrounds = read(ROOT/'samples_2024_background.json')
    groups['continuum'] = {k: backgrounds[k] for k in
        ('GGJets_MGG-40to80_2024', 'GGJets_MGG-80_2024', 'TTGG_2024')}
    if not wgg_files:
        raise ValueError('WGG file discovery returned no files; refusing incomplete production')
    groups['continuum']['WGG_4FS_2024'] = wgg_files
    seen = {}
    removed = {}
    for group, samples in groups.items():
        for name, files in samples.items():
            if not files:
                raise ValueError('Empty sample: ' + name)
            unique = {}
            for path in files:
                key = lfn(path)
                if group == 'data':
                    if 'NANOv15' not in key:
                        raise ValueError('Non-v15 Data file: ' + key)
                elif 'RunIII2024Summer24NanoAODv15/' not in key:
                    raise ValueError('Unexpected MC campaign: ' + key)
                if name == 'WGG_4FS_2024' and ('/WGG-4FS_' not in key or
                    '/150X_mcRun3_2024_realistic_v2-v2/' not in key):
                    raise ValueError('WGG file does not match requested dataset: ' + key)
                if key in seen and seen[key] != name:
                    raise ValueError('File shared between samples: ' + key)
                seen[key] = name
                unique[key] = 'root://cms-xrd-global.cern.ch/' + key
            removed[name] = len(files) - len(unique)
            samples[name] = list(unique.values())
    return groups, removed


def copy_assets(source):
    pending = []
    for directory in ASSETS:
        for src in (source/directory).rglob('*'):
            if src.is_file() and '__pycache__' not in src.parts and src.suffix not in ('.py', '.pyc'):
                dest = ROOT/src.relative_to(source)
                if not dest.exists():
                    pending.append((src, dest))
    size = sum(src.stat().st_size for src, _ in pending)
    if size > 3*1024**3 or shutil.disk_usage(ROOT).free < size + 512*1024**2:
        raise RuntimeError('Insufficient space or assets exceed 3 GiB; inspect AFS quota')
    for src, dest in pending:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    print('ASSETS_COPIED', len(pending), 'BYTES', size)
    for sub, names in {
        '2024_model': ['best_model_.pth', 'input_means.npy', 'input_std.npy',
                       'conditions_means.npy', 'conditions_std.npy'],
        'run3_mvaID_models': ['Egamma_Run3_photonIDMVA_EB.json', 'Egamma_Run3_photonIDMVA_EE.json'],
    }.items():
        for name in names:
            path = ROOT/'higgs_dna/tools/flows'/sub/name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)


def previous_inventory(records):
    groups = {}; seen = set()
    for r in records:
        group, name, path = r['group'], r['sample'], r['file']
        if group not in ('signal','singleH','data','continuum'):
            raise ValueError('Unexpected group: '+group)
        key = lfn(path)
        if key in seen: raise ValueError('Duplicate previous input: '+key)
        seen.add(key)
        if group == 'data':
            if 'NANOv15' not in key: raise ValueError('Non-v15 Data input')
        elif 'RunIII2024Summer24NanoAODv15/' not in key:
            raise ValueError('Unexpected MC campaign')
        groups.setdefault(group,{}).setdefault(name,[]).append(path)
    if set(groups) != {'signal','singleH','data','continuum'}:
        raise ValueError('Incomplete previous production groups')
    if set(groups['signal']) != {'WWgg_FH_2024','WWgg_SL_2024','WWgg_FL_2024'}:
        raise ValueError('Missing signal decay modes')
    return groups


def validated_proxy(path):
    path = path.resolve()
    if not str(path).startswith('/afs/') or not path.is_file():
        raise ValueError('Use a persistent proxy file under AFS, not lxplus /tmp')
    subprocess.run(['voms-proxy-info','-file',str(path),'-exists','-valid','24:00'],check=True)
    return str(path)


def validate_required_assets():
    checks = {
        'higgs_dna/systematics/JSONs/scaleAndSmearing/Hgg/EGMScalesSmearing_ZeeZmmg_2024.v1.json':
            '5dce7711ba27d07172d353f7ddcb17ad09619d13d1cf02335f02c38e78340c5c',
        'higgs_dna/metaconditions/CAF/certification/Collisions24/Cert_Collisions2024_378981_386951_Golden.json':
            '32572466413b0b2ee72ebe18b1e556e7e083f550e74e596830be4f10aa25bf9d',
    }
    for rel, expected in checks.items():
        f=ROOT/rel
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=expected:
            raise ValueError('Missing or changed validated asset: '+rel)


def sample_config(template, name, samples_path, data):
    config=deepcopy(template)
    config['samplejson']=str(samples_path)
    config['year']={name:['2024']}
    config['corrections']={name:['Scale'] if data else
        ['Smearing','TriggerSF','Pileup','LoosePhoIDSF','ElectronVetoSF','PreselSF']}
    config['wwgg_corrections']['b_efficiency']=not data
    return config


def analysis_command(python, config_path, dest, data):
    cmd=[python,str(ROOT/'higgs_dna/scripts/run_analysis.py'),
         '--json-analysis',str(config_path),'--nano-version','15',
         '--executor','iterative','--chunk','10000','--timeout','300',
         '--dump',str(dest/'parquet'),'--save',str(dest/'summary.coffea')]
    if not data:cmd+=['--no-trigger','--doFlow-corrections']
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--assets-from', type=Path)
    parser.add_argument('--previous-production', type=Path, help='Previous EOS production; reuse the exact input inventory')
    parser.add_argument('--proxy', type=Path, default=Path('/afs/cern.ch/user/x/'+os.environ.get('USER','')+'/private/x509up_wwgg'))
    parser.add_argument('--max-materialize', type=int, default=100)
    args = parser.parse_args()
    os.chdir(ROOT)
    if not str(ROOT).startswith('/afs/') or not str(args.output).startswith('/eos/'):
        raise ValueError('Require code under /afs and an absolute /eos output directory')
    jobs = ROOT/'production_jobs_20261106'
    if args.action == 'submit':
        plan = read(jobs/'plan.json')
        if str(args.output) != plan['output']:
            raise ValueError('Output differs from prepared plan')
        if (jobs/'submission_started').exists():
            raise RuntimeError('Submission already attempted; inspect submit.log/condor_q before retrying')
        for name, expected in plan['hashes'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Prepared source/config/assets changed: ' + name)
        proxy = validated_proxy(args.proxy)
        (jobs/'submission_started').write_text('Submission attempt started; do not automatically resubmit.\n')
        result = subprocess.run(['condor_submit', '-terse', '-append', 'x509userproxy = '+proxy,
                                 str(jobs/'all.sub')], text=True, capture_output=True)
        (jobs/'submit.log').write_text(result.stdout + result.stderr)
        print(result.stdout, result.stderr, flush=True)
        result.check_returncode()
        return
    if jobs.exists() or args.output.exists():
        raise FileExistsError('Use a new checkout and a fresh EOS output path; previous production is preserved')
    if not args.assets_from:
        raise ValueError('--assets-from must point to the previously validated installation')
    copy_assets(args.assets_from.resolve())
    if not args.previous_production:
        raise ValueError('--previous-production is required: this rerun preserves the full previous sample inventory')
    previous = args.previous_production.resolve()
    if previous == args.output.resolve():
        raise ValueError('New output must not be the previous production')
    previous_records = read(previous/'job_inputs.json')
    groups = previous_inventory(previous_records)
    removed = {name:0 for samples in groups.values() for name in samples}
    validate_required_assets()
    validated_proxy(args.proxy)
    if args.max_materialize < 1:
        raise ValueError('--max-materialize must be positive')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run([sys.executable, str(ROOT/'higgs_dna/scripts/run_analysis.py'), '--help'],
                   env=env, check=True, stdout=subprocess.DEVNULL)
    jobs.mkdir()
    (jobs/'logs').mkdir()
    args.output.mkdir(parents=True)
    templates = {False: read(ROOT/'production/wwgg/production_signal_20261106.json'),
                 True: read(ROOT/'production/wwgg/production_data_CDEI_20261106.json')}
    job_records, summary = [], []
    for group, samples in groups.items():
        for name, files in samples.items():
            summary.append({'group': group, 'sample': name, 'files': len(files)})
            for path in files:
                index = len(job_records)
                samples_path = jobs/f'samples_{index}.json'
                config_path = jobs/f'config_{index}.json'
                write(samples_path, {name: [path]})
                data = group == 'data'
                config = sample_config(templates[data], name, samples_path, data)
                write(config_path, config)
                dest = args.output/group/name/f'file_{index:06d}'
                dest.mkdir(parents=True)
                cmd = analysis_command(sys.executable, config_path, dest, data)
                job_records.append({'group': group, 'sample': name, 'file': path,
                                    'output': str(dest), 'command': cmd})
    write(jobs/'jobs.json', job_records)
    worker = ROOT/'production/wwgg/run_prepared_job_20261106.py'
    executable = jobs/'worker.sh'
    executable.write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+
        '\nexport PYTHONPATH='+shlex.quote(str(ROOT))+'${PYTHONPATH:+:$PYTHONPATH}\nexec '+
        shlex.quote(sys.executable)+' '+shlex.quote(str(worker))+' "$1"\n')
    executable.chmod(0o755)
    sub = '\n'.join([
        'universe = vanilla', 'executable = '+str(executable), 'arguments = $(ProcId)',
        'output = '+str(jobs/'logs/$(ClusterId).$(ProcId).out'),
        'error = '+str(jobs/'logs/$(ClusterId).$(ProcId).err'),
        'log = '+str(jobs/'logs/$(ClusterId).log'),
        'getenv = True', 'use_x509userproxy = True', 'x509userproxy = '+str(args.proxy), 'should_transfer_files = NO',
        'request_cpus = 1', 'request_memory = 12000', '+JobFlavour = "tomorrow"',
        'max_materialize = '+str(args.max_materialize),
        'on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)',
        'queue '+str(len(job_records)), ''])
    (jobs/'all.sub').write_text(sub)
    tracked = subprocess.check_output(['git', 'ls-files'], text=True).splitlines()
    files_to_hash = {ROOT/p for p in tracked if (ROOT/p).is_file()}
    for directory in ASSETS:
        files_to_hash.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix != '.pyc')
    files_to_hash.update(p for p in jobs.glob('*') if p.is_file())
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files_to_hash}
    plan = {'output': str(args.output), 'python': sys.executable,
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'samples': summary, 'jobs': len(job_records), 'removed_duplicate_files': removed,
            'wgg_dataset': WGG, 'hashes': hashes,
            'previous_production': str(previous),
            'previous_inventory_sha256': hashlib.sha256((previous/'job_inputs.json').read_bytes()).hexdigest(),
            'chunk': 10000, 'diphoton_fiducial': 'classical_noIso',
            'pending_efficiency_sfs': ['b_veto', 'tau_veto']}
    write(jobs/'plan.json', plan)
    write(args.output/'production_plan.json', plan)
    write(args.output/'job_inputs.json', job_records)
    for record in summary:
        print(record['group'], record['sample'], record['files'])
    print('PREPARED', len(job_records), 'jobs; not yet submitted')


if __name__ == '__main__':
    main()

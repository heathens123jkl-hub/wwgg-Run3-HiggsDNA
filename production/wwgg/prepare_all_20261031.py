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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--assets-from', type=Path)
    parser.add_argument('--wgg-files', type=Path, help='Optional JSON list; otherwise query DAS')
    parser.add_argument('--max-materialize', type=int, default=100)
    args = parser.parse_args()
    os.chdir(ROOT)
    if not str(ROOT).startswith('/afs/') or not str(args.output).startswith('/eos/'):
        raise ValueError('Require code under /afs and an absolute /eos output directory')
    jobs = ROOT/'production_jobs_20261031'
    if args.action == 'submit':
        plan = read(jobs/'plan.json')
        if str(args.output) != plan['output']:
            raise ValueError('Output differs from prepared plan')
        if (jobs/'submission_started').exists():
            raise RuntimeError('Submission already attempted; inspect submit.log/condor_q before retrying')
        for name, expected in plan['hashes'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Prepared source/config/assets changed: ' + name)
        proxy = subprocess.check_output(['voms-proxy-info', '-path'], text=True).strip()
        subprocess.run(['voms-proxy-info', '-exists', '-valid', '24:00'], check=True)
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
    if args.wgg_files:
        wgg_files = read(args.wgg_files)
        if not isinstance(wgg_files, list):
            raise ValueError('--wgg-files must contain a JSON list of CMS file paths')
    else:
        das = '/cvmfs/cms.cern.ch/common/dasgoclient'
        output = subprocess.check_output([das, '-query=file dataset='+WGG], text=True, timeout=300)
        wgg_files = [line.strip() for line in output.splitlines() if line.strip()]
    groups, removed = inventory(wgg_files)
    if args.max_materialize < 1:
        raise ValueError('--max-materialize must be positive')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run([sys.executable, str(ROOT/'higgs_dna/scripts/run_analysis.py'), '--help'],
                   env=env, check=True, stdout=subprocess.DEVNULL)
    jobs.mkdir()
    (jobs/'logs').mkdir()
    args.output.mkdir(parents=True)
    templates = {False: read(ROOT/'production/wwgg/production_signal_20261030.json'),
                 True: read(ROOT/'production/wwgg/production_data_CDEI_20261030.json')}
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
                config = deepcopy(templates[data])
                config['samplejson'] = str(samples_path)
                config['year'] = {name: ['2024']}
                config['corrections'] = {name: ['Scale2G_IJazZ'] if data else
                    ['Smearing2G_IJazZ', 'TriggerSF', 'Pileup', 'LoosePhoIDSF', 'ElectronVetoSF', 'PreselSF']}
                config['wwgg_corrections']['b_efficiency'] = not data
                write(config_path, config)
                dest = args.output/group/name/f'file_{index:06d}'
                dest.mkdir(parents=True)
                cmd = [sys.executable, str(ROOT/'higgs_dna/scripts/run_analysis.py'),
                       '--json-analysis', str(config_path), '--nano-version', '15',
                       '--executor', 'iterative', '--timeout', '300', '--dump', str(dest/'parquet'),
                       '--save', str(dest/'summary.coffea')]
                if not data:
                    cmd += ['--no-trigger', '--doFlow-corrections']
                job_records.append({'group': group, 'sample': name, 'file': path,
                                    'output': str(dest), 'command': cmd})
    write(jobs/'jobs.json', job_records)
    worker = ROOT/'production/wwgg/run_prepared_job_20261031.py'
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
        'getenv = True', 'use_x509userproxy = True', 'should_transfer_files = NO',
        'request_cpus = 1', 'request_memory = 10240', '+JobFlavour = "tomorrow"',
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
            'wgg_dataset': WGG, 'hashes': hashes}
    write(jobs/'plan.json', plan)
    write(args.output/'production_plan.json', plan)
    write(args.output/'job_inputs.json', job_records)
    for record in summary:
        print(record['group'], record['sample'], record['files'])
    print('PREPARED', len(job_records), 'jobs; not yet submitted')


if __name__ == '__main__':
    main()

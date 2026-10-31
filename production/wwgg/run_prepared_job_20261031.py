"""Run one frozen input file; keep output partitions and failure status."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
os.chdir(root)
records = json.loads((root/'production_jobs_20261031/jobs.json').read_text())
record = records[int(sys.argv[1])]
dest = Path(record['output'])
done = dest/'SUCCESS.json'
if done.exists():
    raise RuntimeError('Successful output already exists; refusing duplicate execution')
env = dict(os.environ)
env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
env['OMP_NUM_THREADS'] = '1'
env['OPENBLAS_NUM_THREADS'] = '1'
env['MKL_NUM_THREADS'] = '1'
result = subprocess.run(record['command'], env=env)
if result.returncode:
    sys.exit(result.returncode)
if not (dest/'summary.coffea').is_file():
    raise RuntimeError('Runner returned success without the denominator/cutflow output')
done.write_text(json.dumps({'input': record['file'], 'sample': record['sample']})+'\n')

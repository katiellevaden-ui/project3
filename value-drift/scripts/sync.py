#!/usr/bin/env python3
"""Lead-only SSH sync. Never copy credentials, account-control files, or caches."""
import argparse
import json
from pathlib import Path
import subprocess

def main():
    p=argparse.ArgumentParser()
    p.add_argument('direction',choices=['push','pull'])
    p.add_argument('--checkpoints',action='store_true')
    p.add_argument('--connection',default='runs/connection.json')
    p.add_argument('--siblings',action='store_true',help='push: also sync ../value-drift-bench and ../objective-editing (bench loop imports them)')
    a=p.parse_args(); c=json.loads(Path(a.connection).read_text())
    transport=f"ssh -i {c['key_path']} -p {c['port']}"
    remote=f"root@{c['host']}:/workspace/value-drift/"
    args=['rsync','-rz','-e',transport]
    for pattern in ['.git','.env','.env.*','.venv','OpenCharacterTraining','__pycache__','.pytest_cache']:
        args+=['--exclude',pattern]
    if a.direction=='push':
        if a.siblings:
            here=Path(__file__).resolve().parents[1]
            for name in ['value-drift-bench','objective-editing']:
                sib=here.parent/name
                subprocess.run(args+['--exclude','runs','--exclude','logs','--exclude','results','--exclude','*.egg-info',
                                     f'{sib}/',f"root@{c['host']}:/workspace/{name}/"],check=True)
        args+=['--exclude','runs','--exclude','checkpoints', './',remote]
    else:
        for pattern in ['spending.json','connection.json','budget-watchdog*']:
            args+=['--exclude',pattern]
        if not a.checkpoints:
            args+=['--exclude','*.safetensors','--exclude','*.bin','--exclude','*.pt']
        args+=[remote+'runs/','./runs/']
    subprocess.run(args,check=True)
if __name__=='__main__':main()

"""Check recorded runs, archive integrity and optional schedule feasibility."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter
from summarise import ROOT,records,read_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hashes',action='store_true',help='Verify every archived uncompressed file hash.')
    parser.add_argument('--schedules',action='store_true',help='Recheck stored schedules independently.')
    args=parser.parse_args()
    rows=list(records())
    counts=Counter(row['study'] for row in rows)
    assert counts=={'comparison':4320,'development':2760,'mechanism':3360},counts
    identifiers=[(row['study'],row['configuration'],row['instance'],row['seed']) for row in rows]
    assert len(identifiers)==len(set(identifiers))
    assert all(1<=row['seed']<=30 and row['best_tpm']>0 for row in rows)
    for row in rows:
        if row['study']=='comparison' and row['configuration'] in ['MA_EA','SAMA_EA']:
            assert row['sgs']==(10000 if row['configuration']=='MA_EA' else 6000)
    sys.path.insert(0,str(ROOT/'src'))
    from problem_loader import load_instance,load_dev_instance,validate_schedule
    instances={}
    verified=0
    for study in ['comparison','mechanism']:
        for path in sorted((ROOT/'results'/study).rglob('result.json')):
            row=read_json(path)
            assert row['valid_schedule'] and row['validation']['tpm']==row['best_tpm'],path
            if args.schedules:
                key=(study,row['instance'])
                if key not in instances:
                    instances[key]=(load_instance(row['instance']) if study=='comparison' else load_dev_instance(row['instance']))
                validation=validate_schedule(instances[key],row['start'],row['finish'])
                assert validation['tpm']==row['best_tpm'],path
                verified+=1
                if verified%1000==0: print(f'Validated {verified} schedules.',flush=True)
    with (ROOT/'results/files.csv').open(encoding='utf-8',newline='') as stream:
        inventory=list(csv.DictReader(stream))
    for row in inventory:
        path=ROOT/row['path']
        assert path.is_file(),path
        if args.hashes:
            raw=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
            assert len(raw)==int(row['uncompressed_bytes'])
            assert hashlib.sha256(raw).hexdigest()==row['sha256_uncompressed'],path
    print(json.dumps({'runs':dict(counts),'archived_files':len(inventory),
                      'schedules_rechecked':verified,'hashes_checked':args.hashes,'status':'passed'}))

if __name__=='__main__':main()

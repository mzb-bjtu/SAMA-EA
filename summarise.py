"""Recompute descriptive summaries from the archived run-level results."""
from pathlib import Path
import csv
import gzip
import json
import statistics as st

ROOT=Path(__file__).resolve().parent

def read_json(path):
    raw=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
    return json.loads(raw)

def records():
    for study in ['comparison','mechanism']:
        for path in sorted((ROOT/'results'/study).rglob('result.json')):
            row=read_json(path)
            yield {'study':study,'configuration':path.relative_to(ROOT/'results'/study).parts[0],
                   'instance':row['instance'],'seed':row['seed'],'best_tpm':row['best_tpm'],
                   'sgs':None if study=='comparison' and path.relative_to(ROOT/'results'/study).parts[0]=='PSGS_MINSLK' else row.get('n_exact_sgs'),
                   'time_s':row['wall_time_s']}
    for path in sorted((ROOT/'results/development').glob('*/runs.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row.get('ok'):
                yield {'study':'development','configuration':path.parent.name,
                       'instance':row['instance'],'seed':row['seed'],
                       'best_tpm':row['best_tpm'],'sgs':row['n_exact_sgs'],'time_s':row['wall']}

def summarise(output):
    groups={}
    for row in records():
        key=(row['study'],row['configuration'],row['instance'])
        groups.setdefault(key,[]).append(row)
    rows=[]
    for key,values in sorted(groups.items()):
        assert len(values)==30 and {v['seed'] for v in values}==set(range(1,31)),key
        quality=[v['best_tpm'] for v in values]
        work=[v['sgs'] for v in values if v['sgs'] is not None]
        rows.append(dict(zip(['study','configuration','instance'],key))|
                    {'runs':len(values),'mean_tpm':st.mean(quality),'sd_tpm':st.stdev(quality),
                     'mean_sgs':st.mean(work) if work else '',
                     'sd_sgs':st.stdev(work) if len(work)>1 else '',
                     'mean_time_s':st.mean(v['time_s'] for v in values)})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    return rows

if __name__=='__main__':
    target=ROOT/'results/per_instance.csv'
    rows=summarise(target)
    print(f'{len(rows)} complete instance/configuration groups; wrote {target.name}.')

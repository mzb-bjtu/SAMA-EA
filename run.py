"""Run comparison, development and mechanism configurations in separate workers."""
from pathlib import Path
import argparse
import concurrent.futures as cf
from dataclasses import asdict
import json
import multiprocessing as mp
import os
import sys

ROOT=Path(__file__).resolve().parent
METHODS=['MA_EA','SAMA_EA','B23_GA_CB','PSGS_MINSLK','QLESCA','SADE']

def initialise_worker():
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        os.environ[key]='1'
    sys.path.insert(0,str(ROOT/'src'))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

def normalise(value):
    if isinstance(value,dict): return {str(k):normalise(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [normalise(v) for v in value]
    if hasattr(value,'item'): return value.item()
    return value

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(normalise(value),ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)

def execute(job):
    from problem_loader import load_instance,load_dev_instance
    from core_engine import DecoderConfig,run_variant
    kind,name,instance,seed=job['study'],job['configuration'],job['instance'],job['seed']
    folder=Path(job['output'])/kind/name/instance/f'seed_{seed:03d}'
    if (folder/'result.json').is_file():
        return {'configuration':name,'instance':instance,'seed':seed,'status':'existing'}
    inst=load_instance(instance) if kind=='comparison' else load_dev_instance(instance)
    common={'seed':seed,'n_gen':job['generations'] or 200,'pop_size':job['population'] or 50}
    if kind=='comparison' and name in ['B23_GA_CB','PSGS_MINSLK','QLESCA','SADE']:
        from comparators import run_comparator
        outcome=run_comparator(inst,algorithm=name,seed=seed,
                               pop_size=common['pop_size'],generations=common['n_gen'])
    elif kind=='comparison':
        outcome=run_variant(inst,method=name,use_surrogate=name=='SAMA_EA',
                            use_meta_adaptation=True,**common)
    elif kind=='development':
        use_meta,use_sur,decoder_args,extra=job['specification']
        extra=dict(extra)
        decoder_args=dict(decoder_args)
        decoder_kind=extra.pop('decoder','pinned')
        for key in ['exploit']:
            if key in extra: decoder_args[key]=extra.pop(key)
        configured_pop=extra.pop('pop',50)
        configured_gen=extra.pop('gen',200)
        common['pop_size']=job['population'] or configured_pop
        common['n_gen']=job['generations'] or configured_gen
        dec=(DecoderConfig.adaptive(rekey='schedule',pool=('realign',),**decoder_args)
             if decoder_kind=='adaptive' else DecoderConfig.pinned(**decoder_args))
        outcome=run_variant(inst,method='SAMA_EA' if use_sur else 'MA_EA',
            use_surrogate=use_sur,use_meta_adaptation=use_meta,decode_config=dec,
            parent_fraction=extra.pop('parent_fraction',.5),renew=extra.pop('renew',None),
            sa=extra.pop('sa',None),surrogate_type=extra.pop('sur_type',None),
            surrogate_params=extra.pop('sur_params',None),**common)
        if extra: raise ValueError(f'Unused settings: {extra}')
    else:
        cfg=job['specification']
        dec=DecoderConfig.pinned('realign',override_params={'rounds':1,'bw':1},
                                meta_mode='joint',slots=4,de_mut_p=.05,
                                entropy_coef=cfg['entropy_coef'])
        outcome=run_variant(inst,method=cfg['method'],use_surrogate=cfg['method']=='SAMA_EA',
            use_meta_adaptation=True,decode_config=dec,sa={'fraction':.6},
            budget={'mode':cfg['budget_mode'],'stagnation_scale':cfg['stagnation_scale'],
                    'decay_rate':cfg['decay_rate']},capture_mechanism=True,**common)
    payload=asdict(outcome)
    assert payload['valid_schedule'],(name,instance,seed)
    traces=payload.pop('convergence')
    payload.update(study=kind,configuration=name,instance=instance,seed=seed)
    atomic_json(folder/'config.json',job)
    atomic_json(folder/'convergence.json',{'rows':traces})
    atomic_json(folder/'result.json',payload)
    return {'configuration':name,'instance':instance,'seed':seed,'status':'completed',
            'best_tpm':payload['best_tpm'],'n_exact_sgs':payload['n_exact_sgs']}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',choices=['comparison','development','mechanism'])
    parser.add_argument('--configurations',nargs='+')
    parser.add_argument('--instances',nargs='+')
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(1,31)))
    parser.add_argument('--workers',type=int,default=30)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--population',type=int,help='Override population for a small verification run.')
    parser.add_argument('--generations',type=int,help='Override generations for a small verification run.')
    args=parser.parse_args()
    sys.path.insert(0,str(ROOT/'src'))
    from experiment_config import VALIDATION_INSTANCES,DEVELOPMENT_INSTANCES
    if args.study=='comparison': specs={name:None for name in METHODS}
    else:
        saved=json.loads((ROOT/'configs'/f'{args.study}.json').read_text(encoding='utf-8'))
        specs=saved if isinstance(saved,dict) else {row['name']:row for row in saved}
    names=args.configurations or list(specs)
    allowed=VALIDATION_INSTANCES if args.study=='comparison' else DEVELOPMENT_INSTANCES
    instances=args.instances or list(allowed)
    if not set(names)<=set(specs): parser.error('Unknown configuration.')
    if not set(instances)<=set(allowed): parser.error('Instance belongs to another experimental set.')
    if args.workers<1: parser.error('workers must be positive')
    if args.population is not None and args.population<4: parser.error('population must be at least 4')
    if args.generations is not None and args.generations<2: parser.error('generations must be at least 2')
    args.output=args.output.resolve()
    archive=(ROOT/'results').resolve()
    if args.output==archive or archive in args.output.parents:
        parser.error('New runs must use an output directory outside the archived results.')
    jobs=[dict(study=args.study,configuration=name,instance=i,seed=s,
               specification=specs[name],population=args.population,
               generations=args.generations,output=str(args.output))
          for name in names for i in instances for s in args.seeds]
    with cf.ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                               initializer=initialise_worker) as pool:
        for result in pool.map(execute,jobs): print(json.dumps(result),flush=True)

if __name__=='__main__': main()

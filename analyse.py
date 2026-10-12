"""Recompute comparison statistics and development quality gaps."""
from pathlib import Path
from collections import defaultdict
import json
import statistics as st
from scipy.stats import friedmanchisquare,rankdata,wilcoxon,f
from summarise import ROOT,records

BASELINES=['B23_GA_CB','PSGS_MINSLK','QLESCA','SADE']
METHODS=['MA_EA','SAMA_EA']+BASELINES

def gap(a,b,instances):
    return st.mean(100*(st.mean(a[i])-st.mean(b[i]))/st.mean(b[i]) for i in instances)

def holm(values):
    order=sorted(range(len(values)),key=values.__getitem__)
    adjusted=[0.]*len(values);previous=0.
    for position,index in enumerate(order):
        previous=max(previous,min(1.,(len(values)-position)*values[index]))
        adjusted[index]=previous
    return adjusted

def paired(a,b):
    differences=[x-y for x,y in zip(a,b)]
    return float(wilcoxon(differences).pvalue) if any(differences) else 1.

def main():
    rows=list(records())
    grouped=defaultdict(lambda:defaultdict(dict))
    for row in rows: grouped[(row['study'],row['configuration'])][row['instance']][row['seed']]=row['best_tpm']
    comparison={name:{i:list(seed_values.values()) for i,seed_values in grouped[('comparison',name)].items()} for name in METHODS}
    instances=sorted(comparison['MA_EA'])
    means={name:[st.mean(comparison[name][i]) for i in instances] for name in METHODS}
    ranks={name:0. for name in METHODS}
    for i in range(len(instances)):
        for name,rank in zip(METHODS,rankdata([means[a][i] for a in METHODS])): ranks[name]+=float(rank)/len(instances)
    families={}
    for proposed in ['MA_EA','SAMA_EA']:
        family=[proposed]+BASELINES
        statistic,p=friedmanchisquare(*(means[name] for name in family))
        n,k=len(instances),len(family)
        iman=(n-1)*statistic/(n*(k-1)-statistic)
        ps=[paired(means[proposed],means[baseline]) for baseline in BASELINES]
        families[proposed]={'friedman':float(statistic),'friedman_p':float(p),
            'iman_davenport_f':float(iman),'iman_davenport_p':float(f.sf(iman,k-1,(k-1)*(n-1))),
            'baselines':{name:{'gap_percent':gap(comparison[proposed],comparison[name],instances),
                'wilcoxon_p':raw,'holm_p':adjusted,
                'wins':sum(a<b for a,b in zip(means[proposed],means[name])),
                'losses':sum(a>b for a,b in zip(means[proposed],means[name])),
                'ties':sum(a==b for a,b in zip(means[proposed],means[name]))}
                for name,raw,adjusted in zip(BASELINES,ps,holm(ps))}}
    development={}
    for (study,name),instances_map in grouped.items():
        if study!='development':continue
        reference='B2_SAMA_y2' if name.startswith(('S2','S3','S4','S5')) else 'C0_Y2'
        baseline=grouped[('development',reference)]
        identities=sorted((i,s) for i in instances_map for s in instances_map[i])
        a=[instances_map[i][s] for i,s in identities];b=[baseline[i][s] for i,s in identities]
        development[name]={'reference':reference,'mean_tpm':st.mean(a),
            'gap_percent':gap({i:list(v.values()) for i,v in instances_map.items()},
                              {i:list(v.values()) for i,v in baseline.items()},sorted(instances_map)),
            'paired_p':paired(a,b)}
    payload={'mean_ranks':ranks,'comparison_families':families,'development':development}
    target=ROOT/'results/statistics.json'
    target.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({'mean_ranks':ranks,'statistics_file':str(target)},indent=2))

if __name__=='__main__':main()

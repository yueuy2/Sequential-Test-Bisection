"""Subset scheduling and original summary formulas; all kernels remain unchanged."""
import argparse, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import hpc_simulation as sim
from auxiliary_summary import summary, pow10, wilson

def tasks():
    coverage=[('stb',s,r,0,0.) for s in (4,5,6,7) for r in range(200)]
    cubic=[('astb',1,r,10**k,0.) for k in range(3,10) for r in range(400 if k<=6 else 200)]
    return coverage+cubic

def summarize():
    expected=tasks();records={}
    for task in expected:
        path=sim.TASKS/(sim.task_id(*task[:3],n=task[3])+'.pkl')
        with path.open('rb') as f:record=pickle.load(f)
        assert tuple(record['task'])==task
        result=record['result'];assert result['q']==(10**9 if task[0]=='stb' else task[3])
        records[task[:4]]=result
    cs=[];design=sim.read_design()
    for setting in (4,5,6,7):
        cfg=design['settings'][setting];R=200
        runs=[records[('stb',setting,rep,0)] for rep in range(R)]
        for run in runs:assert [r['n'] for r in run['rows']]==list(sim.GRID)
        for j,n in enumerate(sim.GRID):
            rows=[r['rows'][j] for r in runs]
            lengths=summary([r['log10_length'] for r in rows])
            point=sum(r['pointwise'] for r in rows);scount=sum(r['simultaneous'] for r in rows)
            pl,pu=wilson(point,R);sl,su=wilson(scount,R)
            cs.append(dict(**cfg,n=int(n),target=1-cfg['delta'],pointwise=point/R,point_low=pl,point_high=pu,
                simultaneous=scount/R,simultaneous_low=sl,simultaneous_high=su,
                mean_length=pow10(lengths['log10_mean']),median_length=pow10(lengths['log10_median']),
                q10_length=pow10(lengths['log10_q10']),q90_length=pow10(lengths['log10_q90']),length_mcse=pow10(lengths['log10_mcse']),
                **{k+'_length':v for k,v in lengths.items()},repetitions=R,median_stages=np.median([r['stages'] for r in rows]),
                zeros=sum(np.isneginf(r['log10_error']) for r in rows),float64_underflow=sum(r['length']==0 for r in rows)))
    cubic=[records[t[:4]] for t in expected if t[0]=='astb']
    for r in cubic:assert r['tests']+r['extra']==r['n']
    pd.DataFrame(cs).to_csv(sim.OUT/'coverage_length_summary.csv',index=False)
    pd.DataFrame(cubic).to_csv(sim.OUT/'clt_all_replicates.csv',index=False)
    audit={'complete':True,'coverage_tasks':800,'cubic_astb_tasks':2200,'seed':42,'max_n':10**9,'scope':'Original quadratic coverage and positive-slope cubic distribution only'}
    (sim.OUT/'auxiliary_completion.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=4);p.add_argument('--summarize-only',action='store_true');a=p.parse_args()
    for d in (sim.OUT,sim.TASKS,sim.CHECKPOINTS):d.mkdir(parents=True,exist_ok=True)
    # A process-scoped lock protects the original checkpoint paths on macOS/Linux.
    import fcntl
    with (sim.OUT/'driver.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not a.summarize_only:sim.run_pool(tasks(),a.workers,'auxiliary experiments')
        summarize()

if __name__=='__main__':main()

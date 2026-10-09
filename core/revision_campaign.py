"""Independent setting selection, tuning and formal simulation, with checkpoints."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import json,math,time,pickle,hashlib,socket,sys,multiprocessing as mp
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from functools import lru_cache
ROOT=Path(__file__).resolve().parent
BLOCK=262144
METHODS=['SA small','SA selected','SA large','ASA','PJ','STB','ASTB']

def design():return json.loads((ROOT/'revision_design.json').read_text())
def atomic_pickle(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:pickle.dump(value,f,protocol=5)
    temp.replace(path)
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist())+'\n');temp.replace(path)
def cfgs_for(candidate):
    result=[]
    for j,gamma in enumerate(design()['families']):
        cfg=dict(candidate,setting=3*candidate['candidate']+j,gamma=gamma,v=candidate['sigma'])
        cfg['theta_fraction']=cfg['theta_exact'];result.append(cfg)
    return result

def rng(task,method):
    return np.random.default_rng(np.random.SeedSequence([42,20261009,task['phase'],method,task['cfg']['setting'],task.get('n',0),task['rep'],0]))
def config_values(cfg):
    from revision_kernels import kernel_parameters
    values=kernel_parameters(dict(cfg,theta_fraction=cfg['theta_exact']))
    return values['R'],values['b_ref'],values['rounded_sign'],values['center']

def exact_absolute_errors(estimates,cfg):
    truth=Fraction(cfg['theta_exact']);values=np.asarray(estimates,dtype=float)
    return np.asarray([float(abs(Fraction.from_float(float(x))-truth)) if math.isfinite(x) else abs(x) for x in values.flat]).reshape(values.shape)

def state_load(path,task,initializer):
    fingerprint=task_hash(task)
    if path.exists():
        with path.open('rb') as f:s=pickle.load(f)
        assert s['fingerprint']==fingerprint,'Incompatible checkpoint'
        return s
    return dict(initializer(),fingerprint=fingerprint)
def save(path,s,generator):
    s['rng']=generator.bit_generator.state;atomic_pickle(path,s)

def run_tuning(task,checkpoint):
    from revision_kernels import tuning_sa_block,tuning_pj_block
    cfg=task['cfg'];R,b_ref,rounded,center=config_values(cfg)
    plan=design();cs=np.array(plan['sa_grid']);pc=np.array(plan['pj_c_grid']);ps=np.array(plan['pj_s_grid'][str(cfg['gamma'])])
    def initial():return dict(q=0,sa=np.full(len(cs),center),pj=np.full((len(ps),len(pc)),center),sums=np.zeros((len(ps),len(pc))),comp=np.zeros((len(ps),len(pc))))
    s=state_load(checkpoint,task,initial);g=rng(task,10)
    if 'rng' in s:g.bit_generator.state=s['rng']
    saved=time.monotonic();n=task['n']
    while s['q']<n:
        size=min(BLOCK,n-s['q']);eps=g.normal(0,cfg['sigma'],size);start=s['q']+1
        if cfg['gamma']!=1:
            tuning_sa_block(eps,start,s['sa'],cs,cfg['theta'],cfg['gamma'],cfg['A'],R,rounded,b_ref)
        tuning_pj_block(eps,start,s['pj'],s['sums'],s['comp'],pc,ps,cfg['theta'],cfg['gamma'],cfg['A'],R,rounded,b_ref,cfg['left'],cfg['right'],center)
        s['q']+=size
        if time.monotonic()-saved>=30:save(checkpoint,s,g);saved=time.monotonic()
    save(checkpoint,s,g)
    pj_estimates=center+s['sums']/n
    return dict(q=n,sa_errors=exact_absolute_errors(s['sa'],cfg) if cfg['gamma']!=1 else None,pj_errors=exact_absolute_errors(pj_estimates,cfg),sa_estimates=s['sa'].copy() if cfg['gamma']!=1 else None,pj_estimates=pj_estimates,sa_c_grid=cs,pj_c_grid=pc,pj_s_grid=ps)

def run_comparators(task,checkpoint):
    from revision_kernels import comparator_block
    cfg=task['cfg'];R,b_ref,rounded,center=config_values(cfg)
    def initial():
        st=np.zeros(13);st[:5]=center;return dict(q=0,state=st,rows=[])
    s=state_load(checkpoint,task,initial);g=rng(task,20)
    if 'rng' in s:g.bit_generator.state=s['rng']
    saved=time.monotonic();choice=task['choice'];theta=Fraction(cfg['theta_exact'])
    for budget in task['grid']:
        if s['rows'] and s['rows'][-1]['n']>=budget:continue
        while s['q']<budget:
            size=min(BLOCK,budget-s['q']);eps=g.normal(0,cfg['sigma'],size)
            comparator_block(eps,s['q']+1,s['state'],choice['sa_c'],choice['pj_c'],choice['pj_s'],cfg['theta'],cfg['gamma'],cfg['A'],R,rounded,b_ref,cfg['left'],cfg['right'],center)
            s['q']+=size
            if time.monotonic()-saved>=30:save(checkpoint,s,g);saved=time.monotonic()
        st=s['state'];est=np.r_[st[:4],center+st[9]/budget]
        exact=[abs(Fraction.from_float(float(x))-theta) if math.isfinite(x) else None for x in est]
        errors=[float(x) if x is not None else math.inf for x in exact]
        logs=[(math.log10(x.numerator)-math.log10(x.denominator)) if x else (-math.inf if x is not None else math.inf) for x in exact]
        s['rows'].append(dict(n=budget,q=s['q'],estimates=est.copy(),errors=errors,log10_errors=logs,exact_errors=[str(x) for x in exact],clips=st[10:12].copy()))
        save(checkpoint,s,g)
    return dict(q=s['q'],rows=s['rows'],choice=choice)

def task_id(t):return f"{t['kind']}_{t['cfg']['setting']:03d}_{t.get('n',0):010d}_{t['rep']:04d}"
@lru_cache(maxsize=1)
def execution_fingerprint():
    # Include tuning grids and scientific implementations, not only task labels.
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('revision_design.json','revision_campaign.py','revision_kernels.py','revision_simulation.py')}
    return hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
def task_hash(t):return hashlib.sha256(json.dumps(dict(task=t,execution_fingerprint=execution_fingerprint()),sort_keys=True).encode()).hexdigest()
def execute(item):
    out,task=item;out=Path(out);identifier=task_id(task);result_file=out/'tasks'/(identifier+'.pkl');checkpoint=out/'checkpoints'/(identifier+'.pkl')
    if result_file.exists():
        with result_file.open('rb') as f:old=pickle.load(f)
        assert old['task_hash']==task_hash(task);return identifier,'cached',0.
    started=time.time();kind=task['kind']
    if kind=='tuning':result=run_tuning(task,checkpoint)
    elif kind=='comparators':result=run_comparators(task,checkpoint)
    else:
        import revision_simulation as sim
        if kind=='stb':result=sim.run_stb_task(task['cfg'],task['rep'],checkpoint,task['grid'],phase=task['phase'])
        elif kind=='astb':result=sim.run_astb_task(task['cfg'],task['n'],task['rep'],checkpoint,phase=task['phase'])
        else:raise ValueError(kind)
    target=task.get('n',task.get('grid',[0])[-1])
    if result['q']!=target and result.get('status','complete')=='complete':
        result=dict(result,status='budget_failure',expected_q=target)
    record=dict(task=task,task_hash=task_hash(task),execution_fingerprint=execution_fingerprint(),result=result,host=socket.gethostname(),pid=os.getpid(),started=started,finished=time.time())
    atomic_pickle(result_file,record)
    # The persisted result is the resume record for a completed task. Keep the
    # evolving state only for unfinished or failed tasks (it can contain a block
    # of unused Gaussian draws).
    if result.get('status','complete')=='complete' and result['q']==target:
        with result_file.open('rb') as f:verified=pickle.load(f)
        assert verified['task_hash']==record['task_hash'] and verified['result']['q']==target
        checkpoint.unlink(missing_ok=True)
    return identifier,result.get('status','complete'),time.time()-started

def pool_run(tasks,out,workers):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    contract=out/'execution_fingerprint.json';fingerprint=execution_fingerprint()
    if contract.exists():
        assert json.loads(contract.read_text())['execution_fingerprint']==fingerprint,'Scientific source/design changed; use a new output directory.'
    else:
        assert not list((out/'checkpoints').glob('*.pkl')) and not list((out/'tasks').glob('*.pkl')),'Existing records have no execution fingerprint.'
        write_json(contract,dict(execution_fingerprint=fingerprint))
    identifiers=[task_id(task) for task in tasks]
    assert len(set(identifiers))==len(identifiers),'Duplicate task identities'
    if (out/'expected_tasks.json').exists():
        assert json.loads((out/'expected_tasks.json').read_text())==tasks,'Changed task manifest; use a new output directory.'
    write_json(out/'expected_tasks.json',tasks)
    pending=[]
    for task in tasks:
        f=out/'tasks'/(task_id(task)+'.pkl')
        if f.exists():
            with f.open('rb') as h:r=pickle.load(h)
            assert r['task_hash']==task_hash(task)
        else:pending.append((str(out),task))
    completed=len(tasks)-len(pending);started=time.time()
    write_json(out/'progress.json',dict(expected=len(tasks),complete=completed,pending=len(pending),started=started,workers=workers,status='running'))
    if pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(execute,t) for t in pending]
            for future in as_completed(futures):
                identifier,state,seconds=future.result();completed+=1
                with (out/'task_runtime.jsonl').open('a') as f:f.write(json.dumps(dict(task=identifier,status=state,seconds=seconds,finished=time.time()))+'\n')
                write_json(out/'progress.json',dict(expected=len(tasks),complete=completed,elapsed_seconds=time.time()-started,workers=workers,status='running'))
                if completed%20==0 or completed==len(tasks):print(f'{out.name}: {completed}/{len(tasks)}',flush=True)
    assert all((out/'tasks'/(task_id(t)+'.pkl')).exists() for t in tasks)
    write_json(out/'progress.json',dict(expected=len(tasks),complete=completed,elapsed_seconds=time.time()-started,workers=workers,status='completed'))

def load_results(out):
    out=Path(out);tasks=json.loads((out/'expected_tasks.json').read_text())
    expected={task_id(task):task for task in tasks}
    assert len(expected)==len(tasks),'Duplicate task identities'
    actual={path.stem for path in (out/'tasks').glob('*.pkl')}
    assert actual==set(expected),f'Missing or unexpected task records: {actual.symmetric_difference(expected)}'
    for identifier,task in expected.items():
        with (out/'tasks'/(identifier+'.pkl')).open('rb') as f:record=pickle.load(f)
        assert record['task']==task and record['task_hash']==task_hash(task),'Incompatible task record'
        yield record

def select_steps(out):
    by={}
    for record in load_results(out):
        if record['task']['kind']!='tuning':continue
        setting=record['task']['cfg']['setting'];by.setdefault(setting,[]).append(record)
    choices={}
    for setting,records in by.items():
        r=records[0]['result'];linear=records[0]['task']['cfg']['gamma']==1
        assert sorted(x['task']['rep'] for x in records)==list(range(len(records)))
        for x in records:
            assert x['result']['q']==x['task']['n']
            for key in ('sa_c_grid','pj_c_grid','pj_s_grid'):np.testing.assert_array_equal(x['result'][key],r[key])
        sa_raw=np.asarray([x['result']['sa_errors'] for x in records]) if not linear else None
        pj_raw=np.asarray([x['result']['pj_errors'] for x in records])
        # Any failed replicate disqualifies its candidate; raw errors remain intact.
        sa=np.mean(np.where(np.isfinite(sa_raw),sa_raw,np.inf),axis=0) if not linear else None
        pj=np.mean(np.where(np.isfinite(pj_raw),pj_raw,np.inf),axis=0)
        if not np.isfinite(pj).any() or (not linear and not np.isfinite(sa).any()):
            write_json(Path(out)/'tuning_selection_failure.json',dict(setting=setting,sa_tuning_mean=sa,pj_tuning_mean=pj))
            raise ArithmeticError('No finite tuning candidate remains; all raw failures are retained.')
        i,j=np.unravel_index(np.argmin(pj),pj.shape)
        choices[str(setting)]=dict(sa_c=1. if linear else float(r['sa_c_grid'][np.argmin(sa)]),pj_c=float(r['pj_c_grid'][j]),pj_s=float(r['pj_s_grid'][i]),sa_tuning_mean=sa,pj_tuning_mean=pj,repetitions=len(records),sa_nonfinite_counts=None if linear else np.sum(~np.isfinite(sa_raw),axis=0),pj_nonfinite_counts=np.sum(~np.isfinite(pj_raw),axis=0),execution_fingerprint=execution_fingerprint())
    write_json(Path(out)/'selected_steps.json',choices);return choices

def build_tuning(configs,n,reps,phase):return [dict(kind='tuning',cfg=cfg,n=n,rep=r,phase=phase) for cfg in configs for r in range(reps)]
def build_evaluation(configs,grid,reps,phase,choices):
    tasks=[]
    for cfg in configs:
        for rep in range(reps):
            for kind in ('comparators','stb'):
                task=dict(kind=kind,cfg=cfg,grid=grid,rep=rep,phase=phase)
                if kind=='comparators':task['choice']={key:choices[str(cfg['setting'])][key] for key in ('sa_c','pj_c','pj_s')}
                tasks.append(task)
            tasks += [dict(kind='astb',cfg=cfg,n=n,rep=rep,phase=phase) for n in grid]
    return tasks

def screen(workers=4):
    p=design();configs=[cfg for c in p['candidate_nuisance_settings'] for cfg in cfgs_for(c)];d=p['screen']
    tune=ROOT/'screen_tuning';out=ROOT/'screen_results'
    pool_run(build_tuning(configs,d['tuning_n'],d['tuning_repetitions'],d['tuning_phase']),tune,workers);choices=select_steps(tune)
    pool_run(build_evaluation(configs,d['evaluation_grid'],d['evaluation_repetitions'],d['evaluation_phase'],choices),out,workers)
    # Preserve every replicate; selection uses mean absolute errors on a log scale.
    errors={}
    for rec in load_results(out):
        task,r=rec['task'],rec['result'];setting=task['cfg']['setting'];kind=task['kind']
        if kind=='comparators':
            for row in r['rows']:
                for method,logerr in zip(METHODS[:5],row['log10_errors']):errors.setdefault((setting,row['n'],method),[]).append(logerr)
        elif kind=='stb':
            for row in r['rows']:errors.setdefault((setting,row['n'],'STB'),[]).append(row['log10_error'])
        else:errors.setdefault((setting,task['n'],'ASTB'),[]).append(r['log10_error'] if r.get('log10_error') is not None else math.inf)
    def logmean(values):
        a=np.array(values)
        if np.isnan(a).any() or np.isposinf(a).any():return math.inf
        m=np.max(a)
        if math.isinf(m):return float(m)
        return float(m+np.log10(np.mean(10**(a-m))))
    mean={key:logmean(v) for key,v in errors.items()}
    rows=[dict(setting=k[0],n=k[1],method=k[2],log10_mean=val,repetitions=len(errors[k])) for k,val in mean.items()]
    write_json(out/'all_screening_means.json',rows)
    scores=[]
    for c in p['candidate_nuisance_settings']:
        terms=[]
        for cfg in cfgs_for(c):
            for n in d['scoring_n']:
                best=min(mean[cfg['setting'],n,m] for m in ('SA selected','ASA','PJ'))
                for method in ('STB','ASTB'):
                    value=mean[cfg['setting'],n,method]
                    ratio=(math.inf if value==math.inf or best==math.inf else 0. if value==best==-math.inf else value-best)
                    terms.append(dict(gamma=cfg['gamma'],n=n,method=method,log10_ratio=ratio,penalty=max(0.,ratio)))
        scores.append(dict(candidate=c['candidate'],score=sum(x['penalty'] for x in terms),terms=terms,configuration=c))
    winner=min(scores,key=lambda x:(x['score'],x['candidate']))
    if not math.isfinite(winner['score']):
        write_json(ROOT/'setting_selection_failure.json',dict(all_candidates=scores))
        raise ArithmeticError('Every screening candidate has a numerical failure.')
    write_json(ROOT/'setting_selection.json',dict(rule=d['score'],all_candidates=scores,selected=winner,selected_utc=time.time(),execution_fingerprint=execution_fingerprint()))
    print('Frozen selected candidate:',winner['candidate'],winner['configuration'],flush=True)
    return winner

def formal(workers=4):
    p=design();selection=json.loads((ROOT/'setting_selection.json').read_text())
    assert selection['execution_fingerprint']==execution_fingerprint(),'Screening selection belongs to different source/design.'
    selected=selection['selected']['configuration'];configs=cfgs_for(selected);d=p['formal']
    tune=ROOT/'formal_tuning';out=ROOT/'formal_results'
    pool_run(build_tuning(configs,d['tuning_n'],d['tuning_repetitions'],d['tuning_phase']),tune,workers);choices=select_steps(tune)
    tasks=build_evaluation(configs,p['grid'],d['rate_repetitions'],d['evaluation_phase'],choices)
    linear=next(c for c in configs if c['gamma']==1.)
    for n,reps in d['linear_clt_repetitions'].items():
        tasks += [dict(kind='astb',cfg=linear,n=int(n),rep=r,phase=d['evaluation_phase']) for r in range(d['rate_repetitions'],reps)]
    pool_run(tasks,out,workers)
    failures=[task_id(record['task']) for record in load_results(out) if record['result'].get('status','complete')!='complete']
    write_json(ROOT/'formal_completion.json',dict(complete=not failures,failures=failures,tasks=len(tasks),tuning_tasks=len(configs)*d['tuning_repetitions'],selected_configuration=selected,finished=time.time(),execution_fingerprint=execution_fingerprint()))
    if failures:raise ArithmeticError('Formal records include failures; all are retained for audit.')
    return len(tasks)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['screen','formal']);parser.add_argument('--workers',type=int,default=4);a=parser.parse_args()
    (screen if a.phase=='screen' else formal)(a.workers)

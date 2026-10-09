"""Local simulation runner with unchanged task streams and scientific updates.
Explicitly authorized local run, isolated from the GreatLakes results.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'
import json, math, time, pickle, socket, sys, platform, subprocess, hashlib
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import pandas as pd
from hpc_kernels import tuning_block, comparator_block, scan_test

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'data_local'
TASKS = OUT/'tasks'
CHECKPOINTS = OUT/'checkpoints'
MAX_N = 10**9
GRID = 10**np.arange(2,10,dtype=np.int64)
CLT_GRID = 10**np.arange(3,10,dtype=np.int64)
SEED = 42
A_VALUES = np.array([.25,.5,1.,2.,4.,8.,16.,32.])
RATE_REPS = 200
TUNING_REPS = 60
BLOCK = 262144
CHECKPOINT_SECONDS = 30
if hasattr(sys,'set_int_max_str_digits'): sys.set_int_max_str_digits(0)

def read_design():
    return json.loads((ROOT/'experimental_design_hpc.json').read_text())

def rng_for(*keys):
    return np.random.default_rng(np.random.SeedSequence([SEED,*map(int,keys)]))

def task_id(kind,setting,rep,n=0):
    return f'{kind}_{setting:02d}_{n:010d}_{rep:04d}'

def atomic_pickle(path, value):
    temporary=path.with_suffix('.tmp')
    with open(temporary,'wb') as handle:
        pickle.dump(value,handle,protocol=5)
    os.replace(temporary,path)

def load_state(path, initializer):
    if path.exists():
        with open(path,'rb') as handle: return pickle.load(handle)
    return initializer()

def save_state(path,state,gen):
    state['rng']=gen.bit_generator.state
    atomic_pickle(path,state)

def restore_generator(state,*keys):
    gen=rng_for(*keys)
    if 'rng' in state: gen.bit_generator.state=state['rng']
    return gen

def log_fraction(value):
    if value == 0: return -math.inf
    # math.log handles Python integers without first converting them to float.
    return (math.log(abs(value.numerator))-math.log(value.denominator))/math.log(10.)

def float_fraction(value):
    try: return float(value)
    except OverflowError: return math.copysign(math.inf,value.numerator)

def rounded_sign(cfg):
    e=Fraction.from_float(cfg['theta'])-Fraction(cfg['theta_fraction'])
    return 1. if e>0 else -1. if e<0 else 0.

def run_tuning_task(cfg,rep,checkpoint,max_n=MAX_N):
    state=load_state(checkpoint,lambda:dict(q=0,x=np.full(len(A_VALUES),.5)))
    gen=restore_generator(state,10,cfg['setting'],rep)
    saved=time.monotonic()
    while state['q']<max_n:
        size=min(BLOCK,max_n-state['q'])
        eps=gen.normal(0,cfg['sigma'],size)
        tuning_block(eps,state['q']+1,state['x'],A_VALUES,cfg['theta'],cfg['gamma'],rounded_sign(cfg))
        state['q']+=size
        if time.monotonic()-saved>=CHECKPOINT_SECONDS:
            save_state(checkpoint,state,gen);saved=time.monotonic()
    return dict(q=state['q'],errors=np.abs(state['x']-cfg['theta']),x=state['x'])

def run_comparator_task(cfg,rep,a,checkpoint,grid=GRID):
    def initial():
        x=np.zeros(12);x[:5]=.5
        return dict(q=0,state=x,rows=[])
    state=load_state(checkpoint,initial)
    gen=restore_generator(state,20,cfg['setting'],rep)
    saved=time.monotonic()
    for budget in grid:
        if state['rows'] and state['rows'][-1]['n']>=budget: continue
        while state['q']<budget:
            size=min(BLOCK,int(budget)-state['q'])
            eps=gen.normal(0,cfg['sigma'],size)
            comparator_block(eps,state['q']+1,state['state'],a,cfg['theta'],cfg['gamma'],rounded_sign(cfg))
            state['q']+=size
            if time.monotonic()-saved>=CHECKPOINT_SECONDS:
                save_state(checkpoint,state,gen);saved=time.monotonic()
        s=state['state'];est=np.r_[s[:4],s[9]/budget]
        state['rows'].append(dict(n=int(budget),estimates=est.copy(),errors=np.abs(est-cfg['theta']),clips=s[10:12].copy()))
        save_state(checkpoint,state,gen)
    return dict(rows=state['rows'],a=a,q=state['q'])

def run_stb_task(cfg,rep,checkpoint,grid=GRID):
    theta=Fraction(cfg['theta_fraction'])
    K=math.ceil(math.log2(4/cfg['delta']))
    state=load_state(checkpoint,lambda:dict(q=0,l=Fraction(0),r=Fraction(1),x=Fraction(1,2),t=1,j=0,S=0.,best_count=0,best_x=Fraction(1,2),covered=True,rows=[]))
    gen=restore_generator(state,30,cfg['setting'],rep)
    saved=time.monotonic()
    for budget in grid:
        if state['rows'] and state['rows'][-1]['n']>=budget: continue
        while state['q']<budget:
            size=min(4096,int(budget)-state['q'])
            e=state['x']-theta
            # Comparing rational numbers avoids an underflowed sign at deep stages.
            sign=1. if e>0 else -1. if e<0 else 0.
            f=cfg['signal']*sign
            if cfg['gamma']>0: f*=float(abs(e))**cfg['gamma']
            eps=gen.normal(0,cfg['sigma'],size)
            used,total,crossed=scan_test(eps,f,state['S'],state['j'],state['t'],K,cfg['v'])
            state['j']+=used;state['q']+=used;state['S']=total
            if crossed:
                if state['j']>state['best_count']:
                    state['best_count'],state['best_x']=state['j'],state['x']
                state['r' if total>0 else 'l']=state['x']
                state['covered'] &= state['l']<=theta<=state['r']
                state['t']+=1;state['j']=0;state['S']=0.
                state['x']=(state['l']+state['r'])/2
            if time.monotonic()-saved>=CHECKPOINT_SECONDS:
                save_state(checkpoint,state,gen);saved=time.monotonic()
        winner=state['x'] if state['j']>state['best_count'] else state['best_x']
        estimate=max(state['l'],min(state['r'],winner))
        error=abs(estimate-theta);length=state['r']-state['l']
        state['rows'].append(dict(n=int(budget),estimate=float(estimate),error=float_fraction(error),length=float_fraction(length),
            log10_error=log_fraction(error),log10_length=log_fraction(length),
            exact_error=str(error),exact_length=str(length),exact_estimate=str(estimate),
            pointwise=int(state['l']<=theta<=state['r']),simultaneous=int(state['covered']),
            stages=state['t']-1,unfinished=state['j'],q=state['q']))
        save_state(checkpoint,state,gen)
    return dict(q=state['q'],rows=state['rows'])

def merge_fit(state,x,fx,eps):
    count=len(eps); y=fx+eps; mean=float(y.mean());q=state['q'];total=q+count
    dx,dy=x-state['mx'],mean-state['my']
    state['D']+=q*count/total*dx*dx
    state['Cxy']+=q*count/total*dx*dy
    state['Cyy']+=float(np.sum((y-mean)**2))+q*count/total*dy*dy
    state['mx']+=count/total*dx;state['my']+=count/total*dy
    state['noise_sum']+=float(eps.sum());state['q']=total


def run_astb_task(n,nonlinear,rep,checkpoint):
    theta=1/3;sigma=.5
    K=max(2,math.ceil(math.log2(4/.05)))+math.ceil(math.log(math.log(n+math.exp(math.e))))
    state=load_state(checkpoint,lambda:dict(q=0,t=1,l=0.,r=1.,fl=math.nan,fr=math.nan,
        phase='position',j=0,S=0.,mx=0.,my=0.,D=0.,Cxy=0.,Cyy=0.,noise_sum=0.,tests=0,extra=0,covered=True,last_phase=0))
    gen=restore_generator(state,40,int(nonlinear),n,rep);saved=time.monotonic()
    while state['q']<n:
        if state['phase']=='position':
            w=.5
            if math.isfinite(state['fl']) and math.isfinite(state['fr']) and state['fl']<0<state['fr']:
                w=-state['fl']/(state['fr']-state['fl'])
            w=min(5/8,max(3/8,w))+gen.uniform(-1/8,1/8)
            state['x']=(1-w)*state['l']+w*state['r']
            e=state['x']-theta
            state['fx']=e+2*e**3 if nonlinear else e
            state['j']=0;state['S']=0.;state['phase']='test'
        if state['phase']=='test':
            size=min(16384,n-state['q'])
            eps=gen.normal(0,sigma,size)
            used,total,crossed=scan_test(eps,state['fx'],state['S'],state['j'],state['t'],K,sigma)
            merge_fit(state,state['x'],state['fx'],eps[:used])
            state['j']+=used;state['tests']+=used;state['S']=total;state['last_phase']=0
            if crossed:
                state['positive']=total>0
                state['r' if state['positive'] else 'l']=state['x']
                state['covered'] &= state['l']<=theta<=state['r']
                state['extra_target']=min(state['j'],n-state['q'])
                state['extra_done']=0;state['phase']='extra'
        if state['phase']=='extra':
            # This batch is stopped only by its preselected length or total budget.
            remaining=state['extra_target']-state['extra_done']
            if remaining:
                count=min(BLOCK,remaining)
                eps=gen.normal(0,sigma,count)
                merge_fit(state,state['x'],state['fx'],eps)
                state['S']+=float(np.sum(state['fx']+eps))
                state['extra']+=count;state['extra_done']+=count
                state['last_phase']=1 if state['extra_done']==state['j'] else 2
            if state['extra_done']==state['extra_target']:
                state['fr' if state['positive'] else 'fl']=state['S']/(state['j']+state['extra_done'])
                state['t']+=1;state['phase']='position'
        if time.monotonic()-saved>=CHECKPOINT_SECONDS:
            save_state(checkpoint,state,gen);saved=time.monotonic()
    slope=state['Cxy']/state['D'] if state['D']>0 else 0.
    fallback=state['D']<=0 or slope<=0 or not math.isfinite(slope)
    if fallback:
        estimate=.5;noise_var=variance=0.
    else:
        estimate=state['mx']-state['my']/slope
        noise_var=max(0.,(state['Cyy']-state['Cxy']**2/state['D'])/n)
        variance=noise_var/slope**2
    assert state['q']==n==state['tests']+state['extra']
    z=math.sqrt(n)*(estimate-theta)
    return dict(n=n,theta=theta,sigma=sigma,beta=1.,estimate=estimate,normalized=z,standardized=z/sigma,
        slope=slope,D=state['D'],variance_estimate=variance,noise_variance=noise_var,
        tests=state['tests'],extra=state['extra'],stages=state['t']-1,K=K,
        fallback=int(fallback),covered=int(state['covered']),last_phase=state['last_phase'],
        noise_sum=state['noise_sum'],linear_remainder=z+state['noise_sum']/math.sqrt(n),
        model='cubic' if nonlinear else 'linear',replicate=rep,q=n)


def execute(task):
    kind,setting,rep,n,a=task
    identifier=task_id(kind,setting,rep,n)
    result_path=TASKS/(identifier+'.pkl');cp=CHECKPOINTS/(identifier+'.pkl')
    if result_path.exists(): return identifier,'cached',0.
    started=time.time();cfg=read_design()['settings'][setting] if kind!='astb' else None
    if kind=='tuning': result=run_tuning_task(cfg,rep,cp)
    elif kind=='comparators': result=run_comparator_task(cfg,rep,a,cp)
    elif kind=='stb': result=run_stb_task(cfg,rep,cp)
    elif kind=='astb': result=run_astb_task(n,bool(setting),rep,cp)
    else: raise ValueError(kind)
    record=dict(task=task,result=result,host=socket.gethostname(),pid=os.getpid(),started=started,finished=time.time())
    atomic_pickle(result_path,record)
    return identifier,'complete',time.time()-started


def environment():
    import numba,scipy,psutil
    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else []
    job=os.getenv('SLURM_JOB_ID')
    details=subprocess.run(['scontrol','show','job',job],capture_output=True,text=True).stdout if job else ''
    return dict(logical_cpus=os.cpu_count(),memory_total_bytes=psutil.virtual_memory().total,
        machine=platform.machine(),host=socket.gethostname(),python=sys.version,executable=sys.executable,
        platform=platform.platform(),numpy=np.__version__,numba=numba.__version__,pandas=pd.__version__,scipy=scipy.__version__,
        affinity=affinity,slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},scontrol=details,
        threads={k:os.getenv(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')},seed=SEED)


def tuning_choices():
    choices={};rows=[]
    for cfg in read_design()['settings']:
        setting=cfg['setting']
        if cfg['gamma']==1:
            choices[setting]=1.
            for a in A_VALUES:rows.append(dict(setting=setting,a=a,mae=math.nan,mcse=math.nan,selected=int(a==1.),n=MAX_N,repetitions=0,source='oracle 1/beta'))
        else:
            values=[]
            for rep in range(TUNING_REPS):
                with open(TASKS/(task_id('tuning',setting,rep)+'.pkl'),'rb') as handle:
                    values.append(pickle.load(handle)['result']['errors'])
            errors=np.array(values);mean=errors.mean(axis=0)
            choices[setting]=float(A_VALUES[np.argmin(mean)])
            for j,a in enumerate(A_VALUES):rows.append(dict(setting=setting,a=a,mae=mean[j],mcse=errors[:,j].std(ddof=1)/math.sqrt(TUNING_REPS),selected=int(a==choices[setting]),n=MAX_N,repetitions=TUNING_REPS,source='independent tuning grid'))
    pd.DataFrame(rows).to_csv(OUT/'tuning_summary.csv',index=False)
    (OUT/'selected_stepsizes.json').write_text(json.dumps(choices,indent=2))
    return choices


def run_pool(tasks,workers,phase):
    pending=[t for t in tasks if not (TASKS/(task_id(t[0],t[1],t[2],t[3])+'.pkl')).exists()]
    print(f'{phase}: {len(tasks)-len(pending)}/{len(tasks)} cached, {len(pending)} pending',flush=True)
    if not pending:return
    # Spawn avoids inheriting the notebook's live kernel state in worker processes.
    started=time.time();complete=0
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(execute,t):t for t in pending}
        for future in as_completed(futures):
            identifier,status,seconds=future.result();complete+=1
            with open(OUT/'task_log.jsonl','a') as handle:
                handle.write(json.dumps(dict(task=identifier,status=status,seconds=seconds,time=time.time(),phase=phase))+'\n')
            if complete%10==0 or complete==len(pending):
                print(f'{phase}: {complete}/{len(pending)} newly complete, {(time.time()-started)/60:.1f} min',flush=True)


def run_all(workers=None):
    # A second notebook must not write the same task checkpoints concurrently.
    import fcntl
    OUT.mkdir(exist_ok=True)
    with open(OUT/'driver.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run_all(workers)


def _run_all(workers=None):
    OUT.mkdir(exist_ok=True);TASKS.mkdir(exist_ok=True);CHECKPOINTS.mkdir(exist_ok=True)
    env=environment()
    if platform.system()!='Darwin' or env['slurm'].get('SLURM_JOB_ID'):
        raise RuntimeError('This entry point is for the explicitly authorized local Mac run.')
    env['execution_target']='local'
    available=os.cpu_count() or 1
    workers=min(workers or available,available)
    env['workers']=workers
    (OUT/'environment_local.json').write_text(json.dumps(env,indent=2))
    started=time.time()
    configs=read_design()['settings']
    tuning=[('tuning',c['setting'],r,0,0.) for c in configs if c['gamma']!=1 for r in range(TUNING_REPS)]
    # These independent experiments run while the step-size selection is pending.
    independent=[('stb',c['setting'],r,0,0.) for c in configs for r in range(RATE_REPS)]
    independent += [('astb',m,r,int(n),0.) for n in CLT_GRID for m in (0,1) for r in range(200 if n>=10**7 else 400)]
    run_pool(tuning,workers,'independent tuning')
    choices=tuning_choices()
    comparators=[('comparators',c['setting'],r,0,choices[c['setting']]) for c in configs for r in range(RATE_REPS)]
    # Submit larger comparator tasks first, with small-budget ASTB mixed at the end.
    run_pool(comparators+independent,workers,'formal experiments')
    expected=tuning+comparators+independent
    missing=[task_id(t[0],t[1],t[2],t[3]) for t in expected if not (TASKS/(task_id(t[0],t[1],t[2],t[3])+'.pkl')).exists()]
    audit=dict(expected_tasks=len(expected),completed_tasks=len(expected)-len(missing),missing=missing,
        elapsed_seconds=time.time()-started,max_n=MAX_N,workers=workers,finished=time.time())
    (OUT/'completion_audit.json').write_text(json.dumps(audit,indent=2))
    assert not missing,missing
    print(audit,flush=True)
    return audit

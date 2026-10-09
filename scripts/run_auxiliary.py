"""Run only the original quadratic coverage and positive-slope cubic QQ tasks."""
from pathlib import Path
import argparse, os, shutil, socket, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',type=Path,default=ROOT/'results/auxiliary')
    p.add_argument('--summarize-only',action='store_true')
    a=p.parse_args()
    if a.workers<1:p.error('workers must be positive')
    if 'login' in socket.gethostname().lower():p.error('Use a workstation or allocated compute node.')
    out=a.output.resolve()
    if out in (ROOT,ROOT/'legacy',ROOT/'core',ROOT/'scripts'):p.error('Choose a separate output directory.')
    out.mkdir(parents=True,exist_ok=True)
    sources=[*(ROOT/'legacy'/n for n in ('hpc_simulation.py','hpc_kernels.py','experimental_design_hpc.json')),*(ROOT/'scripts'/n for n in ('auxiliary_driver.py','auxiliary_summary.py'))]
    for src in sources:
        dst=out/src.name
        if dst.exists() and dst.read_bytes()!=src.read_bytes():p.error('Source differs; choose a new output directory: '+str(dst))
        shutil.copy2(src,dst)
    env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS'):env[k]='1'
    command=[sys.executable,str(out/'auxiliary_driver.py'),'--workers',str(a.workers)]
    if a.summarize_only:command.append('--summarize-only')
    subprocess.run(command,cwd=out,env=env,check=True)

if __name__=='__main__':main()

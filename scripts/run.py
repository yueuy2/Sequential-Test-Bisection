"""Explicit opt-in rerun in a separate directory; archived results stay unchanged."""
from pathlib import Path
import argparse,os,shutil,subprocess,sys,socket
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['screen','formal']);p.add_argument('--workers',type=int,default=4);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if 'login' in socket.gethostname().lower():p.error('Use a local workstation or an allocated compute node, not a login node.')
 if a.workers<1:p.error('workers must be positive')
 out=a.output.resolve()
 if out==ROOT/'core' or out==ROOT:p.error('Choose a separate output directory.')
 out.mkdir(parents=True,exist_ok=True)
 for name in ['revision_campaign.py','revision_simulation.py','revision_kernels.py','revision_design.json']:
  src=ROOT/'core'/name;dst=out/name
  if dst.exists() and dst.read_bytes()!=src.read_bytes():p.error(f'Existing source differs: {dst}')
  shutil.copy2(src,dst)
 if a.phase=='formal' and not (out/'setting_selection.json').exists():shutil.copy2(ROOT/'core/setting_selection.json',out/'setting_selection.json')
 env=os.environ.copy()
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS']:env[key]='1'
 subprocess.run([sys.executable,str(out/'revision_campaign.py'),a.phase,'--workers',str(a.workers)],cwd=out,env=env,check=True)
if __name__=='__main__':main()

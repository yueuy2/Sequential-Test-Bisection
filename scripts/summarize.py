"""Audit full repetitions, merge raw results and export current numerical tables."""
from pathlib import Path
import argparse, subprocess, sys
import pandas as pd
import plot_helpers as display
ROOT=Path(__file__).resolve().parents[1]

def tables(data,out):
    out.mkdir(parents=True,exist_ok=True)
    errors=pd.read_csv(data/'error_summary.csv')
    for number,gamma in ((2,1),(3,3),(4,0)):
        x=errors[(errors.gamma==gamma)&errors.n.isin([10**8,10**9])].copy()
        x['method']=x.method.map(display.method_label)
        x[['method','n','repetitions','log10_mean','log10_mcse']].to_csv(out/f'table_{number}.csv',index=False)
        (out/f'table_{number}.tex').write_text(display.error_table(errors,gamma)+'\n')
    coverage=pd.read_csv(data/'coverage_length_summary.csv');quadratic=pd.read_csv(data/'legacy_gamma2_coverage.csv')
    for number,gamma in ((5,1),(6,2),(7,3),(8,0)):
        x=quadratic if gamma==2 else coverage[coverage.gamma==gamma]
        columns=['n','pointwise','simultaneous','log10_median_length','repetitions']
        if gamma!=1:columns=['theta','sigma']+columns
        x[x.n.isin([10**8,10**9])][columns].to_csv(out/f'table_{number}.csv',index=False)
    pd.read_csv(data/'clt_summary.csv')[['n','mean','variance','variance_ratio','repetitions']].to_csv(out/'table_9.csv',index=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'results/formal')
    p.add_argument('--auxiliary',type=Path,default=ROOT/'results/auxiliary/data_local')
    p.add_argument('--output',type=Path,default=ROOT/'results/data')
    a=p.parse_args();out=a.output.resolve()
    subprocess.run([sys.executable,str(ROOT/'core/revision_merge.py'),'--run-directory',str(a.run.resolve()),'--legacy-data',str(a.auxiliary.resolve()),'--data-directory',str(out)],check=True)
    tables(out,out.parent/'tables')

if __name__=='__main__':main()

"""Reproduce the integrated views from all audited, already-computed repetitions."""
from pathlib import Path
import sys,os,json,math
ROOT=Path(__file__).resolve().parents[1]
import argparse
import pandas as pd
import plot_helpers as rr
import numpy as np
from scipy import stats
BUDGETS=tuple(10**k for k in range(4,10))

def read_bundle(data):
 import hashlib
 audit=json.loads((data/'formal_completion_audit.json').read_text())
 if audit['status']!='complete':raise ValueError('A complete audited campaign is required.')
 for name,expected in audit['output_sha256'].items():
  if hashlib.sha256((data/name).read_bytes()).hexdigest()!=expected:raise ValueError('Data hash mismatch: '+name)
 files={'error':'error_summary.csv','cs':'coverage_length_summary.csv','raw':'astb_all_replicates.csv','legacy_cs':'legacy_gamma2_coverage.csv','legacy_qq':'legacy_cubic_qq_replicates.csv'}
 b={key:pd.read_csv(data/name) for key,name in files.items()}
 rr.validate_error_summary(b['error'])
 b['linear_raw']=b['raw'][b['raw'].gamma==1].copy()
 b['cfg']={'delta':.05}
 return b

def build(data, out):
 data=Path(data).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
 bundle=read_bundle(data)
 # TeX binaries must be discoverable on PATH; no machine-specific path is required.
 os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/matplotlib'))
 plt=rr.configure_publication_style()
 plt.rcParams.update({'font.size':14,'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,'axes.titlesize':14})
 rr.make_figures(bundle,out)
 fig,axes=plt.subplots(3,1,figsize=(7.2,8.0))
 for ax,gamma,title in zip(axes,(1,3,0),('Linear signal','Flat cubic signal','Crossing jump')):
  f=bundle['error'][bundle['error'].gamma==gamma]
  if gamma==3:f=f[f.n>=1000]
  for method,color,line in zip(rr.METHODS,rr.COLORS,rr.LINE_STYLES):
   a=f[f.method==method].sort_values('n');x=np.log10(a.n.to_numpy());y=a.log10_mean.to_numpy();valid=np.isfinite(y)
   ax.plot(x,y,label=rr.method_label(method),color=color,ls=line,marker='o',ms=3.5,lw=1.4)
   with np.errstate(invalid='ignore',divide='ignore'):
    rel=10.**(a.log10_mcse.to_numpy()[valid]-y[valid]);lo=y[valid]+np.log10(np.where(1-1.96*rel>0,1-1.96*rel,np.nan));hi=y[valid]+np.log10(1+1.96*rel)
   ax.fill_between(x[valid],lo,hi,color=color,alpha=.12)
  rr.logarithmic_axes(ax,'Mean absolute error');ax.set_title(f'({"abc"[(1,3,0).index(gamma)]}) {title}',loc='left')
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.55,1.015),frameon=False)
 fig.tight_layout(rect=(0,0,1,.94));rr.save_figure(plt,fig,out,'sim_error_combined')
 audit={'display_budgets':list(BUDGETS),'stream_reuse':'No new simulation; complete samples used','counts':{},'combined_families':[1,3,0],'root_display':'0.37','coverage_bands':'No Monte Carlo interval bands; length percentiles unchanged','raw_label_mapping':{'SA selected':'SA optimal'}}
 for key,stem,kinds in [('linear_raw','linear',('hist','qq')),('legacy_qq','cubic_reference',('qq',))]:
  raw=bundle[key]
  for kind in kinds:
   fig,axes=plt.subplots(6,1,figsize=(7.2,10.8))
   counts={}
   for i,(ax,n) in enumerate(zip(axes,BUDGETS)):
    a=raw[raw.n==n];count=len(a);expected=400 if n<=10**6 else 200
    assert count==expected and set(a.replicate)==set(range(count))
    assert np.isfinite(a[['normalized','standardized']].to_numpy()).all()
    if kind=='qq':
     theory,empirical,outside=rr.full_sample_qq(a.standardized)
     ax.scatter(theory,empirical,s=9,color=rr.COLORS[0],clip_on=True,label='All-sample quantiles')
     ax.plot(rr.QQ_WINDOW,rr.QQ_WINDOW,color=rr.COLORS[2],lw=1.2,label='Standard normal quantiles')
     ax.set(xlim=rr.QQ_WINDOW,ylim=rr.QQ_WINDOW,ylabel='Standardized\nerror quantile')
     xlabel='Standard normal quantile'
    else:
     scale=float(a.sigma.iloc[0]/a.beta.iloc[0]);bins=np.linspace(-5*scale,5*scale,41);h,_=np.histogram(a.normalized,bins)
     outside=count-int(h.sum());ax.bar((bins[1:]+bins[:-1])/2,h/(count*np.diff(bins)),width=np.diff(bins),color=rr.COLORS[0],alpha=.65,label='Empirical density')
     xx=np.linspace(bins[0],bins[-1],400);ax.plot(xx,stats.norm.pdf(xx,scale=scale),color=rr.COLORS[2],lw=1.3,label='Theoretical normal density')
     ax.set(xlim=(bins[0],bins[-1]),ylabel='Density');xlabel=r'$\sqrt{n}(\widehat\theta_n-\theta)$'
    ax.text(.02,.82,rf'({chr(97+i)}) $n=10^{{{int(math.log10(n))}}}$, $R={count}$',transform=ax.transAxes,fontsize=13,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
    counts[n]={'repetitions':count,'outside_display':int(outside)}
   axes[-1].set_xlabel(xlabel)
   handles,labels=axes[-1].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc='upper center',bbox_to_anchor=(.55,1.01),frameon=False)
   fig.tight_layout(rect=(0,0,1,.975),h_pad=.45)
   rr.save_figure(plt,fig,out,f'sim_{stem}_{kind}_vertical');audit['counts'][f'{stem}_{kind}']=counts
 (out/'plot_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps(audit,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data',type=Path,default=ROOT/'results/data')
 p.add_argument('--output',type=Path,default=ROOT/'results/figures')
 a=p.parse_args();build(a.data,a.output)


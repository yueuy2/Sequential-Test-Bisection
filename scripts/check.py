"""Validate source fingerprints and run tiny deterministic installation checks."""
from pathlib import Path
import hashlib,json,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'))
import revision_campaign as campaign
import revision_simulation as sim
import numpy as np

def main():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    selection=json.loads((ROOT/'core/setting_selection.json').read_text())
    assert campaign.execution_fingerprint()==selection['execution_fingerprint']
    d=campaign.design();assert d['seed']==42 and d['max_n']==10**9 and d['formal']['rate_repetitions']==200
    configs=campaign.cfgs_for(selection['selected']['configuration'])
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);count=0
        for cfg in configs:
            # Only 100 observations; these are installation checks, never reported results.
            tuning=dict(kind='tuning',cfg=cfg,n=100,rep=0,phase=200)
            a=campaign.run_tuning(tuning,tmp/f'tune-{count}.pkl');assert a['q']==100
            choice={'sa_c':1.,'pj_c':1.,'pj_s':1.}
            task=dict(kind='comparators',cfg=cfg,grid=[100],rep=0,phase=210,choice=choice)
            a=campaign.run_comparators(task,tmp/f'comp-{count}.pkl');assert a['q']==100 and len(a['rows'][0]['estimates'])==5
            a=sim.run_stb_task(cfg,0,None,[100],phase=210);assert a['q']==100
            a=sim.run_astb_task(cfg,100,0,None,phase=210);b=sim.run_astb_task(cfg,100,0,None,phase=210)
            assert a['q']==100 and a['tests']+a['extra']==100 and a['estimate']==b['estimate']
            count+=1
    print('PASS: package hashes, original scientific fingerprint, fixed design, all three families, all methods, and deterministic ASTB streams. No formal campaign launched.')

if __name__=='__main__':main()

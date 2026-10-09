"""Original auxiliary summary formulas; no simulation or file I/O on import."""

import math
import numpy as np
from scipy import stats

def pow10(value):
    # Zero here denotes IEEE underflow, identified separately in each output row.
    if value == -math.inf:return 0.
    return 10.**value if value > -324 else 0.

def summary(log_values):
    z=np.asarray(log_values,dtype=float)
    if np.isposinf(z).any() or np.isnan(z).any():raise ArithmeticError('Nonfinite error requires explicit review.')
    if np.all(np.isneginf(z)):
        return {k: -math.inf for k in ('log10_mean','log10_median','log10_q10','log10_q90','log10_mcse')}
    largest=z.max();scaled=10.**(z-largest)
    result={}
    for name,value in [('mean',scaled.mean()),('median',np.median(scaled)),('q10',np.quantile(scaled,.1)),('q90',np.quantile(scaled,.9)),('mcse',scaled.std(ddof=1)/np.sqrt(len(z)))]:
        result['log10_'+name]=largest+math.log10(value) if value>0 else -math.inf
    ordered=np.sort(z)
    for name,prob in [('median',.5),('q10',.1),('q90',.9)]:
        position=(len(z)-1)*prob;left=int(math.floor(position));weight=position-left
        if weight==0: result['log10_'+name]=ordered[left]
        else: result['log10_'+name]=float(np.logaddexp(ordered[left]*math.log(10)+math.log1p(-weight),ordered[left+1]*math.log(10)+math.log(weight))/math.log(10))
    return result

def wilson(count,n):
    z=stats.norm.ppf(.975);p=count/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-radius,center+radius

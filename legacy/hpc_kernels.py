"""Compiled observation-by-observation updates. No fast-math or time skipping."""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def signal(x, theta, gamma, rounded_sign):
    e = x-theta
    if gamma == 0:
        if e == 0:
            return rounded_sign
        return 1. if e > 0 else -1.
    z = abs(e)
    if z <= 1.:
        value = z if gamma == 1 else z*z if gamma == 2 else z*z*z
    else:
        value = 1.+gamma*(z-1.)
    return value if e > 0 else -value

@njit(cache=True)
def tuning_block(noise, start, x, a_values, theta, gamma, rounded_sign):
    for j in range(len(noise)):
        k = start+j
        for m in range(len(x)):
            x[m] -= a_values[m]/k*(signal(x[m],theta,gamma,rounded_sign)+noise[j])

@njit(cache=True)
def comparator_block(noise, start, state, a, theta, gamma, rounded_sign):
    # state: five locations, OLS x/y means and centered sums, PJ sum, clip counts.
    for j in range(len(noise)):
        k = start+j
        eps = noise[j]
        ya = signal(state[3],theta,gamma,rounded_sign)+eps
        state[9] += state[4]
        dx, dy = state[3]-state[5], ya-state[6]
        state[5] += dx/k
        state[6] += dy/k
        state[7] += dx*(state[3]-state[5])
        state[8] += dx*(ya-state[6])
        slope = state[8]/state[7] if state[7] > 0 else 1.
        if slope < .1: state[10] += 1
        if slope > 10.: state[11] += 1
        b = min(10.,max(.1,slope))
        for m in range(3):
            multiplier = .25 if m == 0 else 1. if m == 1 else 4.
            state[m] -= (a*multiplier)*(signal(state[m],theta,gamma,rounded_sign)+eps)/k
        state[3] -= ya/(k*b)
        yp = signal(state[4],theta,gamma,rounded_sign)+eps
        state[4] = min(1.,max(0.,state[4]-k**(-.75)*yp))

@njit(cache=True)
def scan_test(noise, signal_value, previous_sum, j, stage, K, v):
    # Match the original block cumsum convention: previous_sum + cumsum(Y).
    partial = 0.
    for k in range(len(noise)):
        partial += signal_value+noise[k]
        current = previous_sum+partial
        index = j+k+1
        threshold = v*math.sqrt(2.*index*(K+stage)*math.log1p(index))
        if abs(current) > threshold:
            return k+1, current, True
    return len(noise), previous_sum+partial, False

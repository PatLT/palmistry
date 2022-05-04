#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16th 2022

@author: Pat Taylor (pt409)
"""
#%% BLOCK 1
import sympy as sp
from itertools import product
import dill

dill.settings['recurse'] = True

n_max = 3 # max number of component systems to do calculation for. 

def gen_Ab_sys(n,m=2):
    # n = number of elements
    # m = number of phases
    # Constraint tolerances
    tol_a, tol_b = sp.symbols("tol_a tol_b")

    # Corrections to log partitioning coefficients
    dp = sp.symbols(" ".join(
        [" ".join(
            ["dp_{}^{}".format(i,phi) for i in range(n)]
            ) for phi in range(m)]
            ))
    # Initial predicted phase compositions
    xi = sp.symbols(" ".join(
        [" ".join(
            ["xhat_{}^{}".format(i,phi) for i in range(n)]
            ) for phi in range(m)]
            ))

    # Initial predicted phase fraction and correction 
    fi, dq = sp.symbols("fhat dq")
    # Input/overall composition 
    c = sp.symbols(" ".join(["c_{}".format(i) for i in range(n)]))

    # Relations
    x = [xi[phi*n+i]*sp.exp(dp[phi*n+i]) for phi,i in product(range(m),range(n))]

    f = [1-(fi*(1+sp.tanh(dq)))/(1+(2*fi-1)*sp.tanh(dq)),
            (fi*(1+sp.tanh(dq)))/(1+(2*fi-1)*sp.tanh(dq))]

    # Terms in lagrangian derivative
    # wrt dp
    L1 = [-(
        (c[j]-sum([x[phi*n+j]*f[phi] for phi in range(m)]))/(c[j]*tol_a)*f[psi] + \
            (1-sum([x[psi*n+i] for i in range(n)]))/tol_b
        )*sp.diff(x[psi*n+j],dp[psi*n+j])
    for psi,j in product(range(m),range(n))]
    # wrt dq
    L2 = -sum([
        sum([(c[i]-sum(x[phi*n+i]*f[phi] for phi in range(m)))/(c[i]*tol_a)*x[psi*n+i] for i in range(n)])\
            * sp.diff(f[psi],dq)
    for psi in range(m)])

    # Series expansion
    t = sp.symbols("t")
    t_sub = [(dq,dq*t)] + [(dp_i_phi,dp_i_phi*t) for dp_i_phi in dp]
    L1 = [L1_psi_j.subs(t_sub) for L1_psi_j in L1]
    L2 = L2.subs(t_sub)

    # Linearise both expressions
    L1_l = [sp.series(L1_psi_j,x=t,n=2).removeO().subs(t,1) for L1_psi_j in L1]
    L2_l = sp.series(L2,x=t,n=2).removeO().subs(t,1)

    # Convert to matrix form
    A,b = sp.linear_eq_to_matrix(L1_l+[L2_l],list(dp)+[dq])
    return sp.lambdify((*c,*xi,fi,tol_a,tol_b),A), sp.lambdify((*c,*xi,fi,tol_a,tol_b),b)

# Version of above to use for a hard constraint applied to sum over phase components = 1 
def gen_Ab_sys_hc(n,m=2):
    # n = number of elements
    # m = number of phases
    # Constraint tolerances
    tol = sp.symbols(" ".join(["tol_{}".format(i) for i in range(n)]))

    # Corrections to log partitioning coefficients
    dp = sp.symbols(" ".join(
        [" ".join(
            ["dp_{}^{}".format(i,phi) for i in range(n)]
            ) for phi in range(m)]
            ))
    # Initial predicted phase compositions
    xi = sp.symbols(" ".join(
        [" ".join(
            ["xhat_{}^{}".format(i,phi) for i in range(n)]
            ) for phi in range(m)]
            ))

    # Initial predicted phase fraction and correction 
    fi, dq = sp.symbols("fhat dq")
    # Input/overall composition 
    c = sp.symbols(" ".join(["c_{}".format(i) for i in range(n)]))
    # "Chemical potential"
    mu = sp.symbols(" ".join(["mu^{}".format(phi) for phi in range(m)]))

    # Relations
    x = [xi[phi*n+i]*(1+dp[phi*n+i]) for phi,i in product(range(m),range(n))] # <-- This is also modified to just use the linear expression. 

    f = [1-(fi*(1+sp.tanh(dq)))/(1+(2*fi-1)*sp.tanh(dq)),
            (fi*(1+sp.tanh(dq)))/(1+(2*fi-1)*sp.tanh(dq))]

    # Terms in lagrangian derivative
    # wrt dp
    L1 = [-(
        (c[j]-sum([x[phi*n+j]*f[phi] for phi in range(m)]))/(c[j]*tol[j])**2*f[psi]
            #(1-sum([x[psi*n+i] for i in range(n)]))/tol_b    <-- This line gets modified
        )*sp.diff(x[psi*n+j],dp[psi*n+j])
        - mu[psi]*c[j]*xi[psi*n+j]
    for psi,j in product(range(m),range(n))]
    # wrt dq
    L2 = -sum([
        sum([(c[i]-sum(x[phi*n+i]*f[phi] for phi in range(m)))/(c[i]*tol[i])**2*x[psi*n+i] for i in range(n)])\
            * sp.diff(f[psi],dq)
    for psi in range(m)])
    # wrt mu
    L3 = [1-sum([xi[psi*n+i]*(1+dp[psi*n+i]) for i in range(n)])
    for psi in range(m)]

    # Series expansion
    t = sp.symbols("t")
    t_sub = [(dq,dq*t)] + [(dp_i_phi,dp_i_phi*t) for dp_i_phi in dp] + [(mu_phi,mu_phi*t) for mu_phi in mu]
    L1 = [L1_psi_j.subs(t_sub) for L1_psi_j in L1]
    L2 = L2.subs(t_sub)
    L3 = [L3_psi.subs(t_sub) for L3_psi in L3]

    # Linearise both expressions
    L1_l = [sp.series(L1_psi_j,x=t,n=2).removeO().subs(t,1) for L1_psi_j in L1]
    L2_l = sp.series(L2,x=t,n=2).removeO().subs(t,1)
    L3_l = [sp.series(L3_psi,x=t,n=2).removeO().subs(t,1) for L3_psi in L3]

    # Convert to matrix form
    A,b = sp.linear_eq_to_matrix(L1_l+[L2_l]+L3_l,list(dp)+[dq]+list(mu))
    return sp.lambdify((*c,*xi,*tol,fi),A), sp.lambdify((*c,*xi,*tol,fi),b)

# %% BLOCK 2
d_lambda_out = {}
for n in range(2,n_max+1):
    d_lambda_out[n] = gen_Ab_sys_hc(n)

with open("correction_Ab_system.dill","wb") as outfile:
    dill.dump(d_lambda_out,outfile)

# %%

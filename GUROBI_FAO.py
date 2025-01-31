#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:33:13 2025

@author: marisol
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import random
from random import shuffle
import numpy as np

import matplotlib

import sys
import csv
import time
from IPython import get_ipython




#import data ---------------------------------------------------------------------------------------
cdata=pd.read_excel(r'IN_5.xlsx', sheet_name='clientes')
cdata.head()

cdf = pd.DataFrame(cdata, columns=['cliente','pobtot'])
cdf.head()


pdata=pd.read_excel(r'IN_5.xlsx', sheet_name='farmacias')
pdata.head()

pdf = pd.DataFrame(pdata, columns=['farmacia','capacidad','cost','ac'])
pdf.head()


mdata=pd.read_excel(r'IN_5.xlsx', sheet_name='clinicas')
mdata.head()

mdf = pd.DataFrame(mdata, columns=['clinica','capacidad','cost','ac'])
mdf.head()


ndata=pd.read_excel(r'IN_5.xlsx', sheet_name='enfermeras') #clientes-clinicas
ndata.head()

ndf = pd.DataFrame(ndata, columns=['nurse','capacity','cost'])
ndf.head()


rfcdata=pd.read_excel(r'IN_5.xlsx', sheet_name='rutas') #clientes-farmacias
rfcdata.head()

rfcdf = pd.DataFrame(rfcdata, columns=['origen','destino','distancia'])
rfcdf.head()


rmdata=pd.read_excel(r'IN_5.xlsx', sheet_name='rutas-clinicas') #clientes-clinicas
rmdata.head()

rmdf = pd.DataFrame(rmdata, columns=['origen','destino','distancia'])
rmdf.head()


mndata=pd.read_excel(r'IN_5.xlsx', sheet_name='clinicas-enfermeras') #clientes-clinicas
mndata.head()

mndf = pd.DataFrame(mndata, columns=['origen','destino','distancia'])
mndf.head()
#number of nodes -------------------------------------------------------------

ni = len(cdf.iloc[0:])# number of demand nodes
nph = len(pdf.iloc[0:])# number of pharmacies
nm = len(mdf.iloc[0:])# number of mobile clinics
nn = len(ndf.iloc[0:])# number of nurses

#SETS-------------------------------------------------------------------------

#nodos demanda,pob tot, pob 60+ ----------------------------------------------------------------------
ccc={}
for c, row in cdf.iterrows():
    cc = {cdf.iloc[c,0]:cdf.iloc[c,1]}
    ccc.update(cc)
I, h= gp.multidict(ccc)

#farmacias, capacidad y opening costs --------------------------------------------------------------------
ppp={}
for p, row in pdf.iterrows():
    pp = {pdf.iloc[p,0]:list(pdf.iloc[p,1:4])}
    ppp.update(pp)
P, kp, OCp, ACp= gp.multidict(ppp)

ppp={}
for p, row in mdf.iterrows():
    pp = {mdf.iloc[p,0]:list(mdf.iloc[p,1:4])}
    ppp.update(pp)
M, km, OCm, ACm= gp.multidict(ppp)

#nurses
ppp={}
for p, row in ndf.iterrows():
    pp = {ndf.iloc[p,0]:list(ndf.iloc[p,1:4])}
    ppp.update(pp)
N, kn, OCn = gp.multidict(ppp)

#distancia cliente-farmacia vehiculo terrestre
rrr={}
for r, row in rfcdf.iterrows():
    rr = {tuple([rfcdf.iloc[r,0],rfcdf.iloc[r,1]]):list(rfcdf.iloc[r,2:3])}
    rrr.update(rr)
distance_i_p, dip = gp.multidict(rrr)  #

#distancia cliente-farmacia vehiculo terrestre
rrr={}
for r, row in rmdf.iterrows():
    rr = {tuple([rmdf.iloc[r,0],rmdf.iloc[r,1]]):list(rmdf.iloc[r,2:3])}
    rrr.update(rr)
distance_i_m, dim = gp.multidict(rrr)  #

#####################################3
for i in I:
    for p in P:
        if dip[i,p]>0:
            dip[i,p]=dip[i,p]
        else:
            dip[i,p]=5

for i in I:
    for m in M:
        if dim[i,m]>0:
            dim[i,m]=dim[i,m]
        else:
            dim[i,m]=5
########################################

#distancia clinica-enfermera
rrr={}
for r, row in mndf.iterrows():
    rr = {tuple([mndf.iloc[r,0],mndf.iloc[r,1]]):list(mndf.iloc[r,2:3])}
    rrr.update(rr)
distance_m_n, dmn = gp.multidict(rrr)  #

MAX_D = 5 #KM

#FARACIAS QUE PUEDEN CUBRIR EL NODO DE DEMANDA I
N_I={}
for i in I:
    cc=[]
    for p in P:
        c = p
        if str(dip[i,p]) != 'nan' and dip[i,p]<= MAX_D:
            cc.append(c)
            N_Ii = {i:cc}
            N_I.update(N_Ii)

for i in I:
    if i not in N_I:
        N_Ii= {i:[]}
        N_I.update(N_Ii)

#mobile clinic cost/month https://www.mobilehealthmap.org/how-much-it-costs-to-run-a-mobile-clinic/#:~:text=Ballpark%20estimate%20of%20mobile%20clinic,operating%20cost%20is%20approximately%20%24275%2C000.

P_I = {} #farmacias que pueden cubrir el nodo de demanda i
for p in P:
    cc=[]
    for i in I:
        if str(dip[i,p]) != 'nan' and dip[i,p] <= MAX_D:
            c = i
            cc.append(c)
            P_Ii = {p:cc}
            P_I.update(P_Ii)
for p in P:
    if p not in P_I:
        P_Ii = {p:[]}
        P_I.update(P_Ii)

#CLIENTES QUE NO SE PUEDE CUBIR POR NINGUNA FARMACIA
IM=[]
for i in I:
    if N_I[i] == []:
        IM.append(i)
M_I={}
for i in IM:
    for m in M:
        M_Ii = {m:IM}
        M_I.update(M_Ii)

I_M={}
for i in I:
    for m in M:
        if i not in IM:
            I_Mm = {i:[]}
            I_M.update(I_Mm)
        else:
            I_Mm = {i:M}
            I_M.update(I_Mm)

IP=[]
for i in I:
    if i not in IM:
        IP.append(i)

demand={}
for p in P:
    d=0
    for i in P_I[p]:
        d=d+h[i]
    dd={p:d}
    demand.update(dd)
    
    
    
    
    
MAX_P = 10000000  # Maximum number of pharmacies
MAX_M = 10000000  # Maximum number of mobile clinics
BIG1 = 100
BIG2 = 500
BIG = 10000000
o_v = []
r_t = []
x_l = []
y_l = []

for B in range(200000000, 500000000, 50000000):  # Loop over different budget ranges

   
    # Initialize Gurobi Model
    mo = gp.Model("IFORS")

    # Variables
    u = {}  # Allocation for immunization service to pharmacy
    for i in I:
        u[i] = mo.addVar(lb=0.0, ub=float(1), vtype=GRB.CONTINUOUS, name=f'u[{i}]')

    w_ip = {}  # Allocation for immunization service to pharmacy
    for i in I:
        for p in P:
            w_ip[i, p] = mo.addVar(lb=0.0, ub=float(1), vtype=GRB.CONTINUOUS, name=f'w_ip[{i},{p}]')

    w_imn = {}  # Allocation for immunization service to mobile clinic
    for i in I:
        for m in M:
            for n in N:
                w_imn[i, m, n] = mo.addVar(lb=0.0, ub=float(1), vtype=GRB.CONTINUOUS, name=f'w_imn[{i},{m},{n}]')

    # Binary variables for service allocation to pharmacies and mobile clinics
    z_m = {}
    for m in M:
        z_m[m] = mo.addVar(vtype=GRB.BINARY, name=f'z_m[{m}]')

    z_ip = {}
    for i in I:
        for p in P:
            z_ip[i, p] = mo.addVar(vtype=GRB.BINARY, name=f'z_ip[{i},{p}]')

    # Binary variables indicating pharmacy or mobile clinic allocations
    xp = mo.addVars(P, vtype=GRB.BINARY, name="xp")  # Pharmacy selection
    xmn = mo.addVars(M, N, vtype=GRB.BINARY, name="xmn")  # Mobile clinic allocation

    ######################## Add Constraints ########################

    # Constraint for total costs
    mo.addConstr(gp.quicksum(OCp[p] * xp[p] for p in P) + gp.quicksum((OCm[m]) * z_m[m] for m in M) + gp.quicksum((OCn[n]) * xmn[m, n] for m in M for n in N) 
                 + gp.quicksum(ACp[p] * dip[i, p] * h[i] *w_ip[i, p] for i in I for p in P)
                 + gp.quicksum(ACm[m] * dim[i, m] *h[i]* w_imn[i, m,n] for i in I for m in M for n in N) <= B)

   
    #cONSTRAINTS
    
    for m in M:
        mo.addConstr(gp.quicksum(xmn[m,n] for n in N) <= BIG1*z_m[m])
    
    for i in I:
        mo.addConstr(gp.quicksum(w_ip[i,p] for p in P_I) +u[i] ==1)
    
    
    # Constraints on allocations
    for i in I:
        for p in P:
            mo.addConstr(BIG2 * (w_ip[i, p]) >= BIG1 * z_ip[i, p])

    for p in P:
        mo.addConstr(kp[p] * xp[p] - gp.quicksum(w_ip[i,p]*h[i] for i in I) >= 0)
        
    mo.addConstr(gp.quicksum(xp[p] for p in P) <= MAX_P)
    
    for p in P:
        mo.addConstr(gp.quicksum(w_ip[i,p] for i in I) <= BIG1*xp[p])
    

    # Objective function: Minimize uncovered demand
    of = gp.quicksum(h[i] * u[i] for i in I)

    # Solve the optimization model
    mo.setParam('MIPGap', 0)
    mo.setObjective(of, GRB.MINIMIZE)
    mo.optimize()
    

    # Output results
    h_v = gp.quicksum(h[i] for i in I)
    ov = mo.objVal
    ud = (ov / h_v.getValue()) * 100  # Uncovered demand percentage
    cd = 100 - ud  # Covered demand percentage
    runtime = mo.Runtime

    total_cost = (gp.quicksum(OCp[p] * xp[p] for p in P) + gp.quicksum((OCm[m]) * z_m[m] for m in M) + gp.quicksum((OCn[n]) * xmn[m, n] for m in M for n in N) 
                 + gp.quicksum(ACp[p] * dip[i, p] * h[i] *w_ip[i, p] for i in I for p in P)
                 + gp.quicksum(ACm[m] * dim[i, m] *h[i]* w_imn[i, m,n] for i in I for m in M for n in N)).getValue()

    o_v.append(ov)
    r_t.append(runtime)
    x_l.append(cd)
    y_l.append(total_cost)

    print(f'Objective value: {mo.objVal}')
    print(f'Total cost: {total_cost}')
    print(f"The CPU time is {runtime}")
    print(f"The Uncovered demand is {ud}%")
    print(f"The Covered demand is {cd}%")
    
    # Set Gurobi heuristic parameters
    mo.setParam('Heuristics', 0.5)  # Allow Gurobi to use its heuristics (50% weight)
    mo.setParam("MIPFocus", 1)  # Focus on finding better feasible solutions

    # Solve the model
    mo.optimize()

    # Extract results
    '''if mo.status == GRB.OPTIMAL or mo.status == GRB.FEASIBLE:
        xp_solution = {p: xp[p].x for p in P}
        x_mn_solution = {(m, n): xmn[m, n].x for m in M for n in N}
        print("Pharmacy allocation:", xp_solution)
        print("Mobile clinic allocation:", x_mn_solution)
    else:
        print("No feasible solution found.")
        
    if mo.status == GRB.OPTIMAL or mo.status == GRB.FEASIBLE:
        x_mn_solution = {(m, n): xmn[m, n].x for m in M for n in N if xmn[m, n].x > 0}
        print("Mobile clinic allocation (selected locations):", x_mn_solution)
    else:
        print("No feasible solution found.")'''


##############################################################################
########################### PLOT THE RESULTS ##############################
##############################################################################

plt.plot(y_l, x_l)
plt.xlabel('Total cost (USD)')
plt.ylabel('Covered demand (%)')
plt.title("Covered demand vs Cost (IN_5)")
plt.show()


    
    
    
    
    
    
    
    
    
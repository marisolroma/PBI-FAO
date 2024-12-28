#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:01:01 2023

@author: MarisolRoma
"""


#IMPORT DATAFRAMES
import gurobipy as gp
import pandas as pd

#import data ---------------------------------------------------------------------------------------
cdata=pd.read_excel(r'instancia5.xlsx', sheet_name='clientes')
cdata.head()

cdf = pd.DataFrame(cdata, columns=['cliente','pobtot'])
cdf.head()


pdata=pd.read_excel(r'instancia5.xlsx', sheet_name='farmacias')
pdata.head()

pdf = pd.DataFrame(pdata, columns=['farmacia','capacidad','cost','ac'])
pdf.head()


mdata=pd.read_excel(r'instancia5.xlsx', sheet_name='clinicas')
mdata.head()

mdf = pd.DataFrame(mdata, columns=['clinica','capacidad','cost','ac'])
mdf.head()


ndata=pd.read_excel(r'instancia5.xlsx', sheet_name='enfermeras') #clientes-clinicas
ndata.head()

ndf = pd.DataFrame(ndata, columns=['nurse','capacity','cost'])
ndf.head()


rfcdata=pd.read_excel(r'instancia5.xlsx', sheet_name='rutas') #clientes-farmacias
rfcdata.head()

rfcdf = pd.DataFrame(rfcdata, columns=['origen','destino','distancia'])
rfcdf.head()


rmdata=pd.read_excel(r'instancia5.xlsx', sheet_name='rutas-clinicas') #clientes-clinicas
rmdata.head()

rmdf = pd.DataFrame(rmdata, columns=['origen','destino','distancia'])
rmdf.head()


mndata=pd.read_excel(r'instancia5.xlsx', sheet_name='clinicas-enfermeras') #clientes-clinicas
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
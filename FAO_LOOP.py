#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:43:18 2023

@author: MarisolRoma
"""

##############################################################################
####################       FIX-AND-OPTIMIZE LOOP           ###################
##############################################################################


#################            IMPORT LIBRARIES            ####################
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

import sys
import csv
import time


import random
from random import shuffle

from IPython import get_ipython



######################        IMPORT DATA          ##########################
from data_import import I,P,M,N,h,kp,km,kn,OCp,OCm,OCn,ACp,ACm,dip,dim,dmn,N_I,P_I,IM,M_I,I_M,IP,demand



######################        PARÁMETROS          ###########################
MAX_P = 10000000#(3/4)*nph
MAX_M=10000000
BIG1 = 100
BIG2=500
BIG=10000000



                    
##############################################################################             
############################## MODEL #########################################    
##############################################################################             

MAX_P = 10000000#(3/4)*nph
MAX_M=10000000
BIG1 = 100
BIG2=500
BIG=10000000
#B=25000000

o_v=[]
r_t=[]
x_l=[]
y_l=[]


for B in range(20000000,40000000,2000000): 
    #################       IMPORT GA PHARMACIES          ######################
    
    from FAO_MC import best_cost_mc,best_gen_mc,costs
    
    
    ###############       IMPORT GA MOBILE CLINICS        ######################
    from FAO_P import best_cost_p, best_gen_p, results

 
    mo=gp.Model("IFORS")
                
        #VARIABLES -------------------------------------------------------------
    u={} # 1 if an immunization service is allocated to pharmacy site j, 0 otherwise
    for i in I:
        u[i]=mo.addVar(lb=0.0, ub=float(1),vtype=GRB.CONTINUOUS, name='u[{}]'.format(i))
            
    w_ip={} # 1 if an immunization service is allocated to pharmacy site j, 0 otherwise
    for i in I:
        for p in P:
            w_ip[i,p]=mo.addVar(lb=0.0, ub=float(1), vtype=GRB.CONTINUOUS, name='w_ip[{},{}]'.format(i,p))
    
    w_imn={} # 1 if an immunization service is allocated to pharmacy site j, 0 otherwise
    for i in I:
        for m in M:
            for n in N:
                w_imn[i,m,n]=mo.addVar(lb=0.0, ub=float(1), vtype=GRB.CONTINUOUS, name='w_imn[{},{},{}]'.format(i,m,n))
    z_im={}
    for i in I:
        for m in M:
            z_im[i,m]=mo.addVar(vtype=GRB.BINARY, name='z_im[{}{}]'.format(i,m))
    z_ip={}
    for i in I:
        for p in P:
            z_ip[i,p]=mo.addVar(vtype=GRB.BINARY, name='z_ip[{}{}]'.format(i,p))
    
    xp={} # 1 if an immunization service is allocated to pharmacy site j, 0 otherwise
    for p in P:
        xp[p]=mo.addVar(vtype=GRB.BINARY, name='xp[{}]'.format(p))
        
    xmn={} # 1 if an immunization service is allocated to pharmacy site j, 0 otherwise
    for m in M:
        for n in N:
            xmn[m,n]=mo.addVar(vtype=GRB.BINARY, name='xmn[{},{}]'.format(m,n))
            
      
    ########### agregar valores ###############
    c=0
    xp={}
    for p in P:
        xp[p]=best_gen_p[c]
        c=c+1
        
    c=0
    xmn={}
    for m in M:
        for n in N:
            xmn[m,n]=best_gen_mc[c]
            c=c+1    
    ##########################################   
   
    
    
    mo.addConstr(gp.quicksum(OCp[p]*xp[p] for p in P) + gp.quicksum((OCm[m]+OCn[n])*xmn[m,n] for m in M for n in N )+ gp.quicksum(ACp[p]*dip[i,p]*z_ip[i,p] for i in I for p in P) + gp.quicksum(ACm[m]*dim[i,m]*z_im[i,m] for i in I for m in M) <= B) # use of pharmacy costs  +gp.quicksum(ACp*dip[i,p]*z_ip[i,p] for i in I for p in P)
    
    
    
    for i in I:
        for p in P:
            mo.addConstr(BIG2*(w_ip[i,p])>= BIG1*z_ip[i,p])
                
    for i in I:
        for m in M:
            mo.addConstr((gp.quicksum(w_imn[i,m,n]for n in N))*BIG2 >= BIG1*z_im[i,m])
    #----------constr extra
    for i in I:
        mo.addConstr(gp.quicksum(w_ip[i,p] for p in N_I[i]) + gp.quicksum(w_imn[i,m,n] for m in I_M[i] for n in N)  + u[i] == 1)
    #------------------------    
        
    #for p in P:
    #    for i in I:
    #        mo.addConstr(dip[i,p]*z_ip[i,p]<=MAX_D)
    
    for p in P:
        mo.addConstr(gp.quicksum(z_ip[i,p] for i in I)-xp[p]==0)
    
    #for p in PI:
    #    mo.addConstr(xp[p]==0)
    
    for p in P:
        mo.addConstr(kp[p]*xp[p] - gp.quicksum(w_ip[i,p]*h[i] for i in P_I[p]) >= 0)
    
    for m in M:
        mo.addConstr(gp.quicksum((km[m] - 3*dmn[m,n])*xmn[m,n] for n in N) - gp.quicksum(w_imn[i,m,n]*h[i] for i in M_I[m] for n in N) >= 0)
        
    for m in M:
        mo.addConstr(gp.quicksum(xmn[m,n] for n in N )<=5)
        
    for n in N:
        mo.addConstr(kn[n]*gp.quicksum(xmn[m,n] for m in M)-gp.quicksum(w_imn[i,m,n]*h[i] for i in I for m in M)>=0)
        
    mo.addConstr(gp.quicksum(xp[p] for p in P) <= MAX_P)
    
    #tentativa
    mo.addConstr(gp.quicksum(xmn[m,n] for m in M for n in N) <= MAX_M)
    
    for i in I:
        for p in P:
            mo.addConstr(w_ip[i,p]<= BIG*xp[p])
    
    for i in I:
        for m in M:
            for n in N:
                mo.addConstr(w_imn[i,m,n]<= BIG*xmn[m,n])
                     
    
    # OBJECTIVE FUNCTION ---------------------------------------------------------
    of=gp.quicksum(h[i]*u[i] for i in I)

    
        
    
    # SOLVE ----------------------------------------------------------------------
    mo.setParam('MIPGap', 0.1)
    mo.setObjective(of,GRB.MINIMIZE)
    mo.optimize()
    
    
    
    h_v=gp.quicksum(h[i] for i in I)
    ov=mo.objVal
    
    ud=(ov/h_v.getValue())*100
    cd=100-ud
    runtime = mo.Runtime
    
    total_cost=(gp.quicksum(OCp[p]*xp[p] for p in P) + gp.quicksum((OCm[m]+OCn[n])*xmn[m,n] for m in M for n in N )+ gp.quicksum(ACp[p]*dip[i,p]*z_ip[i,p] for i in I for p in P) + gp.quicksum(ACm[m]*dim[i,m]*z_im[i,m] for i in I for m in M) ).getValue()
    o_v.append(ov)
    r_t.append(runtime)
    x_l.append(cd)
    y_l.append(total_cost)
    print('Obj value: %g' % mo.objVal)
    print('Total cost: ', total_cost)
    print("The CPU is %f" % runtime)
    print("The Uncovered demand is %f " % ud, '%')
    print("The Covered demand is %f " % cd, '%')
    
    #get_ipython().magic('reset -sf')



##############################################################################
##############################      PLOT     #################################
##############################################################################

plt.plot(y_l,x_l)
plt.xlabel('total cost (USD)')
plt.ylabel('covered demand (%)')
plt.title("Covered demand respect cost (IN_6)")
plt.show()
    
  


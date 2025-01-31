#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:05:32 2025

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
cdata=pd.read_excel(r'IN_6.xlsx', sheet_name='clientes')
cdata.head()

cdf = pd.DataFrame(cdata, columns=['cliente','pobtot'])
cdf.head()


pdata=pd.read_excel(r'IN_6.xlsx', sheet_name='farmacias')
pdata.head()

pdf = pd.DataFrame(pdata, columns=['farmacia','capacidad','cost','ac'])
pdf.head()


mdata=pd.read_excel(r'IN_6.xlsx', sheet_name='clinicas')
mdata.head()

mdf = pd.DataFrame(mdata, columns=['clinica','capacidad','cost','ac'])
mdf.head()


ndata=pd.read_excel(r'IN_6.xlsx', sheet_name='enfermeras') #clientes-clinicas
ndata.head()

ndf = pd.DataFrame(ndata, columns=['nurse','capacity','cost'])
ndf.head()


rfcdata=pd.read_excel(r'IN_6.xlsx', sheet_name='rutas') #clientes-farmacias
rfcdata.head()

rfcdf = pd.DataFrame(rfcdata, columns=['origen','destino','distancia'])
rfcdf.head()


rmdata=pd.read_excel(r'IN_6.xlsx', sheet_name='rutas-clinicas') #clientes-clinicas
rmdata.head()

rmdf = pd.DataFrame(rmdata, columns=['origen','destino','distancia'])
rmdf.head()


mndata=pd.read_excel(r'IN_6.xlsx', sheet_name='clinicas-enfermeras') #clientes-clinicas
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
    

def SA_MC():
    def initial_sol():
        # Initial solution with random selection of mobile clinics and nurses
        solution = [random.choice([0, 1]) for _ in range(len(M) * len(N))]
        return solution

    def coverage_mc(S):
        # Calculate the total demand coverage based on the current solution
        total_coverage = 0
        i = 0
        for m in M:
            for n in N:
                if S[i] == 1:  # If this clinic-nurse pair is selected
                    total_coverage += dmn[m, n]  # Add the demand covered by this pair
                i += 1
        return total_coverage

    def acceptance_probability(old_coverage, new_coverage, temperature):
        # Accept better solutions (maximize coverage)
        if new_coverage > old_coverage:
            return 1.0  # Always accept a better solution
        else:
            # Probability of accepting a worse solution decreases with temperature
            return math.exp((new_coverage - old_coverage) / temperature)

    def simulated_annealing():
        current_solution = initial_sol()
        current_coverage = coverage_mc(current_solution)
        best_solution = current_solution[:]
        best_coverage = current_coverage

        temperature = 1000.0  # Starting temperature
        cooling_rate = 0.95  # Rate at which the temperature decreases
        min_temperature = 0.1  # Minimum temperature for stopping

        coverages = [current_coverage]  # List to track coverage at each step

        while temperature > min_temperature:
            # Generate new solution by randomly changing one element
            new_solution = current_solution[:]
            i = random.randint(0, len(new_solution) - 1)
            new_solution[i] = 1 - new_solution[i]  # Flip the element at index i

            new_coverage = coverage_mc(new_solution)

            # Accept the new solution based on the acceptance probability
            if random.random() < acceptance_probability(current_coverage, new_coverage, temperature):
                current_solution = new_solution
                current_coverage = new_coverage

                # Update the best solution found so far
                if new_coverage > best_coverage:
                    best_solution = new_solution[:]
                    best_coverage = new_coverage

            # Decrease the temperature
            temperature *= cooling_rate
            coverages.append(current_coverage)

        return best_coverage, best_solution, coverages

    # Run the Simulated Annealing algorithm
    best_coverage, best_solution, coverages = simulated_annealing()

    # Optional: Plot the coverage evolution
    plt.plot(coverages)
    plt.xlabel("Iterations")
    plt.ylabel("Demand Coverage")
    plt.title("Simulated Annealing for Mobile Clinics - Maximizing Demand Coverage")
    plt.show()

    return best_coverage, best_solution, coverages

# Uncomment this line to run the Simulated Annealing algorithm:
#best_coverage, best_solution, coverages = SA_MC()



# Simulated Annealing for Pharmacy Selection

def SA_P():
    def initial_sol():
        # Generate a random initial solution
        rr = int(random.uniform(0, len(P) / 2))
        S = rr * [1] + (len(P) - rr) * [0]
        shuffle(S)
        return S

    def demand_pharmacies(S):
        # Calculate covered demand based on the solution
        covered_demand = 0
        for r in range(len(P)):
            if S[r] == 1:
                # Check if the pharmacy can cover the demand
                if kp[P[r]] - demand[P[r]] <= 0:
                    covered_demand += kp[P[r]]  # Add full capacity of the pharmacy
                else:
                    covered_demand += demand[P[r]]  # Add the demand covered by the pharmacy
        return covered_demand

    def cost_p(S):
        # Calculate cost for the solution (fixed cost + operational cost)
        cost = 0
        for i in range(len(P)):
            if S[i] == 1:
                cost += OCp[P[i]]  # Sum operational cost for selected pharmacies
        return cost

    def perturb_solution(S):
        # Create a neighboring solution by flipping a random pharmacy selection
        new_S = S.copy()
        flip_index = random.randint(0, len(S) - 1)
        new_S[flip_index] = 1 - new_S[flip_index]
        return new_S

    def acceptance_probability(current_demand, new_demand, temperature):
        # Accept the new solution based on the temperature and demand difference
        if new_demand > current_demand:
            return 1  # Always accept a better solution (higher coverage)
        else:
            return math.exp((new_demand - current_demand) / temperature)  # Accept with a probability based on temperature

    def simulated_annealing(M_p, T_initial, T_min, alpha):
        # Simulated Annealing procedure
        current_solution = initial_sol()  # Generate initial solution
        current_demand = demand_pharmacies(current_solution)  # Calculate initial demand coverage
        best_solution = current_solution
        best_demand = current_demand  # Track the best demand coverage
        temperature = T_initial
        iteration = 0
        demands = []  # Track demand coverage over time
        costs = []  # Track costs over time (optional, for analysis)

        while temperature > T_min:
            # Generate a new solution by perturbing the current solution
            new_solution = perturb_solution(current_solution)
            new_demand = demand_pharmacies(new_solution)  # Calculate demand for new solution

            # Decide whether to accept the new solution
            if random.random() < acceptance_probability(current_demand, new_demand, temperature):
                current_solution = new_solution
                current_demand = new_demand

            # Track the best solution found
            if current_demand > best_demand:
                best_solution = current_solution
                best_demand = current_demand

            # Store results for plotting
            demands.append(current_demand)
            costs.append(cost_p(current_solution))  # Optional, to analyze cost evolution

            # Decrease temperature
            temperature *= alpha
            iteration += 1

        return best_solution, best_demand, demands, costs

    # Parameters for Simulated Annealing
    M_p = 200  # Number of iterations for each solution
    T_initial = 1000  # Initial temperature
    T_min = 1  # Minimum temperature
    alpha = 0.995  # Temperature decrease factor

    # Running the Simulated Annealing
    best_solution, best_demand, demands, costs = simulated_annealing(M_p, T_initial, T_min, alpha)

    # Plot the results (optional)
    plt.xlabel("Iterations")
    plt.ylabel("Covered Demand")
    plt.plot(demands)
    plt.show()

    return best_solution, best_demand, demands, costs

# Running the algorithm
#best_solution, best_demand, demands, costs = SA_P()
    

###################### PARAMETER SETUP ###########################
MAX_P = 10000000  # Maximum number of pharmacies
MAX_M = 10000000  # Maximum number of mobile clinics
BIG1 = 100
BIG2 = 500
BIG = 10000000

o_v = []
r_t = []
x_l = []
y_l = []

for B in range(200000000, 600000000, 40000000): # Loop over different budget ranges

    ################# Run Simulated Annealing for Pharmacies ##################
    best_gen_p, best_coverage_p, demands, costs_p = SA_P()  # Run SA for pharmacies


    ################# Run Simulated Annealing for Mobile Clinics ##################
    best_cost_mc, best_gen_mc, costs_mc = SA_MC()  # Run SA for mobile clinics

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
    xp = {}  # Allocation for pharmacy
    for p in P:
        xp[p] = best_gen_p.pop(0)  # Using SA solution for pharmacy

    xmn = {}  # Allocation for mobile clinic
    for m in M:
        for n in N:
            xmn[m, n] = best_gen_mc.pop(0)  # Using SA solution for mobile clinic

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

##############################################################################
########################### PLOT THE RESULTS ##############################
##############################################################################

plt.plot(y_l, x_l)
plt.xlabel('Total cost (USD)')
plt.ylabel('Covered demand (%)')
plt.title("Covered demand vs Cost (IN_6)")
plt.show()

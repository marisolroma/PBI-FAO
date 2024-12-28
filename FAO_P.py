#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:41:05 2023

@author: MarisolRoma
"""


import random
from random import shuffle
import matplotlib.pyplot as plt

from data_import import P,kp,P_I,demand,OCp

def FA0_P(B):
    def initial_sol():
        for i in range(10):
            rr=int(random.uniform(0,len(P)/2))
            S=rr*[1]+(len(P)-rr)*[0]
            shuffle(S)
            p_1=S
            shuffle(p_1)
            p_2=p_1
            if demand_pharmacies(p_1)>demand_pharmacies(p_2):
                best_parent=p_1
                best_parent_2=p_2
            else:
                best_parent=p_2
                best_parent_2=p_1
        
        return best_parent, best_parent_2
                
      
    def demand_pharmacies(S):
        covered_demand=0
        cd=0
        r=0
        for r in range(len(P)):
            if S[r]==1:
                if kp[P[r]]-demand[P[r]]<=0:
                    cd=kp[P[r]]
                if kp[P[r]]-demand[P[r]]>=0:
                    cd=demand[P[r]]
                covered_demand+=cd
        return covered_demand
    
    
    def cost_p(S):
        cost_p=0
        i=0
        for p in P:
            if S[i]==1:
                cost_p += (OCp[p])
                i=i+1
            else:
                i=i+1      
        return cost_p
    
    
    def p_GA(M_p,N_p,pc,pm):
        
        best_child_generation_n= []
        for m in range(M_p):
            best_parent,best_parent_2=initial_sol()
            parent_1=best_parent
            parent_2=best_parent_2
    
            for n in range(int(N_p/2)):
                #2 POINT CROSSOVER
                if random.uniform(0, 1) < pc:
                    c1=int(len(parent_1)/3)
                    c2=int(c1*2)
                    child_1=parent_1[0:c1]+parent_2[c1:c2]+parent_1[c2:]
                    child_2=parent_2[0:c1]+parent_1[c1:c2]+parent_2[c2:]
                    
                    ### MUTATION
                    child_1m=child_1
                    child_2m=child_2
                    cost_child_1m = 0
                    cost_child_2m = 0
                    for i in range(len(parent_1)):
                        if random.uniform(0, 1) < pm:
                            if child_1m[i]==0:
                                child_1m[i]=1
                            else:
                                child_1m[i]=0
                            
                            if child_2m[i]==0:
                                child_2m[i]=1
                            else:
                                child_2m[i]=0
                            
                    for i in range(len(P_I)):
                        if P_I[P[i]]==[]:
                            child_1m[i]=0
                            child_2m[i]=0
                            
                    cost_child_1m = demand_pharmacies(child_1m)
                    cost_child_2m = demand_pharmacies(child_2m)
                    if cost_child_1m > cost_child_2m and cost_p(child_1m)<B:
                        best_child_m = child_1m
                        best_cost = cost_child_1m
                        best_n = [best_cost,best_child_m]
                        best_child_generation_n.append(best_n) 
                    else:
                        best_child_m = child_2m
                        best_cost = cost_child_2m
                        best_n = [best_cost,best_child_m]
                        best_child_generation_n.append(best_n) 
                      
                best_gen_m_cost=1
                best_gen_m=0
                best_m=[]
                best_child_generation_m=[]
                for b_m in best_child_generation_n:
                    if b_m[0]>best_gen_m_cost:
                        best_gen_m_cost=b_m[0]
                        best_gen_m=b_m[1]  
                        best_m=[best_gen_m_cost,best_gen_m]
                        best_child_generation_m.append(best_m)
                        #costs.append(best_gen_m_cost)
            #global best_gen  
            best_gen=[]
            best_cost=1
            for b in best_child_generation_m:
                if b[0]>best_cost:
                    best_cost=b[0]
                    best_gen=b[1]
                    demands.append(best_cost)
                    
                else:
                    demands.append(best_cost)
        return best_gen, best_cost, demands
    
    
    
    
    M_p=100
    N_p=10
    pc=0.1
    pm=1
    
    demands=[]  
    best_cost_p=1
    results=[]
    for i in range (1):
        best_gen,best_cost, demands=p_GA(M_p,N_p,pc,pm)
        if best_cost > best_cost_p:
            best_cost_p=best_cost
            best_gen_p=best_gen
            results.append(best_cost_p)
            #plt.xlabel("# Iterations")
            #plt.ylabel("Covered demand")
            #plt.plot(demands)
            #plt.show()
            
    return best_cost_p, best_gen_p, results

best_cost_p, best_gen_p, results = FA0_P(B)
           
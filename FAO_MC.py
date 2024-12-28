#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:08:04 2023

@author: MarisolRoma
"""


import random
from random import shuffle
import matplotlib.pyplot as plt
from data_import import M,N,OCm,OCn,dmn,ACm

def FAO_MC():

    def initial_sol():
        for i in range(1000):
            r=int(random.uniform(0,len(N)))
            S=r*[1]+(len(N)*len(M)-r)*[0]
            shuffle(S)
            p_1=S
            shuffle(p_1)
            p_2=p_1
            if cost_mc(p_1)<cost_mc(p_2):
                best_parent=p_1
                best_parent_2=p_2
            else:
                best_parent=p_2
                best_parent_2=p_1
        return best_parent,best_parent_2
                
       
    # Define a function to calculate the location COST
    def cost_mc(S):
        cost_mc=0
        i=0
        sm=0
        for m in M:
            for n in N:
                if S[i]==1:
                    cost_mc += (OCn[n] + dmn[m,n]*ACm[m]) #dmn[m,n]*10000)
                    i=i+1
                    sm+=1
                else:
                    i=i+1
            #if sm>5:
            #    cost_mc += (OCm[m])+ 1000000000000)
        return cost_mc
    
    
    def mc_GA(M_p,N_p,pc,pm):
        
        best_child_generation_n= []
       
        for i in range(M_p):
            best_parent,best_parent_2=initial_sol()
            parent_1=best_parent
            parent_2=best_parent_2
            
            for n in range(int(N_p/2)):
                #2 POINT CROSSOVER
                if random.uniform(0, 1) <= pc:
                    c1=int(len(parent_1)/3)
                    c2=int(c1*2)
                    child_1=parent_1[0:c1]+parent_2[c1:c2]+parent_1[c2:]
                    child_2=parent_2[0:c1]+parent_1[c1:c2]+parent_2[c2:]
                    
                    ### MUTATION
                    child_1m=child_1
                    child_2m=child_2
                    cost_child_1m = 0
                    cost_child_2m = 0
                    for i in range(len(child_1)):
                        if random.uniform(0, 1) <= pm:
                            if child_1m[i]==0:
                                child_1m[i]=1
                            else:
                                child_1m[i]=0
                            
                            if child_2m[i]==0:
                                child_2m[i]=1
                            else:
                                child_2m[i]=0 
                 
                    cost_child_1m = cost_mc(child_1m)
                    cost_child_2m = cost_mc(child_2m)   
                    
                    if cost_child_1m < cost_child_2m:
                        best_child_m = child_1m
                        '''j=0
                        for m in M:
                            ss=0
                            for n in N:
                                ss=+best_child_m[j]
                                j+=1
                                if ss>5:
                                    best_child_m[j]'''
                        best_cost = cost_child_1m
                        best_n = [best_cost,best_child_m]
                        best_child_generation_n.append(best_n)
                        #costs.append(best_cost)
                        
                    else:
                        best_child_m = child_2m
                        best_cost = cost_child_2m
                        best_n = [best_cost,best_child_m]
                        best_child_generation_n.append(best_n) 
                        #costs.append(best_cost)
            ##########
            for child in best_child_generation_n:
                j=0
                feasible=0
                for m in M:
                    ss=0
                    for n in N:
                        ss=ss+child[1][j]
                        if ss>5:
                            child[1][j]=0
                            feasible+=1
                        j+=1
                        #break
                #if feasible > 0:
                #    best_child_generation_n.remove(child)
                    #print('removed')
                        
                #else:
                    #print('keep')
            ##########
                
            
            best_gen_m_cost=100000000000000000000
            best_gen_m=0
            best_m=[]
            best_child_generation_m=[]
            for b_m in best_child_generation_n:
                if b_m[0]<best_gen_m_cost:
                    best_gen_m_cost=b_m[0]
                    best_gen_m=b_m[1]  
                    best_m=[best_gen_m_cost,best_gen_m]
                    best_child_generation_m.append(best_m)
                    #costs.append(best_gen_m_cost)
     
        best_cost=100000000000000000000
        #global best_gen  
        best_gen2=[]
        for b in best_child_generation_m:
            if b[0]<best_cost:
                best_cost=b[0]
                best_gen2=b[1]
                costs.append(best_cost)
                
            else:
                costs.append(best_cost)
          
    
    
        return best_cost,best_gen2,costs
    
    
    M_p=80
    N_p=10
    pc=0.6
    pm=0.1
    
    results=[]
    costs=[]
    best_cost_mc=100000000000000000000000000000000
    for i in range (1):
        best_cost,best_gen2,costs=mc_GA(M_p,N_p,pc,pm)
    
        if best_cost < best_cost_mc:
            best_cost_mc=best_cost
            best_gen_mc=best_gen2
            costs.append(best_cost_mc)
            #plt.xlabel("# Iterations")
            #plt.ylabel("Cost")
            #plt.plot(costs)
            #plt.show()
            
    return best_cost_mc,best_gen_mc,costs

best_cost_mc,best_gen_mc,costs = FAO_MC()
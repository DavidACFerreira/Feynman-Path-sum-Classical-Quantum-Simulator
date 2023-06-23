import time
import math
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import copy
import tracemalloc
from itertools import product, islice
import memory_profiler as mem_profile
import csv


# importing Qiskit
from qiskit import BasicAer
from qiskit import Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, transpile, assemble
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import StatevectorSimulator

from qiskit.tools.visualization import plot_state_city, plot_histogram

def S(In, Out):
    if (In == 0 and Out == 0):
        return 1
    elif (In == 1 and Out == 1):
        return 1j
    else:
        return 0

def T(In, Out):
    if (In == 0 and Out == 0):
        return 1
    elif (In == 1 and Out == 1):
        return math.e**((np.pi/4)*1j)
    else:
        return 0

def H(In, Out):
    if (In == 1 and Out == 1):
        return -1/math.sqrt(2)
    else:
        return 1/math.sqrt(2)

def CX(In, Out):
    if (In[0] == 0 and Out != In):
        return 0
    elif (In[0] == 0 and Out == In):
        return 1
    elif (In == (1,0) and Out == (1,1)):
        return 1
    elif (In == (1,1) and Out == (1,0)):
        return 1
    else:
        return 0


# In[3]:


def CXmap(In):
    if (In == (0,0)):
        return (0,0)
    elif (In == (0,1)):
        return (0,1)
    elif (In == (1,0)):
        return (1,1)
    elif (In == (1,1)):
        return (1,0)
    

def forwardSweep(layers):
    res=[]
    depth = len(layers)
    for i in range(depth-1):
        a=[0 for i in range(amp_final.n_qbits)]
        for gate in (layers[i]):
            if(i==0):
                if(gate[0]=='S' or gate[0]=='T'):
                    a[gate[1]]=1
                if(gate[0]=='CX'): 
                    a[gate[1]]=1
                    a[gate[2]]=1
            else:
                if((gate[0]=='S' or gate[0]=='T') and res[i-1][gate[1]]==1):
                    a[gate[1]]=1
                if(gate[0]=='CX' and res[i-1][gate[1]]==1):
                    a[gate[1]]=1
                if(gate[0]=='CX' and res[i-1][gate[2]]==1 and res[i-1][gate[1]]==1):
                    a[gate[2]]=1
        res.append(a)
    return res


def backwardsSweep(layers):
    res=[]
    depth=len(layers)
    
    for i in range(depth-1):
        a=[0 for i in range(amp_final.n_qbits)]
        for gate in (layers[depth-1-i]):
            if(i==0):
                if(gate[0]=='S' or gate[0]=='T'):
                    a[gate[1]]=1
                if(gate[0]=='CX'): 
                    a[gate[1]]=1
                    a[gate[2]]=1
            else:
                if((gate[0]=='S' or gate[0]=='T') and res[i-1][gate[1]]==1):
                    a[gate[1]]=1
                if(gate[0]=='CX' and res[i-1][gate[1]]==1):
                    a[gate[1]]=1
                if(gate[0]=='CX' and res[i-1][gate[2]]==1 and res[i-1][gate[1]]==1):
                    a[gate[2]]=1
        res.append(a)
    res.reverse()
    return res


def nullAmpCheckAux(state, layer, nxtStateColor):
    stateAux = copy.deepcopy(state)
    for gate in layer:
        if(nxtStateColor[gate[1]]==1):
            if(gate[0] == 'CX'):
                if(state[gate[1]]==1 and nxtStateColor[gate[2]]==1):
                    stateAux[gate[2]] = abs(state[gate[2]]-1)
                elif(state[gate[1]]==0  and nxtStateColor[gate[2]]==1):
                    stateAux[gate[2]] = state[gate[2]]
                elif(nxtStateColor[gate[2]]==0):
                    stateAux[gate[2]]='red'
        elif(gate[0]=='CX'):
            stateAux[gate[2]]='red'
        else:
            stateAux[gate[1]]='red'
    return stateAux


# In[139]:


def nullAmpCheck(stateIn, layers, forwardSweep, backwardsSweep, stateOut):
    a=1
    statesAuxFor=[]
    statesAuxBack=[]
    depth=len(layers)
    #Check for inconsistencies
    for i in range(depth-1):
        if(i==0):
            stateAuxFor = nullAmpCheckAux(stateIn, layers[i], forwardSweep[i])
            statesAuxFor.append(stateAuxFor)
            stateAuxBack = nullAmpCheckAux(stateOut, layers[depth-1], backwardsSweep[depth-2])
            statesAuxBack.append(stateAuxBack)
        else:
            stateAuxFor=nullAmpCheckAux(stateAuxFor, layers[i], forwardSweep[i])
            statesAuxFor.append(stateAuxFor)
            stateAuxBack=nullAmpCheckAux(stateAuxBack, layers[depth-i-1],backwardsSweep[depth-2-i])
            statesAuxBack.append(stateAuxBack)
    statesAuxBack.reverse()
    for i in range(depth-1):
        for j in range(amp_final.n_qbits):
            if(statesAuxFor[i][j]!=statesAuxBack[i][j] and statesAuxFor[i][j]!='red' and statesAuxBack[i][j]!='red'):
                a=0
                break
        if(statesAuxFor[i][j]!=statesAuxBack[i][j] and statesAuxFor[i][j]!='red' and statesAuxBack[i][j]!='red'):
            break
    return (a, statesAuxFor, statesAuxBack)


def IMStates(stateIn, layers, statesAuxFor, statesAuxBack, stateOut):
    reds_index=[]
    imstates=[]
    IMStates.n_of_reds=0
    depth=len(layers)
    for i in range(depth-1):
        IMAux=[]
        for j in range(amp_final.n_qbits):
            if(statesAuxFor[i][j]!='red'):
                IMAux.append(statesAuxFor[i][j])
            elif(statesAuxBack[i][j]!='red'):
                IMAux.append(statesAuxBack[i][j])
            if(statesAuxFor[i][j]=='red' and statesAuxBack[i][j]=='red'):
                IMAux.append('red')
                IMStates.n_of_reds+=1
                reds_index.append((i,j))
        imstates.append(IMAux)
    IMStates.ratio_reds=(IMStates.n_of_reds/(amp_final.n_qbits*(depth-1)))*100
    return imstates, reds_index


def constantAmpAux(In, Out, layer):
    a = 1
    n = len(layer)
    for i in range(n):
        if(In[layer[i][1]] != 'red' and Out[layer[i][1]] != 'red'):
            if(layer[i][0] == 'CX'):
                if(In[layer[i][2]] != 'red' and Out[layer[i][2]] != 'red'):
                    a = a * CX((In[layer[i][1]], In[layer[i][2]]), (Out[layer[i][1]], Out[layer[i][2]]))
            elif(layer[i][0] == 'T'):
                a = a * T(In[layer[i][1]], Out[layer[i][1]])
            elif(layer[i][0] == 'S'):
                a = a * S(In[layer[i][1]], Out[layer[i][1]])
            elif(layer[i][0] == 'H'):
                a = a * H(In[layer[i][1]], Out[layer[i][1]])
    #print(a)
    return a


def constantAmp(all_states, layers):
    a=1
    depth=len(layers)
    for i in range(depth):
        a = a * constantAmpAux(all_states[i], all_states[i+1], layers[i])
    return a


def amp_final(stateIn, layers, stateOut, n_qbits):
    
    depth=len(layers)
    amp_final.n_qbits=n_qbits
    fs=forwardSweep(layers)
    bs=backwardsSweep(layers)
    a, statesFor, statesBack = nullAmpCheck(stateIn, layers, fs, bs, stateOut)
    
    if(a==0):
        return 0
    imstates, reds = IMStates(stateIn, layers, statesFor, statesBack, stateOut)
    all_states=copy.deepcopy(imstates)
    all_states.insert(0, stateIn)
    all_states.append(stateOut)
    if(reds==[]):
        return abs(constantAmp(all_states, layers))**2
    
    gates_to_iterate=[]
    for i in range(depth):
        gates_to_iterate.append([])
        
    for red in (reds):
        for j in range(red[0], red[0]+2):
            for gate in (layers[j]):
                if(gate[0] == 'CX'):
                    if(gate[1] == red[1] or gate[2] == red[1]):
                        if(gate not in gates_to_iterate[j]):
                            gates_to_iterate[j].append(gate)
                elif(gate[1] == red[1]):
                    if(gate not in gates_to_iterate[j]):
                        gates_to_iterate[j].append(gate)
    n=len(reds)
    def it(max):
        for i in range(1<<max):
            s=bin(i)[2:]
            s='0'*(max-len(s))+s
            s = list(map(int,list(s)))
            yield s
            
    amp_final=0
    b=constantAmp(all_states, layers)
    for i in it(n):
        a=1
        for j in range(len(reds)):
            all_states[reds[j][0]+1][reds[j][1]]=i[j]
        for k in range(len(gates_to_iterate)):
            for l in range(len(gates_to_iterate[k])):
                if(gates_to_iterate[k][l][0]=='CX'):
                    a = a*CX((all_states[k][gates_to_iterate[k][l][1]],all_states[k][gates_to_iterate[k][l][2]]),\
                             (all_states[k+1][gates_to_iterate[k][l][1]],all_states[k+1][gates_to_iterate[k][l][2]]))                    
                elif(gates_to_iterate[k][l][0]=='S'):
                    a = a*S(all_states[k][gates_to_iterate[k][l][1]], all_states[k+1][gates_to_iterate[k][l][1]])
                elif(gates_to_iterate[k][l][0]=='T'):
                    a = a*T(all_states[k][gates_to_iterate[k][l][1]], all_states[k+1][gates_to_iterate[k][l][1]])
                elif(gates_to_iterate[k][l][0]=='H'):
                    a = a*H(all_states[k][gates_to_iterate[k][l][1]], all_states[k+1][gates_to_iterate[k][l][1]])
                
        a = a * b
        amp_final = amp_final + a
    return amp_final
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


#THE FOLLOWING FUNCTIONS ARE USED TO CALCULATE THE AMPLITUDE FOR EACH OF THE GATES USED, GIVEN THEIR INPUT AND OUTPUT STATES
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
    
def I(In, Out):
    if(In == Out):
        return 1
    else:
        return 0


#SWEEP OF THE CIRCUIT IN INPUT -> OUTPUT DIRECTION, MARKING RED (0) ALL POSITIONS WHERE THE STATE IS NON CLASSICAL AND GREEN (1) OTHERWISE. TAKES AS ARGUMENT THE LIST WITH ALL LAYERS OF THE CIRCUIT

def forwardSweepPB(layers):
    #res IS THE VARIABLE WHERE THE RESULT OF THE SWEEP WILL BE STORED
    #LIST OF LISTS WHERE EACH INNER LIST CORRESPONDS TO AN INTERMEDIATE STATE OF ALL QBITS
    res=[]
    depth = len(layers)
    #GO THROUGH INTERMEDIATE STATE AFTER INTERMEDIATE STATE IN THE INPUT -> OUTPUT DIRECTION
    for i in range(depth-1):
        #FOR EACH OF THESE, INITIALLY ALL POSITIONS ARE MARKED AS RED
        #THE VARIABLE res_aux TO STORE THE COLORS OF EACH INTERMEDIATE STATE
        res_aux=[0 for i in range(amp_final_PB.n_qbits)]
        #NEXT, IT IS ANALYZED CASE BY CASE WHICH ONES CAN BE MARKED AS GREEN
        for gate in (layers[i]):
            if(i==0):
                if(gate[0]=='S' or gate[0]=='T' or gate[0]=='I'):
                    res_aux[gate[1]]=1
                if(gate[0]=='CX'): 
                    res_aux[gate[1]]=1
                    res_aux[gate[2]]=1
            else:
                if((gate[0]=='S' or gate[0]=='T' or gate[0]=='I') and res[i-1][gate[1]]==1):
                    res_aux[gate[1]]=1
                if(gate[0]=='CX' and res[i-1][gate[1]]==1):
                    res_aux[gate[1]]=1
                if(gate[0]=='CX' and res[i-1][gate[2]]==1 and res[i-1][gate[1]]==1):
                    res_aux[gate[2]]=1
        #FINALLY, WHEN ALL POSITIONS IN AN INTERMEDIATE STATE ARE MARKED, WE ADD res_aux TO THE VARIABLE res AND MOVE ON TO THE NEXT INTERMEDIATE STATE
        res.append(res_aux)
    return res


#SWEEPING OF THE CIRCUIT IN THE OUTPUT -> INPUT DIRECTION, MARKING RED (0) ALL POSITIONS WHERE THE STATE IS NON CLASSICAL AND GREEN (1) OTHERWISE. SIMILAR TO THE forwardSweepPB FUNCTION, BUT NOW STARTING FROM OUTPUT.

def backwardsSweepPB(layers):
    res=[]
    depth=len(layers)
    for i in range(depth-1):
        res_aux=[0 for i in range(amp_final_PB.n_qbits)]
        for gate in (layers[depth-1-i]):
            if(i==0):
                if(gate[0]=='S' or gate[0]=='T' or gate[0]=='I'):
                    res_aux[gate[1]]=1
                if(gate[0]=='CX'): 
                    res_aux[gate[1]]=1
                    res_aux[gate[2]]=1
            else:
                if((gate[0]=='S' or gate[0]=='T' or gate[0]=='I') and res[i-1][gate[1]]==1):
                    res_aux[gate[1]]=1
                if(gate[0]=='CX' and res[i-1][gate[1]]==1):
                    res_aux[gate[1]]=1
                if(gate[0]=='CX' and res[i-1][gate[2]]==1 and res[i-1][gate[1]]==1):
                    res_aux[gate[2]]=1
        res.append(res_aux)
    res.reverse()
    return res


#The green_and_red_coloring FUNCTION RECEIVES AS ARGUMENTS THE LIST WITH ALL THE CIRCUIT LAYERS AND THE RESULTS OF THE SWEEPS. RETURNS THE RESULT OF THE LOGICAL 'OR' OPERATION BETWEEN THE RESULTS OF THE SWEEPS, WHICH CAN BE SEEN AS THE COMPLETE GREEN/RED COLORING OF ALL INTERMEDIATE STATES. THIS RESULT WILL BE USEFUL FOR THE NEXT FUNCTION THAT COMPUTES THE PINK/BLUE COLORING.

def green_and_red_coloring(layers, fs, bs):
    #res IS THE VARIABLE WHERE THE RESULT OF THE 'OR' BETWEEN THE RESULTS OF THE SWEEPS WILL BE STORED
    #LIST OF LISTS WHERE EACH INNER LIST CORRESPONDS TO AN INTERMEDIATE STATE OF ALL QBITS
    res=[]
    green_and_red_coloring.n_of_reds=0
    for i in range(len(fs)):
        res1=[]
        for color_fs, color_bs in (zip(fs[i], bs[i])):
            if(color_fs==1 or color_bs==1):
                res1.append(1)
            else:
                res1.append(0)
                green_and_red_coloring.n_of_reds+=1
        res.append(res1)
    return res


#THE FUNCTION pink_and_blue_coloring_for RECEIVES AS ARGUMENTS THE LIST WITH ALL THE CIRCUIT LAYERS AND THE RESULTS OF green_and_red_coloring. RETURNS PINK/BLUE COLORING IN THE INPUT -> OUTPUT DIRECTION.

def pink_and_blue_coloring_for(layers, green_red_color):
    pink_and_blue_coloring_for.n_of_pinks=0
    #res IS THE VARIABLE WHERE THE RESULT OF THE COLORING WILL BE STORED
    #LIST OF LISTS WHERE EACH INNER LIST CORRESPONDS TO AN INTERMEDIATE STATE OF ALL QBITS
    res=copy.deepcopy(green_red_color)
    depth=len(layers)
    for i in range(depth-1):
        for gate in layers[i]:
            #MARKS IN PINK (2) ALL POSITIONS IMMEDIATELY FOLLOWING AN H GATE, MARKED PREVIOUSLY IN RED.
            if(gate[0]=='H' and res[i][gate[1]]==0):
                res[i][gate[1]]=2
                pink_and_blue_coloring_for.n_of_pinks+=1
    for i in range(depth-1):
        for j in range(len(res[i])):
            #MARKS IN BLUE (3) THE REMAINING PREVIOUSLY RED POSITIONS.
            if (res[i][j]==0):
                res[i][j]=3
                
    return res


#THE FOLLOWING TWO FUNCTIONS ARE USED TO CHECK FOR INCONSISTENCIES BETWEEN THE INTERMEDIATE STATES CALCULATED USING THE SWEEPS.
#USING THE RESULT OF THE forwardSweepPB AND backwardsSweepPB FUNCTIONS, TWO VERSIONS OF THE INTERMEDIATE STATES FOR EACH QUBIT ARE CALCULATED.
#The nullAmpCheckAux FUNCTION RECEIVES A STATE, THE LAYER OF GATES AFTER THIS STATE AND THE COLORS OF THE POSITIONS CORRESPONDING TO THE STATE IMMEDIATELY FOLLOWING THE LAYER OF GATES, PROVIDED BY THE forwardSweepPB AND backwardsSweepPB FUNCTIONS. THIS FUNCTION RETURNS THE STATE AFTER THE LAYER OF GATES.

def nullAmpCheckAux(state, layer, nxtStateColor):
    #THE RESULT WILL BE STORED IN THE stateAux VARIABLE. THIS VARIABLE WILL BE A LIST WHOSE ELEMENTS WILL BE THE STATES OF EACH QUBIT IN A GIVEN INTERMEDIATE STATE. 
    stateAux = copy.deepcopy(state)
    for gate in layer:
        if(nxtStateColor[gate[1]]==1):
            if(gate[0] == 'CX'):
                if(state[gate[1]]==1 and nxtStateColor[gate[2]]==1):
                    stateAux[gate[2]] = abs(state[gate[2]]-1)
                elif(state[gate[1]]==0  and nxtStateColor[gate[2]]==1):
                    stateAux[gate[2]] = state[gate[2]]
                #THE QUBITS FOR WHICH THE STATE IS NON CLASSICAL ARE MARKED AS 'red'.
                elif(nxtStateColor[gate[2]]==0):
                    stateAux[gate[2]]='red'
        elif(gate[0]=='CX'):
            stateAux[gate[2]]='red'
        else:
            stateAux[gate[1]]='red'
    return stateAux


#THE FUNCTION nullAmpCheckPB RECEIVES THE INPUT STATE, THE LIST WITH ALL THE CIRCUIT LAYERS, THE SWEEP COLORINGS AND THE OUTPUT STATE. RETURNS A BINARY VARIABLE a, WHICH HAS VALUE 0 IF INCONSISTENCIES ARE DETECTED AND VALUE 1 OTHERWISE, AND THE AUXILIARY INTERMEDIATE STATES CORRESPONDING TO THE FORWARD AND BACKWARD SWEEPS. THESE WILL BE USED IN A LATER FUNCTION TO CALCULATE THE COMPLETE INTERMEDIATE STATES.

def nullAmpCheckPB(stateIn, layers, forwardSweep, backwardsSweep, stateOut):
    #THE VARIABLES statesAuxFor AND statesAuxBack ARE LISTS OF LISTS IN WHICH statesAuxFor/Back[i][j] CORRESPONDS TO THE STATE OF QUBIT j IN THE INTERMEDIATE STATE AFTER LAYER i AND MAY HAVE A VALUE OF 0,1, OR 'red'.
    a=1
    statesAuxFor=[]
    statesAuxBack=[]
    depth=len(layers)
    #USING THE nullAmpCheckAux FUNCTION CALCULATE THE TWO VERSIONS OF THE INTERMEDIATE STATES THAT WILL BE STORED IN THE statesAuxFor/Back VARIABLES
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
    #THEN LOOK AT THE TWO LISTS statesAuxFor/Back COMPARING THEIR VALUES
    for i in range(depth-1):
        for j in range(amp_final_PB.n_qbits):
            #IF IN THESE TWO VERSIONS THE SAME QUBIT, IN THE SAME INTERMEDIATE STATE, HAS TWO DIFFERENT DETERMINISTIC VALUES (ONE OF THEM 0 AND THE OTHER 1), VAR a -> 0, MEANING THAT AN INCONSISTENCY HAS BEEN DETECTED. AS SOON AS THE INCONSISTENCY IS DETECTED, THE FUNCTION BREAKS AND RETURNS THE VARIABLES. 
            if(statesAuxFor[i][j]!=statesAuxBack[i][j] and statesAuxFor[i][j]!='red' and statesAuxBack[i][j]!='red'):
                a=0
                break
        if(statesAuxFor[i][j]!=statesAuxBack[i][j] and statesAuxFor[i][j]!='red' and statesAuxBack[i][j]!='red'):
            break
    #IF NO INCONSISTENCY IS DETECTED THE FUNCTION RETURNS THE VARIABLES.
    return (a, statesAuxFor, statesAuxBack)


#The IMStatesPB FUNCTION CALCULATES THE COMPLETE INTERMEDIATE STATES. RECEIVES AS ARGUMENTS THE INPUT STATE, THE LIST OF ALL CIRCUIT LAYERS, THE VARIABLES statesAuxFor/Back FROM THE nullAmpCheckPB FUNCTION, THE RESULT OF PINK/BLUE COLORATION, AND THE OUTPUT STATE. THIS FUNCTION RETURNS THE COMPLETE INTERMEDIATE STATES.

def IMStatesPB(stateIn, layers, statesAuxFor, statesAuxBack, pink_and_blue_coloring, stateOut):
    #THE VARIABLE imstates IS A LIST OF LISTS IN WHICH imstates[i][j] CORRESPONDS TO THE STATE OF QUBIT j IN THE INTERMEDIATE STATE AFTER THE i LAYER AND CAN HAVE A VALUE OF 0,1, 'pink', OR 'blue'.
    imstates=[]
    IMStatesPB.n_of_pinks=0
    depth=len(layers)
    #TO COMPOSE THE LIST imstates, GO THROUGH THE LISTS statesAuxFor/Back. WHEN AT LEAST ONE OF THE statesAuxFor/Back LISTS HAS A VALUE OTHER THAN 'red' FOR THE SAME QUBIT AND FOR THE SAME INTERMEDIATE STATE, THIS VALUE (0 OR 1) WILL BE ADDED. WHEN THE SAME QUBIT, FOR THE SAME INTERMEDIATE STATE, HAS THE VALUE 'red' IN BOTH LISTS, AND THE SAME QUBIT IS MARKED AS PINK IN THE PINK/BLUE COLORATION, 'pink' IS ADDED TO imstates. WHEN THE SAME QUBIT, FOR THE SAME INTERMEDIATE STATE, HAS THE VALUE 'red' IN BOTH LISTS, BUT THE SAME QUBIT IS NOT MARKED AS PINK IN THE PINK/BLUE COLORATION, 'blue' IS ADDED TO imstates. 
    for i in range(depth-1):
        IMAux=[]
        for j in range(amp_final_PB.n_qbits):
            if(statesAuxFor[i][j]!='red'):
                IMAux.append(statesAuxFor[i][j])
            elif(statesAuxBack[i][j]!='red'):
                IMAux.append(statesAuxBack[i][j])
            if(statesAuxFor[i][j]=='red' and statesAuxBack[i][j]=='red'):
                if(pink_and_blue_coloring[i][j]==2):
                    IMAux.append('pink')
                    IMStatesPB.n_of_pinks+=1
                else:
                    IMAux.append('blue')
        imstates.append(IMAux)
    IMStatesPB.ratio_pinks=(IMStatesPB.n_of_pinks/(amp_final_PB.n_qbits*(depth-1)))*100
    return imstates


#THE FOLLOWING TWO FUNCTIONS ARE USED TO CALCULATE THE AMPLITUDE GIVEN THE INPUT AND OUTPUT STATES AND THE LAYER OF GATES BETWEEN THEM, IGNORING THE AMPLITUDES COMING FROM GATES WITH NON CLASSICAL INPUT AND/OR OUTPUT.
#The constantAmpAuxPB FUNCTION RECEIVES AS ARGUMENTS THE INPUT AND OUTPUT STATES AND A LAYER OF GATES BETWEEN THEM.

def constantAmpAuxPB(In, Out, layer):
    a = 1
    n = len(layer)
    for i in range(n):
        if(In[layer[i][1]] != 'pink' and In[layer[i][1]] != 'blue' and\
           Out[layer[i][1]] != 'pink' and Out[layer[i][1]] != 'blue'):
            if(layer[i][0] == 'CX'):
                if(In[layer[i][2]] != 'pink' and In[layer[i][2]] != 'blue' and\
                   Out[layer[i][2]] != 'pink' and Out[layer[i][2]] != 'blue'):
                    a = a * CX((In[layer[i][1]], In[layer[i][2]]), (Out[layer[i][1]], Out[layer[i][2]]))
            elif(layer[i][0] == 'T'):
                a = a * T(In[layer[i][1]], Out[layer[i][1]])
            elif(layer[i][0] == 'S'):
                a = a * S(In[layer[i][1]], Out[layer[i][1]])
            elif(layer[i][0] == 'H'):
                a = a * H(In[layer[i][1]], Out[layer[i][1]])
            elif(layer[i][0] == 'I'):
                a = a * I(In[layer[i][1]], Out[layer[i][1]])
    return a


#The constantAmpPB FUNCTION RECEIVES AS ARGUMENTS ALL STATES (INPUT AND OUTPUT INCLUDED) AND THE VARIOUS LAYERS OF GATES. APPLIES THE FUNCTION constantAmpAux SUCCESSIVELY TO EACH LAYER. AT THE END THE TOTAL AMPLITUDE IS OBTAINED AND RETURNED.

def constantAmpPB(all_states, layers):
    a=1
    depth=len(layers)
    for i in range(depth):
        a = a * constantAmpAuxPB(all_states[i], all_states[i+1], layers[i])
    return a

#it IS A PYTHON GENERATOR THAT GENERATES ALL BINARY STRING COMBINATIONS WITH max BITS. IT WILL BE USEFUL FOR ITERATING OVER ALL INTERMEDIATE STATES TO PINK
def it(max):
        for i in range(1<<max):
            s=bin(i)[2:]
            s='0'*(max-len(s))+s
            s = list(map(int,list(s)))
            yield s
            
#THE FUNCTION amp_final_PB IS THE FUNCTION THAT GIVEN THE INPUT AND OUTPUT STATES AND THE LAYERS OF GATES, CALCULATES THE TOTAL AMPLITUDE FOR A GIVEN CIRCUIT.              

def amp_final_PB(stateIn, layers, stateOut, n_qbits):
    
    depth=len(layers)
    amp_final_PB.n_qbits=n_qbits
    
    #PRE-PROCESSING (GREEN/RED COLORING, CHECK FOR FORWARD AND BACKWARD COLOR CONFLICTS)
    fs=forwardSweepPB(layers)
    bs=backwardsSweepPB(layers)
    a, statesFor, statesBack = nullAmpCheckPB(stateIn, layers, fs, bs, stateOut)
    
    if(a==0): #MEANS A CONFLICT WAS DETECTED -> AMPLITUDE = 0
        return 0
    
    #CALCULATION OF GREEN/RED AND PINK/BLUE COLORING
    green_red_coloring = green_and_red_coloring(layers, fs, bs)
    
    #IF N_OF_REDS = 0 -> IMMEDIATE AMPLITUDE CALCULATION
    if(green_and_red_coloring.n_of_reds==0):
        imstates = IMStatesPB(stateIn, layers, statesFor, statesBack, green_red_coloring, stateOut) 
        all_states=copy.deepcopy(imstates)
        all_states.insert(0, stateIn)
        all_states.append(stateOut)
        return constantAmpPB(all_states, layers)
    pink_blue_coloring = pink_and_blue_coloring_for(layers, green_red_coloring)
    
    #CALCULATION OF INTERMEDIATE STATES
    imstates = IMStatesPB(stateIn, layers, statesFor, statesBack, pink_blue_coloring, stateOut) 
    #CREATION OF THE all_states VARIABLE THAT JOINS THE INPUT AND OUTPUT STATES TO THE INTERMEDIATE STATES
    all_states=copy.deepcopy(imstates)
    all_states.insert(0, stateIn)
    all_states.append(stateOut)
    
    #n_pinks IS THE NUMBER OF PINK STATES. THIS VALUE WILL BE THE NUMBER OF BITS WITH WHICH TO GENERATE ALL THE BINARY COMBINATIONS TO ITERATE.
    n_pinks=IMStatesPB.n_of_pinks
    
    #amp_final IS THE VARIABLE THAT WILL CONTAIN THE TOTAL FINAL AMPLITUDE
    amp_final=0
    #VARIABLE b STORES THE TOTAL AMPLITUDE OF ALL GATES THAT DO NOT HAVE PINK OR BLUE INPUTS AND/OR OUTPUTS
    b=constantAmpPB(all_states, layers)
    
    #ITERATION ON ALL BINARY COMBINATIONS OF n_reds_middle NUMBER OF BITS
    #FOR EACH BINARY COMBINATION
    for i in it(n_pinks):
        c=0
        a=1
        for j in range(depth):
            for gate in layers[j]:
                #BLUE STATES ARE RESOLVED AFTER THE CNOT GATES
                if (gate[0] == 'CX' and (all_states[j+1][gate[1]] == 'blue' or all_states[j+1][gate[2]] == 'blue')):
                    if(all_states[j+1][gate[1]] == 'blue'):
                        all_states[j+1][gate[1]] = all_states[j][gate[1]]
                    if(all_states[j+1][gate[2]] == 'blue'):
                        all_states[j+1][gate[2]] = abs(all_states[j][gate[2]]-all_states[j][gate[1]])
                    #CALCULATE THE AMPLITUDE OF THESE GATES
                    a = a * CX((all_states[j][gate[1]], all_states[j][gate[2]]),\
                               (all_states[j+1][gate[1]], all_states[j+1][gate[2]]))
                elif(gate[0] == 'CX' and j>0 and (pink_blue_coloring[j-1][gate[1]] != 1 or pink_blue_coloring[j-1][gate[2]] != 1)):
                    #CALCULATE THE AMPLITUDE OF THE CNOT GATES WITH PREVIOUSLY NON CLASSICAL INPUT, BUT SOLVED IN THE MEANTIME
                    a = a * CX((all_states[j][gate[1]], all_states[j][gate[2]]),\
                               (all_states[j+1][gate[1]], all_states[j+1][gate[2]]))   
                #REPLACE THE PINK VALUES WITH THE VALUES OF THE CORRESPONDING BITS OF THE COMBINATION
                elif(gate[0] == 'H' and all_states[j+1][gate[1]] == 'pink'):
                    all_states[j+1][gate[1]] = i[c]
                    c += 1
                    #CALCULATE THE AMPLITUDE OF THE H GATES THAT HAD PINK OUTPUT STATES 
                    a = a * H(all_states[j][gate[1]], all_states[j+1][gate[1]])
                elif(gate[0] == 'H' and j>0 and pink_blue_coloring[j-1][gate[1]] != 1):
                    #CALCULATE THE AMPLITUDE OF THE H GATES WITH PREVIOUSLY NON CLASSICAL INPUT, BUT SOLVED IN THE MEANTIME
                    a = a * H(all_states[j][gate[1]], all_states[j+1][gate[1]])
                #THE BLUE STATES AFTER THE 'T', 'S' AND 'I' GATES ARE SOLVED. CALCULATE THE AMPLITUDES OF THESE GATES
                elif(gate[0] == 'T' and all_states[j+1][gate[1]] == 'blue'):
                    all_states[j+1][gate[1]] = all_states[j][gate[1]]
                    a = a * T(all_states[j][gate[1]], all_states[j+1][gate[1]])
                elif(gate[0] == 'S' and all_states[j+1][gate[1]] == 'blue'):
                    all_states[j+1][gate[1]] = all_states[j][gate[1]]
                    a = a * S(all_states[j][gate[1]], all_states[j+1][gate[1]])
                elif(gate[0] == 'I' and all_states[j+1][gate[1]] == 'blue'):
                    all_states[j+1][gate[1]] = all_states[j][gate[1]]
                    a = a * I(all_states[j][gate[1]], all_states[j+1][gate[1]])
        all_states=copy.deepcopy(imstates)
        all_states.insert(0, stateIn)
        all_states.append(stateOut)
        #STORE THE VALUE OF THE CALCULATED AMPLITUDES, MULTIPLICATE THEM INTO THE CUMULATIVE VARIABLE a AND MULTIPLICATE THEM BY THE AMPLITUDE STORED IN VARIABLE B
        a = a * b
        #FINALLY ADD THIS RESULT TO THE VARIABLE amp_final.
        amp_final = amp_final + a
    return amp_final

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

#AS FUNÇÕES SEGUINTES SERVEM PARA O CÁLCULO DA AMPLITUDE PARA CADA UMA DAS GATES UTILIZADAS, DADOS O ESTADO À ENTRADA E À SAÍDA DAS MESMAS
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




#VARRIMENTO NO SENTIDO INPUT -> OUTPUT DO CIRCUITO, MARCANDO A VERMELHO (0) TODAS AS POSIÇÕES EM QUE O ESTADO NÃO É DETERMINÍSTICO E A VERDE (1) CASO CONTRÁRIO. RECEBE COMO ARGUMENTO A LISTA COM TODAS AS CAMADAS DO CIRCUITO

def forwardSweepRec(layers):
    #res É A VARIÁVEL ONDE SE ARMAZENARÁ O RESULTADO DO VARRIMENTO
    #lISTA DE LISTAS EM QUE CADA LISTA INTERIOR CORRESPONDE A UM ESTADO INTERMÉDIO DE TODOS OS QBITS
    res=[]
    depth=len(layers)
    #PERCORRE-SE ESTADO INTERMÉDIO APÓS ESTADO INTERMÉDIO NO SENTIDO INPUT -> OUTPUT
    for i in range(depth-1):
        #PARA CADA UM DESTES, MARCA-SE INICIALMENTE TODAS AS POSIÇÕES COMO VERMELHAS
        #UTILIZA-SE A VARIÁVEL res_aux PARA ARMAZENAR AS CORES DE CADA ESTADO INTERMÉDIO
        res_aux=[0 for i in range(amp_final_recursive.n_qbits)]
        #DE SEGUIDA, ANALISA-SE CASO A CASO QUAIS PODERÃO SER MARCADAS COMO VERDES
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
        #POR FIM, QUANDO TODOS AS POSIÇÕES NUM ESTADO INTERMÉDIO SÃO MARCADAS, ADICIONA-SE res_aux À VARIÁVEL res
        res.append(res_aux)
    return res


#VARRIMENTO NO SENTIDO OUTPUT -> INPUT DO CIRCUITO, MARCANDO A VERMELHO (0) TODAS AS POSIÇÕES EM QUE O ESTADO NÃO É DETERMINÍSTICO E A VERDE (1) CASO CONTRÁRIO. SEMELHANTE À FUNÇÃO forwardSweepRec, MAS AGORA COMEÇANDO PELO OUTPUT

def backwardsSweepRec(layers):
    res=[]
    depth=len(layers)
    for i in range(depth-1):
        res_aux=[0 for i in range(amp_final_recursive.n_qbits)]
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


#AS DUAS FUNÇÕES SEGUINTES SERVEM PARA VERIFICAR SE HÁ INCONSISTÊNCIAS ENTRE OS ESTADOS INTERMÉDIOS CALCULADOS COM RECURSO AOS VARRIMENTOS
#USANDO O RESULTADO DAS FUNÇÕES forwardSweepRec E backwardsSweepRec SÃO CALCULADAS DUAS VERSÕES DOS ESTADOS INTERMÉDIOS PARA CADA QUBIT.
#A FUNÇÃO nullAmpCheckAux RECEBE UM ESTADO (QUE PODE SER O ESTADO DE INPUT (OU OUTPUT, QUANDO SE ESTÁ A CALCULAR A VERSÃO DOS ESTADOS INTERMÉDIO CORRESPONDENTE AO VARRIMENTO DO OUTPUT PARA O INPUT) OU QUALQUER UM DOS INTERMÉDIOS), A CAMADA DE PORTAS À FRENTE DESSE ESTADO E AS CORES DAS POSIÇÕES CORRESPONDENTES AO ESTADO IMEDIATAMENTE A SEGUIR À CAMADA DE PORTAS, PROVENIENTES DAS FUNÇÕES forwardSweepRec E backwardsSweepRec. ESTA FUNÇÃO RETORNA O ESTADO A SEGUIR À CAMADA DE PORTAS.

def nullAmpCheckAux(state, layer, nxtStateColor):
    #O RESULTADO SERÁ ARMAZENADO NA VARIÁVEL stateAux. ESTA VARIÁVEL SERÁ UMA LISTA CUJOS ELEMENTOS SERÃO OS ESTADOS DE CADA QUBIT EM DETERMINADO ESTADO INTERMÉDIO. INICIALMENTE, COPIA-SE O VALOR DO ESTADO ANTES DA CAMADA PARA A VARIÁVEL stateAux, VISTO QUE MUITAS DAS PORTAS MAPEIAM 0 -> 0 E 1 -> 1.
    stateAux = copy.deepcopy(state)
    for gate in layer:
        #DE SEGUIDA RESOLVE-SE O MAPEAMENTO DAS PORTAS NÃO TRIVIAIS (NESTE CASO CNOT)
        if(nxtStateColor[gate[1]]==1):
            if(gate[0] == 'CX'):
                if(state[gate[1]]==1 and nxtStateColor[gate[2]]==1):
                    stateAux[gate[2]] = abs(state[gate[2]]-1)
                elif(state[gate[1]]==0  and nxtStateColor[gate[2]]==1):
                    stateAux[gate[2]] = state[gate[2]]
                #OS QUBITS EM QUE O ESTADO É NÃO DETERMINÍSTICO, MARCAM-SE COM 'red'
                elif(nxtStateColor[gate[2]]==0):
                    stateAux[gate[2]]='red'
        elif(gate[0]=='CX'):
            stateAux[gate[2]]='red'
        else:
            stateAux[gate[1]]='red'
    return stateAux


#A FUNÇÃO nullAmpCheckRec RECEBE O ESTADO DE INPUT, A LISTA COM TODAS AS CAMADAS DO CIRCUITO, AS COLORAÇÕES DOS VARRIMENTOS E O ESTADO DE OUTPUT. RETORNA UMA VARIÁVEL BINÁRIA a, QUE TEM VALOR ZERO SE FOREM DETETADAS INCONSISTÊNCIAS E VALOR UM CASO CONTRÁRIO, E OS ESTADOS INTERMÉDIOS AUXILIARES CORRESPONDENTES AOS VARRIMENTOS PARA A FRENTE E PARA TRÁS. ESTES SERÃO USADOS NUMA FUNÇÃO POSTERIOR PARA CALCULAR OS ESTADOS INTERMÉDIOS COMPLETOS.

def nullAmpCheckRec(stateIn, layers, forwardSweep, backwardsSweep, stateOut):
    #INICIALIZAM-SE AS VARIÁVEIS. AS VARIÁVEIS statesAuxFor E statesAuxBack SÃO LISTAS DE LISTAS EM QUE statesAuxFor/Back[i][j] CORRESPONDE AO ESTADO DO QUBIT j NO ESTADO INTERMÉDIO APÓS A CAMADA i E PODE TER COMO VALOR 0,1, OU 'red'.
    a=1
    statesAuxFor=[]
    statesAuxBack=[]
    depth=len(layers)
    #COM RECURSO À FUNÇÃO nullAmpCheckAux CALCULAM-SE AS DUAS VERSÕES DOS ESTADOS INTERMÉDIOS QUE FICARÃO ARMAZENADAS NAS VARIÁVEIS statesAuxFor/Back
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
    #DE SEGUIDA PERCORREM-SE AS DUAS LISTAS statesAuxFor/Back COMPARANDO OS SEUS VALORES
    for i in range(depth-1):
        for j in range(amp_final_recursive.n_qbits):
            #SE NESTAS DUAS VERSÕES O MESMO QUBIT, NO MESMO ESTADO INTERMÉDIO, TIVER DOIS VALORES DETERMINÍSTICOS DIFERENTES (UM DELES 0 E O OUTRO 1) A VARIÁVEL a PASSA A ZERO, SIGNIFICANDO QUE FOI DETETADA UMA INCONSISTÊNCIA. MAL É DETETADA A INCONSISTÊNCIA, A FUNÇÃO PÁRA E RETORNA AS VARIÁVEIS. 
            if(statesAuxFor[i][j]!=statesAuxBack[i][j] and statesAuxFor[i][j]!='red' and statesAuxBack[i][j]!='red'):
                a=0
                break
        if(statesAuxFor[i][j]!=statesAuxBack[i][j] and statesAuxFor[i][j]!='red' and statesAuxBack[i][j]!='red'):
            break
    #CASO NÃO SEJA DETETADA NENHUMA INCONSISTÊNCIA A FUNÇÃO RETORNA AS VARIÁVEIS.
    return (a, statesAuxFor, statesAuxBack)



#A FUNÇÃO IMStatesRec CALCULA OS ESTADOS INTERMÉDIOS COMPLETOS. RECEBE COMO ARGUMENTOS O ESTADO DE INPUT, A LISTA COM TODAS AS CAMADAS DO CIRCUITO, AS VARIÁVEIS statesAuxFor/Back PROVENIENTES DA FUNÇÃO nullAmpCheckRec E O ESTADO DE OUTPUT. ESTA FUNÇÃO DEVOLVE OS ESTADOS INTERMÉDIOS COMPLETOS E OS ÍNDICES DAS POSIÇÕES A VERMELHO, QUE SERÃO ÚTEIS PARA CALCULAR A AMPLITUDE TOTAL.

def IMStatesRec(stateIn, layers, statesAuxFor, statesAuxBack, stateOut):
    #INICIALIZAM-SE AS VARIÁVEIS. reds_index[i][j] É UMA LISTA CUJOS ELEMENTOS SÃO LISTAS (UMA POR CADA ESTADO INTERMÉDIO) COM O ÍNDICE ONDE HÁ VERMELHOS. POR EXEMPLO reds_index = [[], [2, 3], [1, 3]] SIGNIFICA QUE NO PRIMEIRO ESTADO INTERMÉDIO NÃO HÁ VERMELHOS, NO SEGUNDO HÁ UM VERMELHO NO 2º E 3º QUBITS, E NO TERCEIRO HÁ UM VERMELHO NO 1º E 3º QUBITS. A VARIÁVEL imstates É UMA LISTA DE LISTAS EM QUE imstates[i][j] CORRESPONDE AO ESTADO DO QUBIT j NO ESTADO INTERMÉDIO APÓS A CAMADA i E PODE TER COMO VALOR 0,1, OU 'red'.
    reds_index=[]
    imstates=[]
    #IMStatesRec.n_of_reds SERVE SOMENTE PARA TRATAMENTO DE DADOS
    IMStatesRec.n_of_reds=0
    depth=len(layers)
    #PARA COMPOR A LISTA imstates, PERCORREM-SE AS LISTAS statesAuxFor/Back. APENAS QUANDO O MESMO QUBIT, PARA O MESMO ESTADO INTERMÉDIO, TEM O VALOR 'red' EM AMBAS AS LISTAS É QUE SERÁ ADICIONADO UM 'red' A imstates. QUANDO, PELO MENOS, NUMA DAS LISTAS statesAuxFor/Back PARA O MESMO QUBIT, PARA O MESMO ESTADO INTERMÉDIO, EXISTIR UM VALOR DIFERENTE DE 'red' SERÁ ADICIONADO ESSE VALOR (0 OU 1)
    for i in range(depth-1):
        reds_indexAux=[]
        IMAux=[]
        for j in range(amp_final_recursive.n_qbits):
            if(statesAuxFor[i][j]!='red'):
                IMAux.append(statesAuxFor[i][j])
            elif(statesAuxBack[i][j]!='red'):
                IMAux.append(statesAuxBack[i][j])
            if(statesAuxFor[i][j]=='red' and statesAuxBack[i][j]=='red'):
                IMAux.append('red')
                #SEMPRE QUE FOR ADICIONADO UM 'red' AS VARIÁVEIS IMStatesRec.n_of_reds E reds_index SÃO ATUALIZADAS
                IMStatesRec.n_of_reds+=1
                reds_indexAux.append((j))
        imstates.append(IMAux)
        reds_index.append(reds_indexAux)
    return imstates, reds_index


#AS DUAS FUNÇÕES SEGUINTES SERVEM PARA CALCULAR A AMPLITUDE DADOS OS ESTADOS DE INPUT E OUTPUT E AS CAMADAS DE PORTAS ENTRE ESTES, IGNORANDO AS AMPLITUDES PROVENIENTES DAS PORTAS COM INPUT E/OU OUTPUT NÃO DETERMINÍSTICO.
#A FUNÇÃO constantAmpAux RECEBE COMO ARGUMENTOS OS ESTADOS DE INPUT E OUTPUT E UMA CAMADA DE PORTAS ENTRE ELES.

def constantAmpAux(In, Out, layer):
    #A VARIÁVEL a É A VARIÁVEL ONDE SERÁ ARMAZENADA A AMPLITUDE. É INICIALIZADA COM O VALOR 1.
    a = 1
    n = len(layer)
    #É PERCORRIDA A CAMADA E, CASO NÃO HAJA POSIÇÕES A VERMELHO NO INPUT OU OUTPUT DE CADA UMA DAS PORTAS, É CALCULADA A AMPLITUDE PARA CADA UMA DAS PORTAS, USANDO AS FUNÇÕES DEFINIDAS NO INÍCIO DESTE MÓDULO. CADA AMPLITUDE CALCULADA SERÁ MULTIPLICADA PELO VALOR DA AMPLITUDE CALCULADA ATÉ ENTÃO. NO FIM DEVOLVE-SE ESTA AMPLITUDE.
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
            elif(layer[i][0] == 'I'):
                a = a * I(In[layer[i][1]], Out[layer[i][1]])
    #print(a)
    return a

#A FUNÇÃO constantAmp RECEBE COMO ARGUMENTOS TODOS OS ESTADOS (INPUT E OUTPUT INCLUÍDOS) E AS VÁRIAS CAMADAS DE PORTAS. APLICA-SE A FUNÇÃO constantAmpAux SUCESSIVAMENTE A CADA CAMADA. NO FINAL OBTÉM-SE E RETORNA-SE A AMPLITUDE TOTAL. 
def constantAmp(all_states, layers):
    a=1
    depth=len(layers)
    for i in range(depth):
        a = a * constantAmpAux(all_states[i], all_states[i+1], layers[i])
    return a

#it É UM GERADOR DE PYTHON QUE GERA TODAS AS COMBINAÇÕES DE STRINGS BINÁRIAS COM max BITS. SERÁ ÚTIL PARA A ITERAÇÃO SOBRE TODOS OS ESTADOS INTERMÉDIOS A VERMELHO
def it(max):
    for i in range(1<<max):
        s=bin(i)[2:]
        s='0'*(max-len(s))+s
        s = list(map(int,list(s)))
        yield s

#A FUNÇÃO amp_final_recursive É A FUNÇÃO QUE DADOS OS ESTADOS DE INPUT E OUTPUT E AS CAMADAS DE PORTAS, CALCULA A AMPLITUDE TOTAL PARA UM DADO CIRCUITO. RECEBE COMO ARGUMENTOS AS CAMADAS DE PORTAS, O NÍVEL DE RECURSÃO ATUAL (SOMENTE PARA FAZER DEBUG), OS ESTADOS DE INPUT E OUTPUT E O NÚMERO DE QUBITS.     
def amp_final_recursive(layers, rec_level, stateIn, stateOut, n_qbits):
    amp_final_recursive.n_qbits=n_qbits
    depth = len(layers)
    #SE DEPTH=1 -> CÁLCULO IMEDIATO DA AMPLITUDE
    if(depth==1):
        return constantAmp([stateIn,stateOut], layers)
    #PRÉ PROCESSAMENTO (COLORAÇÃO VERDE/VERMELHO, VERIFICAÇÃO DE CONFLITOS DA COLORAÇÃO PARA A FRENTE E PARA TRÁS)
    fs=forwardSweepRec(layers)
    bs=backwardsSweepRec(layers)
    a, statesFor, statesBack = nullAmpCheckRec(stateIn, layers, fs, bs, stateOut)
    
    if(a==0): #SIGNIFICA QUE HOUVE CONFLITO -> AMPLITUDE = 0
        return 0
    
    #CÁLCULO DOS ESTADOS INTERMÉDIOS E ARMAZENAMENTO DOS ÍNDICES DAS POSIÇÕES A VERMELHO
    imstates, reds = IMStatesRec(stateIn, layers, statesFor, statesBack, stateOut)
    #CRIAÇÃO DA VARIÁVEL all_states QUE JUNTA OS ESTADOS DE INPUT E OUTPUT AOS ESTADOS INTERMÉDIOS
    all_states=copy.deepcopy(imstates)
    all_states.insert(0, stateIn)
    all_states.append(stateOut)
    
    #CASO NÃO HAJA POSIÇÕES A VERMELHO, DEVOLVE IMEDIATAMENTE A AMPLITUDE TOTAL DA FATIA DE CIRCUITO EM ANÁLISE NO NÍVEL DE RECURSÃO ATUAL
    reds_aux = [x for x in reds if x]
    if(not reds_aux):
        return constantAmp(all_states, layers)
    
    #SE DEPTH > 2: CONTAR O NÚMERO DE POSIÇÕES A VERMELHO NO ESTADO INTERMÉDIO CENTRAL (FLOOR CASO SEJA UM NÚMERO PAR DE ESTADOS   INTERMÉDIOS) - N_REDS_MIDDLE
    if(depth>2):
        n_reds_middle=len(reds[int((len(reds)-1)/2)])
        amp_final=0
        
        #ITERAR SOBRE TODAS AS COMBINAÇÕES BINÁRIAS DE n_reds_middle NÚMERO DE BITS
        for i in it(n_reds_middle):
            for j in range(n_reds_middle):
                all_states[int((len(all_states)-1)/2)][reds[int((len(reds)-1)/2)][j]]=i[j]
            #PARA CADA COMBINAÇÃO, FAZER A CHAMADA RECURSIVA (DIVIDIR EM 2 O CIRCUITO DO NÍVEL ATUAL DE RECURSÃO) 
            #amp_final É A VARIÁVEL A DEVOLVER NO FINAL
            amp_final = amp_final +\
                        amp_final_recursive(layers[0:int(depth/2)], rec_level+1, all_states[0],\
                                            all_states[int((len(all_states)-1)/2)], n_qbits)*\
                        amp_final_recursive(layers[int(depth/2):depth], rec_level+1, all_states[int((len(all_states)-1)/2)],\
                                            all_states[len(all_states)-1], n_qbits)
            
    #SE DEPTH <= 2: FAZER O CÁLCULO DA AMPLITUDE COM O MÉTODO GREEN/RED
    if(depth<=2):
        #A VARIÁVEL gates_to_iterate É UMA LISTA CUJOS ELEMENTOS SÃO LISTAS (UMA POR CAMADA DE PORTAS, POR ORDEM) COM AS PORTAS QUE TÊM PELO MENOS UM DOS SEUS INPUTS OU OUTPUTS COM ESTADO VERMELHO. POR EXEMPLO gates_to_iterate=[[['H', 0], ['H', 1]], [['H', 0], ['H', 1]]] SIGNIFICA QUE NA PRIMEIRA CAMADA A EXISTE UMA PORTA H APLICADA AO QUBIT 0 E OUTRA AO QUBIT 1 COM INPUT E/OU OUTPUT A VERMELHO, E ASSIM SUCESSIVAMENTE 
        gates_to_iterate=[]
        for i in range(depth):
            gates_to_iterate.append([])
        for i in range(len(reds)):
            for red in reds[i]:
                for j in range(i, i+2):
                    for gate in (layers[j]):
                        if(gate[0] == 'CX'):
                            if(gate[1] == red or gate[2] == red):
                                if(gate not in gates_to_iterate[j]):
                                    gates_to_iterate[j].append(gate)
                        elif(gate[1] == red):
                            if(gate not in gates_to_iterate[j]):
                                gates_to_iterate[j].append(gate)
        
        #n_reds É O NÚMERO DE ESTADOS A VERMELHO. ESTE VALOR SERÁ O NÚMERO DE BITS COM O QUAL SE IRÃO GERAR TODAS AS COMBINAÇÕES BINÁRIAS PARA FAZER A ITERAÇÃO.
        n_reds=0
        for layer in reds:
            for red in layer:
                if (red!=[]):
                    n_reds+=1
        #A VARIÁVEL b ARMAZENA A AMPLITUDE TOTAL DE TODAS AS GATES QUE NÃO TÊM INPUTS E/OU OUTPUTS A VERMELHO
        b=constantAmp(all_states, layers)
        #amp_final É A VARIÁVEL QUE VAI CONTER A AMPLITUDE TOTAL FINAL
        amp_final=0
        #PARA CADA UMA DAS COMBINAÇÕES BINÁRIAS
        for i in it(n_reds):
            c=0
            a=1
            for j in range(len(reds)):
                #SUBSTITUEM-SE OS VALORES A VERMELHO PELOS VALORES DOS BITS CORRESPONDENTES DA COMBINAÇÃO
                for red in reds[j]:
                    all_states[j+1][red]=i[c]
                    c+=1
            #E ITERA-SE CALCULANDO A AMPLITUDE POR TODAS AS GATES NA LISTA gates_to_iterate
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
                    elif(gates_to_iterate[k][l][0]=='I'):
                        a = a*I(all_states[k][gates_to_iterate[k][l][1]], all_states[k+1][gates_to_iterate[k][l][1]])  
            #GUARDA-SE O VALOR DAS AMPLITUDES CALCULADAS, MULTIPLICANDO-AS NA VARIÁVEL CUMULATIVA a E MULTIPLICA-SE PELA AMPLITUDE ARMAZENADA NA VARIÁVEL B
            a = a * b
            #FINALMENTE ADICIONA-SE À VARIÁVEL amp_final.
            amp_final = amp_final + a
    return amp_final
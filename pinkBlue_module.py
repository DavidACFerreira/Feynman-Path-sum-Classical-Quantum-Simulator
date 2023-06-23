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

def forwardSweepPB(layers):
    #res É A VARIÁVEL ONDE SE ARMAZENARÁ O RESULTADO DO VARRIMENTO
    #lISTA DE LISTAS EM QUE CADA LISTA INTERIOR CORRESPONDE A UM ESTADO INTERMÉDIO DE TODOS OS QBITS
    res=[]
    depth = len(layers)
    #PERCORRE-SE ESTADO INTERMÉDIO APÓS ESTADO INTERMÉDIO NO SENTIDO INPUT -> OUTPUT
    for i in range(depth-1):
        #PARA CADA UM DESTES, MARCA-SE INICIALMENTE TODAS AS POSIÇÕES COMO VERMELHAS
        #UTILIZA-SE A VARIÁVEL res_aux PARA ARMAZENAR AS CORES DE CADA ESTADO INTERMÉDIO
        a=[0 for i in range(amp_final_PB.n_qbits)]
        #DE SEGUIDA, ANALISA-SE CASO A CASO QUAIS PODERÃO SER MARCADAS COMO VERDES
        for gate in (layers[i]):
            if(i==0):
                if(gate[0]=='S' or gate[0]=='T' or gate[0]=='I'):
                    a[gate[1]]=1
                if(gate[0]=='CX'): 
                    a[gate[1]]=1
                    a[gate[2]]=1
            else:
                if((gate[0]=='S' or gate[0]=='T' or gate[0]=='I') and res[i-1][gate[1]]==1):
                    a[gate[1]]=1
                if(gate[0]=='CX' and res[i-1][gate[1]]==1):
                    a[gate[1]]=1
                if(gate[0]=='CX' and res[i-1][gate[2]]==1 and res[i-1][gate[1]]==1):
                    a[gate[2]]=1
        #POR FIM, QUANDO TODOS AS POSIÇÕES NUM ESTADO INTERMÉDIO SÃO MARCADAS, ADICIONA-SE res_aux À VARIÁVEL res
        res.append(a)
    return res


#VARRIMENTO NO SENTIDO OUTPUT -> INPUT DO CIRCUITO, MARCANDO A VERMELHO (0) TODAS AS POSIÇÕES EM QUE O ESTADO NÃO É DETERMINÍSTICO E A VERDE (1) CASO CONTRÁRIO. SEMELHANTE À FUNÇÃO forwardSweepPB, MAS AGORA COMEÇANDO PELO OUTPUT

def backwardsSweepPB(layers):
    res=[]
    depth=len(layers)
    for i in range(depth-1):
        a=[0 for i in range(amp_final_PB.n_qbits)]
        for gate in (layers[depth-1-i]):
            if(i==0):
                if(gate[0]=='S' or gate[0]=='T' or gate[0]=='I'):
                    a[gate[1]]=1
                if(gate[0]=='CX'): 
                    a[gate[1]]=1
                    a[gate[2]]=1
            else:
                if((gate[0]=='S' or gate[0]=='T' or gate[0]=='I') and res[i-1][gate[1]]==1):
                    a[gate[1]]=1
                if(gate[0]=='CX' and res[i-1][gate[1]]==1):
                    a[gate[1]]=1
                if(gate[0]=='CX' and res[i-1][gate[2]]==1 and res[i-1][gate[1]]==1):
                    a[gate[2]]=1
        res.append(a)
    res.reverse()
    return res


#A FUNÇÃO green_and_red_coloring RECEBE COMO ARGUMENTOS A LISTA COM TODAS AS CAMADAS DO CIRCUITO E OS RESULTADOS DOS VARRIMENTOS. DEVOLVE O RESULTADO DA OPERAÇÃO LÓGICA 'OR' ENTRE OS RESULTADOS DOS VARRIMENTOS, QUE PODE SER VISTA COMO A COLORAÇÃO GREEN/RED COMPLETA DE TODOS OS ESTADOS INTERMÉDIOS. ESTE RESULTADO SERÁ ÚTIL PARA A FUNÇÃO SEGUINTE QUE COMPUTA A COLORAÇÃO PINK/BLUE.

def green_and_red_coloring(layers, fs, bs):
    #res É A VARIÁVEL ONDE SE ARMAZENARÁ O RESULTADO DO 'OR' ENTRE OS RESULTADOS DOS VARRIMENTOS
    #lISTA DE LISTAS EM QUE CADA LISTA INTERIOR CORRESPONDE A UM ESTADO INTERMÉDIO DE TODOS OS QBITS
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


#A FUNÇÃO pink_and_blue_coloring_for RECEBE COMO ARGUMENTOS A LISTA COM TODAS AS CAMADAS DO CIRCUITO E OS RESULTADOS DE green_and_red_coloring. DEVOLVE A COLORAÇÃO PINK/BLUE NO SENTIDO INPUT -> OUTPUT.

def pink_and_blue_coloring_for(layers, green_red_color):
    pink_and_blue_coloring_for.n_of_pinks=0
    #res É A VARIÁVEL ONDE SE ARMAZENARÁ O RESULTADO DA COLORAÇÃO
    #lISTA DE LISTAS EM QUE CADA LISTA INTERIOR CORRESPONDE A UM ESTADO INTERMÉDIO DE TODOS OS QBITS
    #COPIA-SE PARA ESTA VARIÁVEL O RESULTADO DA COLORAÇÃO GREEN/RED.
    res=copy.deepcopy(green_red_color)
    depth=len(layers)
    for i in range(depth-1):
        for gate in layers[i]:
            #MARCA A ROSA (2) TODAS AS POSIÇÕES IMEDIATAMENTE A SEGUIR A UMA PORTA H, MARCADAS ANTERIORMENTE A VERMELHO.
            if(gate[0]=='H' and res[i][gate[1]]==0):
                res[i][gate[1]]=2
                pink_and_blue_coloring_for.n_of_pinks+=1
    for i in range(depth-1):
        for j in range(len(res[i])):
            #MARCA A AZUL (3) AS RESTANTES POISÇÕES MARCADAS ANTERIORMENTE A VERMELHO.
            if (res[i][j]==0):
                res[i][j]=3
                
    return res


#AS DUAS FUNÇÕES SEGUINTES SERVEM PARA VERIFICAR SE HÁ INCONSISTÊNCIAS ENTRE OS ESTADOS INTERMÉDIOS CALCULADOS COM RECURSO AOS VARRIMENTOS
#USANDO O RESULTADO DAS FUNÇÕES forwardSweepPB E backwardsSweepPB SÃO CALCULADAS DUAS VERSÕES DOS ESTADOS INTERMÉDIOS PARA CADA QUBIT.
#A FUNÇÃO nullAmpCheckAux RECEBE UM ESTADO (QUE PODE SER O ESTADO DE INPUT (OU OUTPUT, QUANDO SE ESTÁ A CALCULAR A VERSÃO DOS ESTADOS INTERMÉDIOS CORRESPONDENTE AO VARRIMENTO DO OUTPUT PARA O INPUT) OU QUALQUER UM DOS INTERMÉDIOS), A CAMADA DE PORTAS À FRENTE DESSE ESTADO E AS CORES DAS POSIÇÕES CORRESPONDENTES AO ESTADO IMEDIATAMENTE A SEGUIR À CAMADA DE PORTAS, PROVENIENTES DAS FUNÇÕES forwardSweepPB E backwardsSweepPB. ESTA FUNÇÃO RETORNA O ESTADO A SEGUIR À CAMADA DE PORTAS.

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


#A FUNÇÃO nullAmpCheckPB RECEBE O ESTADO DE INPUT, A LISTA COM TODAS AS CAMADAS DO CIRCUITO, AS COLORAÇÕES DOS VARRIMENTOS E O ESTADO DE OUTPUT. RETORNA UMA VARIÁVEL BINÁRIA a, QUE TEM VALOR ZERO SE FOREM DETETADAS INCONSISTÊNCIAS E VALOR UM CASO CONTRÁRIO, E OS ESTADOS INTERMÉDIOS AUXILIARES CORRESPONDENTES AOS VARRIMENTOS PARA A FRENTE E PARA TRÁS. ESTES SERÃO USADOS NUMA FUNÇÃO POSTERIOR PARA CALCULAR OS ESTADOS INTERMÉDIOS COMPLETOS.

def nullAmpCheckPB(stateIn, layers, forwardSweep, backwardsSweep, stateOut):
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
        for j in range(amp_final_PB.n_qbits):
            #SE NESTAS DUAS VERSÕES O MESMO QUBIT, NO MESMO ESTADO INTERMÉDIO, TIVER DOIS VALORES DETERMINÍSTICOS DIFERENTES (UM DELES 0 E O OUTRO 1) A VARIÁVEL a PASSA A ZERO, SIGNIFICANDO QUE FOI DETETADA UMA INCONSISTÊNCIA. MAL É DETETADA A INCONSISTÊNCIA, A FUNÇÃO PÁRA E RETORNA AS VARIÁVEIS. 
            if(statesAuxFor[i][j]!=statesAuxBack[i][j] and statesAuxFor[i][j]!='red' and statesAuxBack[i][j]!='red'):
                a=0
                break
        if(statesAuxFor[i][j]!=statesAuxBack[i][j] and statesAuxFor[i][j]!='red' and statesAuxBack[i][j]!='red'):
            break
    #CASO NÃO SEJA DETETADA NENHUMA INCONSISTÊNCIA A FUNÇÃO RETORNA AS VARIÁVEIS.
    return (a, statesAuxFor, statesAuxBack)


#A FUNÇÃO IMStatesPB CALCULA OS ESTADOS INTERMÉDIOS COMPLETOS. RECEBE COMO ARGUMENTOS O ESTADO DE INPUT, A LISTA COM TODAS AS CAMADAS DO CIRCUITO, AS VARIÁVEIS statesAuxFor/Back PROVENIENTES DA FUNÇÃO nullAmpCheckPB, O RESULTADO DA COLORAÇÃO PINK/BLUE E O ESTADO DE OUTPUT. ESTA FUNÇÃO DEVOLVE OS ESTADOS INTERMÉDIOS COMPLETOS.

def IMStatesPB(stateIn, layers, statesAuxFor, statesAuxBack, pink_and_blue_coloring, stateOut):
    #INICIALIZAM-SE AS VARIÁVEIS. A VARIÁVEL imstates É UMA LISTA DE LISTAS EM QUE imstates[i][j] CORRESPONDE AO ESTADO DO QUBIT j NO ESTADO INTERMÉDIO APÓS A CAMADA i E PODE TER COMO VALOR 0,1, 'pink', OU 'blue'.
    imstates=[]
    IMStatesPB.n_of_pinks=0
    depth=len(layers)
    #PARA COMPOR A LISTA imstates, PERCORREM-SE AS LISTAS statesAuxFor/Back. QUANDO, PELO MENOS, NUMA DAS LISTAS statesAuxFor/Back PARA O MESMO QUBIT, PARA O MESMO ESTADO INTERMÉDIO, EXISTIR UM VALOR DIFERENTE DE 'red' SERÁ ADICIONADO ESSE VALOR (0 OU 1). QUANDO O MESMO QUBIT, PARA O MESMO ESTADO INTERMÉDIO, TEM O VALOR 'red' EM AMBAS AS LISTAS, E O MESMO QUBIT ESTÁ MARCADO COMO ROSA NA COLORAÇÃO PINK/BLUE, É ADICIONADO 'pink' A imstates. QUANDO O MESMO QUBIT, PARA O MESMO ESTADO INTERMÉDIO, TEM O VALOR 'red' EM AMBAS AS LISTAS, MAS O MESMO QUBIT NÃO ESTÁ MARCADO COMO ROSA NA COLORAÇÃO PINK/BLUE, É ADICIONADO 'blue' A imstates. 
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


#AS DUAS FUNÇÕES SEGUINTES SERVEM PARA CALCULAR A AMPLITUDE DADOS OS ESTADOS DE INPUT E OUTPUT E AS CAMADAS DE PORTAS ENTRE ESTES, IGNORANDO AS AMPLITUDES PROVENIENTES DAS PORTAS COM INPUT E/OU OUTPUT NÃO DETERMINÍSTICO.
#A FUNÇÃO constantAmpAuxPB RECEBE COMO ARGUMENTOS OS ESTADOS DE INPUT E OUTPUT E UMA CAMADA DE PORTAS ENTRE ELES.

def constantAmpAuxPB(In, Out, layer):
    #A VARIÁVEL a É A VARIÁVEL ONDE SERÁ ARMAZENADA A AMPLITUDE. É INICIALIZADA COM O VALOR 1.
    a = 1
    n = len(layer)
    #É PERCORRIDA A CAMADA E, CASO NÃO HAJA POSIÇÕES A ROSA OU AZUL NO INPUT OU OUTPUT DE CADA UMA DAS PORTAS, É CALCULADA A AMPLITUDE PARA CADA UMA DAS PORTAS, USANDO AS FUNÇÕES DEFINIDAS NO INÍCIO DESTE MÓDULO. CADA AMPLITUDE CALCULADA SERÁ MULTIPLICADA PELO VALOR DA AMPLITUDE CALCULADA ATÉ ENTÃO. NO FIM DEVOLVE-SE ESTA AMPLITUDE.
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
    #print(a)
    return a


#A FUNÇÃO constantAmpPB RECEBE COMO ARGUMENTOS TODOS OS ESTADOS (INPUT E OUTPUT INCLUÍDOS) E AS VÁRIAS CAMADAS DE PORTAS. APLICA-SE A FUNÇÃO constantAmpAux SUCESSIVAMENTE A CADA CAMADA. NO FINAL OBTÉM-SE E RETORNA-SE A AMPLITUDE TOTAL.

def constantAmpPB(all_states, layers):
    a=1
    depth=len(layers)
    for i in range(depth):
        a = a * constantAmpAuxPB(all_states[i], all_states[i+1], layers[i])
    return a

#it É UM GERADOR DE PYTHON QUE GERA TODAS AS COMBINAÇÕES DE STRINGS BINÁRIAS COM max BITS. SERÁ ÚTIL PARA A ITERAÇÃO SOBRE TODOS OS ESTADOS INTERMÉDIOS A ROSA
def it(max):
        for i in range(1<<max):
            s=bin(i)[2:]
            s='0'*(max-len(s))+s
            s = list(map(int,list(s)))
            yield s
            
#A FUNÇÃO amp_final_PB É A FUNÇÃO QUE DADOS OS ESTADOS DE INPUT E OUTPUT E AS CAMADAS DE PORTAS, CALCULA A AMPLITUDE TOTAL PARA UM DADO CIRCUITO. RECEBE COMO ARGUMENTOS AS CAMADAS DE PORTAS,  OS ESTADOS DE INPUT E OUTPUT E O NÚMERO DE QUBITS.              

def amp_final_PB(stateIn, layers, stateOut, n_qbits):
    
    depth=len(layers)
    amp_final_PB.n_qbits=n_qbits
    
    #PRÉ PROCESSAMENTO (COLORAÇÃO VERDE/VERMELHO, VERIFICAÇÃO DE CONFLITOS DA COLORAÇÃO PARA A FRENTE E PARA TRÁS)
    fs=forwardSweepPB(layers)
    bs=backwardsSweepPB(layers)
    a, statesFor, statesBack = nullAmpCheckPB(stateIn, layers, fs, bs, stateOut)
    
    if(a==0): #SIGNIFICA QUE HOUVE CONFLITO -> AMPLITUDE = 0
        return 0
    
    #CÁLCULO DAS COLORAÇÕES GREEN/RED E PINK/BLUE
    green_red_coloring = green_and_red_coloring(layers, fs, bs)
    if(green_and_red_coloring.n_of_reds==0):
        imstates = IMStatesPB(stateIn, layers, statesFor, statesBack, green_red_coloring, stateOut) 
        all_states=copy.deepcopy(imstates)
        all_states.insert(0, stateIn)
        all_states.append(stateOut)
        return constantAmpPB(all_states, layers)
    pink_blue_coloring = pink_and_blue_coloring_for(layers, green_red_coloring)
    
    #CÁLCULO DOS ESTADOS INTERMÉDIOS
    imstates = IMStatesPB(stateIn, layers, statesFor, statesBack, pink_blue_coloring, stateOut) 
    #CRIAÇÃO DA VARIÁVEL all_states QUE JUNTA OS ESTADOS DE INPUT E OUTPUT AOS ESTADOS INTERMÉDIOS
    all_states=copy.deepcopy(imstates)
    all_states.insert(0, stateIn)
    all_states.append(stateOut)
    
    #CASO NÃO EXISTAM ROSAS, DEVOLVE IMEDIATAMENTE A AMPLITUDE TOTAL
    #if(IMStatesPB.n_of_pinks==0):
       # return constantAmpPB(all_states, layers)
    
    #n_pinks É O NÚMERO DE ESTADOS A ROSA. ESTE VALOR SERÁ O NÚMERO DE BITS COM O QUAL SE IRÃO GERAR TODAS AS COMBINAÇÕES BINÁRIAS PARA FAZER A ITERAÇÃO.
    n_pinks=IMStatesPB.n_of_pinks
    
    #amp_final É A VARIÁVEL QUE VAI CONTER A AMPLITUDE TOTAL FINAL
    amp_final=0
    #A VARIÁVEL b ARMAZENA A AMPLITUDE TOTAL DE TODAS AS GATES QUE NÃO TÊM INPUTS E/OU OUTPUTS A ROSA OU AZUL
    b=constantAmpPB(all_states, layers)
    
    #ITERAR SOBRE TODAS AS COMBINAÇÕES BINÁRIAS DE n_reds_middle NÚMERO DE BITS
    #PARA CADA UMA DAS COMBINAÇÕES BINÁRIAS
    for i in it(n_pinks):
        c=0
        a=1
        for j in range(depth):
            for gate in layers[j]:
                #RESOLVEM-SE OS ESTADOS A AZUL APÓS AS PORTAS CNOT
                if (gate[0] == 'CX' and (all_states[j+1][gate[1]] == 'blue' or all_states[j+1][gate[2]] == 'blue')):
                    if(all_states[j+1][gate[1]] == 'blue'):
                        all_states[j+1][gate[1]] = all_states[j][gate[1]]
                    if(all_states[j+1][gate[2]] == 'blue'):
                        all_states[j+1][gate[2]] = abs(all_states[j][gate[2]]-all_states[j][gate[1]])
                    #CALCULA-SE A AMPLITUDE DESTAS PORTAS
                    a = a * CX((all_states[j][gate[1]], all_states[j][gate[2]]),\
                               (all_states[j+1][gate[1]], all_states[j+1][gate[2]]))
                elif(gate[0] == 'CX' and j>0 and (pink_blue_coloring[j-1][gate[1]] != 1 or pink_blue_coloring[j-1][gate[2]] != 1)):
                    #CALCULA-SE A AMPLITUDE DAS PORTAS CNOT COM INPUT ANTERIORMENTE NÃO DETERMINÍSTICO, MAS ENTRETANTO JÁ RESOLVIDO
                    a = a * CX((all_states[j][gate[1]], all_states[j][gate[2]]),\
                               (all_states[j+1][gate[1]], all_states[j+1][gate[2]]))   
                #SUBSTITUEM-SE OS VALORES A ROSA PELOS VALORES DOS BITS CORRESPONDENTES DA COMBINAÇÃO
                elif(gate[0] == 'H' and all_states[j+1][gate[1]] == 'pink'):
                    all_states[j+1][gate[1]] = i[c]
                    c += 1
                    #CALCULA-SE A AMPLITUDE DAS PORTAS H QUE TINHAM COMO OUTPUT ESTADOS A ROSA
                    a = a * H(all_states[j][gate[1]], all_states[j+1][gate[1]])
                elif(gate[0] == 'H' and j>0 and pink_blue_coloring[j-1][gate[1]] != 1):
                    #CALCULA-SE A AMPLITUDE DAS PORTAS H COM INPUT ANTERIORMENTE NÃO DETERMINÍSTICO, MAS ENTRETANTO JÁ RESOLVIDO
                    a = a * H(all_states[j][gate[1]], all_states[j+1][gate[1]])
                #RESOLVEM-SE OS ESTADOS A AZUL APÓS AS PORTAS 'T', 'S' E 'I'. CALCULAM-SE AS AMPLITUDES DESTAS PORTAS
                elif(gate[0] == 'T' and all_states[j+1][gate[1]] == 'blue'):
                    all_states[j+1][gate[1]] = all_states[j][gate[1]]
                    a = a * T(all_states[j][gate[1]], all_states[j+1][gate[1]])
                elif(gate[0] == 'S' and all_states[j+1][gate[1]] == 'blue'):
                    all_states[j+1][gate[1]] = all_states[j][gate[1]]
                    a = a * S(all_states[j][gate[1]], all_states[j+1][gate[1]])
                elif(gate[0] == 'I' and all_states[j+1][gate[1]] == 'blue'):
                    all_states[j+1][gate[1]] = all_states[j][gate[1]]
                    a = a * I(all_states[j][gate[1]], all_states[j+1][gate[1]])
        #NO FIM DE CADA ITERAÇÃO, VOLTA-SE AOS ESTADOS INTERMÉDIOS COM A COLORAÇÃO BLUE/PINK
        all_states=copy.deepcopy(imstates)
        all_states.insert(0, stateIn)
        all_states.append(stateOut)
        #GUARDA-SE O VALOR DAS AMPLITUDES CALCULADAS, MULTIPLICANDO-AS NA VARIÁVEL CUMULATIVA a E MULTIPLICA-SE PELA AMPLITUDE ARMAZENADA NA VARIÁVEL B
        a = a * b
        #FINALMENTE ADICIONA-SE À VARIÁVEL amp_final.
        amp_final = amp_final + a
    return amp_final

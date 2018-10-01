import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from sklearn.svm import SVC
from sklearn.base import clone
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, cross_val_predict, cross_validate, StratifiedKFold, RepeatedStratifiedKFold, GroupKFold,LeaveOneGroupOut

##########################################################################################################################################################################
''' CLASSES '''
##########################################################################################################################################################################

from SinalVoluntario import SinalVoluntario
from TreinaValidacaoCruzada import TreinaValidacaoCruzada
from PCA_Teste import PCA_Teste               

##########################################################################################################################################################################
''' FUNÇÕES '''
##########################################################################################################################################################################
     
def TamanhoJanela(TriggerFinal):
    flag = True
    for i, v in enumerate(TriggerFinal):
        if v >= 1.4 and flag == True:
            Yinf = i
            flag = False
        if v < 1 and flag == False:
            flag = True
            Ysup = i
            break
    Y = Ysup - Yinf

    return Y # Retorna um tamanho de janela. Usado para fazer o janelamento e separação das amostras de Contração
def PassaFiltros(DataFrameCH, F):
    '''    
    Retorna o SINAL filtrado:
    1 - Filtra sinal DC - Passa-Alta
    2 - Filtra sinal 60Hz, 120Hz e 180Hz
    3 - Retifica sinal
    4 - Filtra o sinal com um filtro Passa-Baixa
    '''
    #
    # Detalhes do sinal e parâmetros para a construção dos filtros
    #
    fs = 2000  # Frequência de amostragem (Hz)
    Fn60Hz = 60.0  # Frequência para remoer com o filtro NOTCH - Remover interferência da rede 60 HZ
    Fn120Hz = 120.0  # Frequência para remoer com o filtro NOTCH - Remover interferência da rede 120 HZ
    Fn180Hz = 180.0  # Frequência para remoer com o filtro NOTCH - Remover interferência da rede 180 HZ
    Fpa = 10.0 # Frequência de corte do filtro PASSA-ALTA - Remoção do Offset gerado pelo sinal DC
    Fpb = 10.0 # Frequência de corte do filtro PASSA-BAIXA - Suavização do sinal
    Q = 1  # Fator de qualidade do filtro NOTCH

    # Frequência normalizada:
    Wn60Hz = Fn60Hz/(fs/2) # Para o filtro NOTCH 60 Hz
    Wn120Hz = Fn120Hz/(fs/2) # Para o filtro NOTCH 120 Hz
    Wn180Hz = Fn180Hz/(fs/2) # Para o filtro NOTCH 180 HZ
    Wpb = Fpb/(fs/2) # Para o filtro PASSA-BAIXA
    Wpa = Fpa/(fs/2) # Para o filtro PASSA-ALTA

    #
    # Construção de filtros
    #
    b11, a11 = signal.iirnotch(Wn60Hz, Q) # Design notch filter - Fc = 60Hz
    b12, a12 = signal.iirnotch(Wn120Hz, Q) # Design notch filter - Fc = 120Hz
    b13, a13 = signal.iirnotch(Wn180Hz, Q) # Design notch filter - Fc = 180Hz
    b2, a2 = signal.butter(2, Wpa, 'highpass') # Design butter filter - Fc = 10Hz
    b3, a3 = signal.butter(6, Wpb, 'lowpass') # Design butter filter - Fc = 20Hz

    filtradoDC = signal.filtfilt(b2, a2, DataFrameCH) # Passa um filtro PASSA-ALTA para remover nível DC do SINAL
    filtradoRede = signal.filtfilt(b11, a11, filtradoDC) # Passa um filtro NOTCH no SINAL para remover 60Hz
    #filtradoRede2 = signal.filtfilt(b12, a12, filtradoRede1) # Passa um filtro NOTCH no SINAL para remover 60Hz
    #filtradoRede3 = signal.filtfilt(b13, a13, filtradoRede2) # Passa um filtro NOTCH no SINAL para remover 60Hz
    
    retificado = np.abs(filtradoRede) # Retifica o SINAL filtrado
    passaBaixa = signal.filtfilt(b3, a3, retificado) # Passa um filtro PASSA-BAIXA no SINAL retificado
    if F == 1:
        return filtradoDC
    if F == 2:
        return filtradoRede
    if F == 3:
        return retificado
    if F == 4:
        return passaBaixa # Passando filtros no sinal
    pass
def Amplificar(SINAL, X):
    S = list(SINAL)
    amp = [*map(lambda x: x*X,list(S))] #amplificar o sinal retificado
    return amp # amplifica um sinal em X vezes
def CorrigeTrigger(atraso, TriggerDataFrame):
    '''
    Corrige o Trigger de acordo com o tempo de reação estimado do voluntário
    atraso -> atraso em amostras
              Em segundos: X = (atraso/2) [s]
    '''
    TR = [0]*atraso
    TR.extend(TriggerDataFrame)
    while len(TR) > len(TriggerDataFrame):
        del TR[len(TR) - 1]
    return TR 
def RemoveInicioColeta(TRIGGER, SINAL, J):
    '''
    Remove o inicio de coleta, o SINAL começa a partir da primeira borda de subida do TRIGGER. 
    J = 0 -> retorna o SINAL
    J = 1 -> retorna o TRIGGER
    '''
    ListaRemove = []
    S = list(SINAL)
    T = list(TRIGGER)
    for i in T:
        if i < 1.4:
            ListaRemove.append(i)
        if i > 1.4:
            break        
    DLista = int(len(ListaRemove))

    if J == 0:
        del S[:DLista]
        S.extend(ListaRemove)
        return S
    elif J == 1:
        del T[:DLista]
        T.extend(ListaRemove)
        return T
    pass
def SinalFinalGERAL(ListaDataFrame, SignalType):
    '''
    Recebe o bruto, sincroniza com o trigger e remove todo o inicio desnecessario da coleta.
    Retorna os sinais e od triggers finais de todas as coletas.
    O sinal pode assumir qualquer forma de acordo com o parâmetro 'SignalType'.
    SignalType -> Recebe valores 1, 2, 3 e 4 (Cada valor está de acordo com a função PassarFiltros()
    '''
    DelayT = 500 # Tempo de atraso no TRIGER para sincronização SINAL/TRIGGER em amostras
    
    T1 = CorrigeTrigger(DelayT, ListaDataFrame[0]['Trigger'])
    SR11 = RemoveInicioColeta(T1, PassaFiltros(ListaDataFrame[0]['CH1'], SignalType), 0)
    SR12 = RemoveInicioColeta(T1, PassaFiltros(ListaDataFrame[0]['CH2'], SignalType), 0)
    SR13 = RemoveInicioColeta(T1, PassaFiltros(ListaDataFrame[0]['CH3'], SignalType), 0)
    SR14 = RemoveInicioColeta(T1, PassaFiltros(ListaDataFrame[0]['CH4'], SignalType), 0)

    T2 = CorrigeTrigger(DelayT, ListaDataFrame[1]['Trigger'])
    SR21 = RemoveInicioColeta(T2, PassaFiltros(ListaDataFrame[1]['CH1'], SignalType), 0)
    SR22 = RemoveInicioColeta(T2, PassaFiltros(ListaDataFrame[1]['CH2'], SignalType), 0)
    SR23 = RemoveInicioColeta(T2, PassaFiltros(ListaDataFrame[1]['CH3'], SignalType), 0)
    SR24 = RemoveInicioColeta(T2, PassaFiltros(ListaDataFrame[1]['CH4'], SignalType), 0)

    T3 = CorrigeTrigger(DelayT, ListaDataFrame[2]['Trigger'])
    SR31 = RemoveInicioColeta(T3, PassaFiltros(ListaDataFrame[2]['CH1'], SignalType), 0)
    SR32 = RemoveInicioColeta(T3, PassaFiltros(ListaDataFrame[2]['CH2'], SignalType), 0)
    SR33 = RemoveInicioColeta(T3, PassaFiltros(ListaDataFrame[2]['CH3'], SignalType), 0)
    SR34 = RemoveInicioColeta(T3, PassaFiltros(ListaDataFrame[2]['CH4'], SignalType), 0)

    T4 = CorrigeTrigger(DelayT, ListaDataFrame[3]['Trigger'])
    SR41 = RemoveInicioColeta(T4, PassaFiltros(ListaDataFrame[3]['CH1'], SignalType), 0)
    SR42 = RemoveInicioColeta(T4, PassaFiltros(ListaDataFrame[3]['CH2'], SignalType), 0)
    SR43 = RemoveInicioColeta(T4, PassaFiltros(ListaDataFrame[3]['CH3'], SignalType), 0)
    SR44 = RemoveInicioColeta(T4, PassaFiltros(ListaDataFrame[3]['CH4'], SignalType), 0)

    T5 = CorrigeTrigger(DelayT, ListaDataFrame[4]['Trigger'])
    SR51 = RemoveInicioColeta(T1, PassaFiltros(ListaDataFrame[4]['CH1'], SignalType), 0)
    SR52 = RemoveInicioColeta(T1, PassaFiltros(ListaDataFrame[4]['CH2'], SignalType), 0) 
    SR53 = RemoveInicioColeta(T1, PassaFiltros(ListaDataFrame[4]['CH3'], SignalType), 0)
    SR54 = RemoveInicioColeta(T1, PassaFiltros(ListaDataFrame[4]['CH4'], SignalType), 0)

    T6 = CorrigeTrigger(DelayT, ListaDataFrame[5]['Trigger'])
    SR61 = RemoveInicioColeta(T2, PassaFiltros(ListaDataFrame[5]['CH1'], SignalType), 0)
    SR62 = RemoveInicioColeta(T2, PassaFiltros(ListaDataFrame[5]['CH2'], SignalType), 0)
    SR63 = RemoveInicioColeta(T2, PassaFiltros(ListaDataFrame[5]['CH3'], SignalType), 0)
    SR64 = RemoveInicioColeta(T2, PassaFiltros(ListaDataFrame[5]['CH4'], SignalType), 0)

    T7 = CorrigeTrigger(DelayT, ListaDataFrame[6]['Trigger'])
    SR71 = RemoveInicioColeta(T3, PassaFiltros(ListaDataFrame[6]['CH1'], SignalType), 0)
    SR72 = RemoveInicioColeta(T3, PassaFiltros(ListaDataFrame[6]['CH2'], SignalType), 0)
    SR73 = RemoveInicioColeta(T3, PassaFiltros(ListaDataFrame[6]['CH3'], SignalType), 0)
    SR74 = RemoveInicioColeta(T3, PassaFiltros(ListaDataFrame[6]['CH4'], SignalType), 0)

    T8 = CorrigeTrigger(DelayT, ListaDataFrame[7]['Trigger'])
    SR81 = RemoveInicioColeta(T4, PassaFiltros(ListaDataFrame[7]['CH1'], SignalType), 0)
    SR82 = RemoveInicioColeta(T4, PassaFiltros(ListaDataFrame[7]['CH2'], SignalType), 0)
    SR83 = RemoveInicioColeta(T4, PassaFiltros(ListaDataFrame[7]['CH3'], SignalType), 0)
    SR84 = RemoveInicioColeta(T4, PassaFiltros(ListaDataFrame[7]['CH4'], SignalType), 0)

    # Trigger arrumado para cada coleta
    TC1 = RemoveInicioColeta(T1, PassaFiltros(ListaDataFrame[0]['CH1'], SignalType), 1)
    TC2 = RemoveInicioColeta(T2, PassaFiltros(ListaDataFrame[1]['CH1'], SignalType), 1)
    TC3 = RemoveInicioColeta(T3, PassaFiltros(ListaDataFrame[2]['CH1'], SignalType), 1)
    TC4 = RemoveInicioColeta(T4, PassaFiltros(ListaDataFrame[3]['CH1'], SignalType), 1)
    TC5 = RemoveInicioColeta(T5, PassaFiltros(ListaDataFrame[4]['CH1'], SignalType), 1)
    TC6 = RemoveInicioColeta(T6, PassaFiltros(ListaDataFrame[5]['CH1'], SignalType), 1)
    TC7 = RemoveInicioColeta(T7, PassaFiltros(ListaDataFrame[6]['CH1'], SignalType), 1)
    TC8 = RemoveInicioColeta(T8, PassaFiltros(ListaDataFrame[7]['CH1'], SignalType), 1)

    return TC1, SR11, SR12, SR13, SR14, TC2, SR21, SR22, SR23, SR24, TC3, SR31, SR32, SR33, SR34, TC4, SR41, SR42, SR43, SR44, TC5, SR51, SR52, SR53, SR54, TC6, SR61, SR62, SR63, SR64, TC7, SR71, SR72, SR73, SR74, TC8, SR81, SR82, SR83, SR84
def Amostras(TriggerFinal, SINAL):
    '''
    Remove o tempo de repouso
    Retorna um vetor com todas as contrações. 
    '''
    T = list(TriggerFinal)
    S = list(SINAL)    
    vetorMov = []
    vetorRep = []

    flagMov = True
    tJ = int(TamanhoJanela(T))
    count = 0
    while count < 120:
        for i, v in enumerate(T):
            if v > 1.4 and flagMov == True:
                vetorMov.extend(S[i:i+tJ])
                flagMov = False
                count += 1
            if v < 1 and flagMov == False:
                vetorRep.extend(S[i:i+tJ])
                flagMov = True
                count += 1
    #vetorMov.extend(vetorRep)
    return vetorMov, vetorRep
def VetorDeAmostras(TriggerFinal, SINAL):
    '''
    Retorna o vetor com os SINAIS DE CONTRAÇÃO separados.
    Contem uma lista com N listas. Cada lista N é uma janela de contração.
    '''
    sMov, sRep = Amostras(TriggerFinal, SINAL)
    AmostraMov = []
    AmostraRep = []
    tJ = int(TamanhoJanela(TriggerFinal))
    for T in range(60):
        AmostraMov.append(sMov[0:tJ])
        AmostraRep.append(sRep[0:tJ])
        del sMov[0:tJ]
        del sRep[0:tJ]
    return AmostraMov, AmostraRep
def MAV(SINAL):
    mav = np.sum(SINAL)/len(SINAL)
    return float(mav)
def SSC(SINAL):
    ssc = np.count_nonzero(np.diff(np.sign(np.diff(SINAL))))
    return float(ssc)
def WL(SINAL):
    wl = np.sum(np.abs(np.diff(SINAL)))
    return float(wl)
def RMS(SINAL): 
    rms = np.sqrt(np.mean(np.square(SINAL)))
    return float(rms) 
def VAR(SINAL):
    var = np.var(SINAL)
    return var
def ZC(SINAL):
    zc = ((SINAL[:-1] * SINAL[1:]) < 0).sum()
    return float(zc)
def VetorATRIBUTOS(TriggerFinal, SINAL, Atributo, sRep=False):
    '''
    Retorna um vetor com os valores RMS de cada SINAL DE CONTRAÇÃO. 
    Retorna como -Pandas.Series-
    '''
    aMov, aRep = VetorDeAmostras(TriggerFinal, SINAL)
    ListaAtributos = []
    if sRep == False:
        A = aMov
    elif sRep == True:
        A = aRep
    if Atributo == 'RMS':
        for i, v in enumerate(A):
            ListaAtributos.append(RMS(A[i]))
        rms = pd.Series(ListaAtributos)
        return rms 
    if Atributo == 'ZC':
        for i, v in enumerate(A):
            ListaAtributos.append(ZC(np.array(A[i])))
        zc = pd.Series(ListaAtributos)
        return zc 
    if Atributo == 'VAR':
        for i, v in enumerate(A):
            ListaAtributos.append(VAR(np.array(A[i])))
        var = pd.Series(ListaAtributos)
        return var 
    if Atributo == 'SSC':
        for i, v in enumerate(A):
            ListaAtributos.append(SSC(np.array(A[i])))
        ssc = pd.Series(ListaAtributos)
        return ssc 
    if Atributo == 'MAV':
        for i, v in enumerate(A):
            ListaAtributos.append(MAV(np.array(A[i])))
        mav = pd.Series(ListaAtributos)
        return mav 
    if Atributo == 'WL':
        for i, v in enumerate(A):
            ListaAtributos.append(WL(np.array(A[i])))
        wl = pd.Series(ListaAtributos)
        return wl
    pass
def DataFrameCarac(ListaDF, a, sRep=False):
    # Concatena todos os valores RMS de todas as coletas de um mesmo canal
    # Valor RMS de todas as 240 CONTRAÇÕES do CANAL 1 - COLETA 1
    AtributoCH1_C1 = pd.concat([VetorATRIBUTOS(ListaDF[0], ListaDF[1], a, sRep),VetorATRIBUTOS(ListaDF[5], ListaDF[6], a, sRep),VetorATRIBUTOS(ListaDF[10], ListaDF[11], a, sRep),VetorATRIBUTOS(ListaDF[15], ListaDF[16], a, sRep)], ignore_index=True)
    # Valor RMS de todas as 240 CONTRAÇÕES do CANAL 2 - COLETA 1
    AtributoCH2_C1 = pd.concat([VetorATRIBUTOS(ListaDF[0], ListaDF[2], a, sRep),VetorATRIBUTOS(ListaDF[5], ListaDF[7], a, sRep),VetorATRIBUTOS(ListaDF[10], ListaDF[12], a, sRep),VetorATRIBUTOS(ListaDF[15], ListaDF[17], a, sRep)], ignore_index=True)
    # Valor RMS de todas as 240 CONTRAÇÕES do CANAL 3 - COLETA 1
    AtributoCH3_C1 = pd.concat([VetorATRIBUTOS(ListaDF[0], ListaDF[3], a, sRep),VetorATRIBUTOS(ListaDF[5], ListaDF[8], a, sRep),VetorATRIBUTOS(ListaDF[10], ListaDF[13], a, sRep),VetorATRIBUTOS(ListaDF[15], ListaDF[18], a, sRep)], ignore_index=True)
    # Valor RMS de todas as 240 CONTRAÇÕES do CANAL 4 - COLETA 1
    AtributoCH4_C1 = pd.concat([VetorATRIBUTOS(ListaDF[0], ListaDF[4], a, sRep),VetorATRIBUTOS(ListaDF[5], ListaDF[9], a, sRep),VetorATRIBUTOS(ListaDF[10], ListaDF[14], a, sRep),VetorATRIBUTOS(ListaDF[15], ListaDF[19], a, sRep)], ignore_index=True)
    
    # Valor RMS de todas as 240 CONTRAÇÕES do CANAL 1 - COLETA 2
    AtributoCH1_C2 = pd.concat([VetorATRIBUTOS(ListaDF[20], ListaDF[21], a, sRep),VetorATRIBUTOS(ListaDF[25], ListaDF[26], a, sRep),VetorATRIBUTOS(ListaDF[30], ListaDF[31], a, sRep),VetorATRIBUTOS(ListaDF[35], ListaDF[36], a, sRep)], ignore_index=True)
    # Valor RMS de todas as 240 CONTRAÇÕES do CANAL 2 - COLETA 2
    AtributoCH2_C2 = pd.concat([VetorATRIBUTOS(ListaDF[20], ListaDF[22], a, sRep),VetorATRIBUTOS(ListaDF[25], ListaDF[27], a, sRep),VetorATRIBUTOS(ListaDF[30], ListaDF[32], a, sRep),VetorATRIBUTOS(ListaDF[35], ListaDF[37], a, sRep)], ignore_index=True)
    # Valor RMS de todas as 240 CONTRAÇÕES do CANAL 3 - COLETA 2
    AtributoCH3_C2 = pd.concat([VetorATRIBUTOS(ListaDF[20], ListaDF[23], a, sRep),VetorATRIBUTOS(ListaDF[25], ListaDF[28], a, sRep),VetorATRIBUTOS(ListaDF[30], ListaDF[33], a, sRep),VetorATRIBUTOS(ListaDF[35], ListaDF[38], a, sRep)], ignore_index=True)
    # Valor RMS de todas as 240 CONTRAÇÕES do CANAL 4 - COLETA 2
    AtributoCH4_C2 = pd.concat([VetorATRIBUTOS(ListaDF[20], ListaDF[24], a, sRep),VetorATRIBUTOS(ListaDF[25], ListaDF[29], a, sRep),VetorATRIBUTOS(ListaDF[30], ListaDF[34], a, sRep),VetorATRIBUTOS(ListaDF[35], ListaDF[39], a, sRep)], ignore_index=True)

    # Cria um DATAFRAME para colocar todas as CARACTERÍSTICAS do SINAL - COLETA 1
    FeaturesEMG_C1 = pd.DataFrame()

    # Cria um DATAFRAME para colocar todas as CARACTERÍSTICAS do SINAL - COLETA 2
    FeaturesEMG_C2 = pd.DataFrame()

    #Coloca todos os valores RMS de todos os canais em um DATAFRAME, sendo as CARACTERÍSTICAS do SINAL
    FeaturesEMG_C1[a+'CH1'], FeaturesEMG_C1[a+'CH2'], FeaturesEMG_C1[a+'CH3'], FeaturesEMG_C1[a+'CH4'] = AtributoCH1_C1, AtributoCH2_C1, AtributoCH3_C1, AtributoCH4_C1
    FeaturesEMG_C2[a+'CH1'], FeaturesEMG_C2[a+'CH2'], FeaturesEMG_C2[a+'CH3'], FeaturesEMG_C2[a+'CH4'] = AtributoCH1_C2, AtributoCH2_C2, AtributoCH3_C2, AtributoCH4_C2
    
    return FeaturesEMG_C1, FeaturesEMG_C2
def sRep80(Ref):
    L = []
    for df1 in Ref:
        df2 = df1.mean(axis=1).sort_values(ascending=False)
        m = df2.mean()
        df2 = df2[df2 < df2.mean()]
        L.extend([df2])
    L1 = np.random.choice(L[0].index, 40, False)
    L2 = np.random.choice(L[1].index, 40, False)    
    
    return (L1, L2)
def SinalFinal(DF_Mov, DF_Rep, sR):
    B1 = DF_Rep[0].drop(list(DF_Rep[0].drop(list(sR[0])).index))
    B2 = DF_Rep[1].drop(list(DF_Rep[1].drop(list(sR[1])).index))
    df_F = [pd.concat([DF_Mov[0], B1], ignore_index=True), pd.concat([DF_Mov[1], B2], ignore_index=True)]
    return df_F
def NormalizaDadosCOLUNA(DataFrame):    
    df = DataFrame.copy()
    for S in df:
        Max = float(df[S].max())
        Min = float(df[S].min())
        for i, v in enumerate(df[S]):
            df[S][i] = (100*(float(v)-Min))/(Max-Min) #(((float(v)/100)*(Max - Min)) + Min) #(100.0/Max)*float(v)
    return df
def ConcatAtrib(*atributos):
    return pd.concat(atributos, axis=1)
def ExtraiNovoAtributo(DATAFRAME, Atributo):
    df1 = DATAFRAME[Atributo+'CH1']
    df2 = DATAFRAME[Atributo+'CH2']
    df3 = DATAFRAME[Atributo+'CH3']
    df4 = DATAFRAME[Atributo+'CH4']
    NA = pd.DataFrame()
    NA[Atributo+'N01'] = np.abs(df1-df2)
    NA[Atributo+'N02'] = np.abs(df1-df3)
    NA[Atributo+'N03'] = np.abs(df1-df4)
    NA[Atributo+'N04'] = np.abs(df2-df3)
    NA[Atributo+'N05'] = np.abs(df2-df4)
    NA[Atributo+'N06'] = np.abs(df3-df4)
    return NA
def DesenhaFront(CARAC, RESP, I, L):
    CP_12 = ConcatAtrib(CARAC.iloc[:I, L[0]:L[1]], CARAC.iloc[:I, L[2]:L[3]])
    X = np.array(CP_12)
    y = np.ravel(RESP[:I].copy())
    X0, X1 = X[:, 0], X[:, 1]   
    X0_min, X0_max = X0.min() - 1, X0.max() + 1
    X1_min, X1_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(X0_min, X0_max, 0.1), np.arange(X1_min, X1_max, 0.1))
    models = (SVC(kernel='rbf', gamma=Val1.grid_cv.best_estimator_.gamma, C=Val1.grid_cv.best_estimator_.C))
    models.fit(X, y)
    Z = models.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.plasma) #colors=('black','green','royalblue','red','darkorange','darkmagenta','sienna')
    DF = pd.DataFrame()
    DF['CP1'], DF['CP2'], DF['Resposta']= X0, X1, y
    plt.scatter(DF['CP1'], DF['CP2'], c=y, cmap=plt.cm.plasma,s=20, edgecolors='k')
def NovaOrdem(CARAC, RESP):
    df = ConcatAtrib(CARAC, RESP)
    df = df.sort_values('Resposta').reset_index(drop=True)
    df1 = df.iloc[:,:-1].copy()
    return df1

##########################################################################################################################################################################
''' CARREGANDO DADOS DO VOLUNTÁRIO '''
##########################################################################################################################################################################

# CARREGA DADOS DO VOLUNTÁRIO - 8 COLETAS DE 360seg - EMG DOS 4 CANAIS 
VOL = SinalVoluntario('Hallef') 
VOL_Coleta = VOL.CarregaDados()
VOL_Resp = VOL.CarregaRespostas() # CARREGA UMA LISTA COM AS RESPOSTAS DE CADA CONTRAÇÃO
VOL_Resp1, VOL_Resp2 = VOL_Resp[:int(len(VOL_Resp)/2)], VOL_Resp[int(len(VOL_Resp)/2):].reset_index(drop=True)
VOL_Ordem_Resp = VOL.RespEmOrdem[:int(len(VOL.RespEmOrdem)/2)].reset_index(drop=True)

##########################################################################################################################################################################
''' GERANDO TABELA COM PRINCIPAIS CARACTERÍSTICAS DE CADA CANAL '''
##########################################################################################################################################################################

VOL_SinalFinalBruto = SinalFinalGERAL(VOL_Coleta, 2)
VOL_SinalFinalRetificado = SinalFinalGERAL(VOL_Coleta, 3)

VOL_TabMAV_Mov, VOL_TabMAV_Rep = DataFrameCarac(VOL_SinalFinalRetificado, 'MAV'), DataFrameCarac(VOL_SinalFinalRetificado, 'MAV', True)
VOL_TabSSC_Mov, VOL_TabSSC_Rep = DataFrameCarac(VOL_SinalFinalBruto, 'SSC'), DataFrameCarac(VOL_SinalFinalBruto, 'SSC', True)
VOL_TabWL_Mov, VOL_TabWL_Rep = DataFrameCarac(VOL_SinalFinalBruto, 'WL'), DataFrameCarac(VOL_SinalFinalBruto, 'WL', True)
VOL_TabRMS_Mov, VOL_TabRMS_Rep = DataFrameCarac(VOL_SinalFinalRetificado, 'RMS'), DataFrameCarac(VOL_SinalFinalRetificado, 'RMS', True)
VOL_TabVAR_Mov, VOL_TabVAR_Rep = DataFrameCarac(VOL_SinalFinalBruto, 'VAR'), DataFrameCarac(VOL_SinalFinalBruto, 'VAR', True)
VOL_TabZC_Mov, VOL_TabZC_Rep = DataFrameCarac(VOL_SinalFinalBruto, 'ZC'), DataFrameCarac(VOL_SinalFinalBruto, 'ZC', True)

sRep = sRep80(VOL_TabMAV_Rep)

VOL_TabelasMAV = SinalFinal(VOL_TabMAV_Mov, VOL_TabMAV_Rep, sRep)
VOL_TabelasSSC = SinalFinal(VOL_TabSSC_Mov, VOL_TabSSC_Rep, sRep)
VOL_TabelasWL = SinalFinal(VOL_TabWL_Mov, VOL_TabWL_Rep, sRep)
VOL_TabelasRMS = SinalFinal(VOL_TabRMS_Mov, VOL_TabRMS_Rep, sRep)
VOL_TabelasVAR = SinalFinal(VOL_TabVAR_Mov, VOL_TabVAR_Rep, sRep)
VOL_TabelasZC = SinalFinal(VOL_TabZC_Mov, VOL_TabZC_Rep, sRep)

Carac_MAV_Coleta1, Carac_MAV_Coleta2  = NormalizaDadosCOLUNA(VOL_TabelasMAV[0]), NormalizaDadosCOLUNA(VOL_TabelasMAV[1])
Carac_SSC_Coleta1, Carac_SSC_Coleta2  = NormalizaDadosCOLUNA(VOL_TabelasSSC[0]), NormalizaDadosCOLUNA(VOL_TabelasSSC[1])
Carac_WL_Coleta1, Carac_WL_Coleta2  = NormalizaDadosCOLUNA(VOL_TabelasWL[0]), NormalizaDadosCOLUNA(VOL_TabelasWL[1])
Carac_RMS_Coleta1, Carac_RMS_Coleta2  = NormalizaDadosCOLUNA(VOL_TabelasRMS[0]), NormalizaDadosCOLUNA(VOL_TabelasRMS[1])
Carac_VAR_Coleta1, Carac_VAR_Coleta2  = NormalizaDadosCOLUNA(VOL_TabelasVAR[0]), NormalizaDadosCOLUNA(VOL_TabelasVAR[1])
Carac_ZC_Coleta1, Carac_ZC_Coleta2 = NormalizaDadosCOLUNA(VOL_TabelasZC[0]), NormalizaDadosCOLUNA(VOL_TabelasZC[1])

TabelaDeAtributos_VOL_1 = ConcatAtrib(Carac_MAV_Coleta1, Carac_SSC_Coleta1, Carac_WL_Coleta1, Carac_RMS_Coleta1, Carac_VAR_Coleta1, Carac_ZC_Coleta1)
TabelaDeAtributos_VOL_2 = ConcatAtrib(Carac_MAV_Coleta2, Carac_SSC_Coleta2, Carac_WL_Coleta2, Carac_RMS_Coleta2, Carac_VAR_Coleta2, Carac_ZC_Coleta2)

TOTALC1 = pd.concat([TabelaDeAtributos_VOL_1,TabelaDeAtributos_VOL_2], ignore_index=True)
TOTALR = pd.concat([VOL_Resp1,VOL_Resp2], ignore_index=True)

MAV_SSC_WL1 = ConcatAtrib(Carac_MAV_Coleta1, Carac_SSC_Coleta1, Carac_WL_Coleta1)
MAV_SSC_WL2 = ConcatAtrib(Carac_MAV_Coleta2, Carac_SSC_Coleta2, Carac_WL_Coleta2)
MAV_SSC_WL_Total = pd.concat([MAV_SSC_WL1,MAV_SSC_WL2], ignore_index=True)

############################################################################################################################################################################
''' Amostras ORDENADAS '''
############################################################################################################################################################################

TDA_VOL_1 = NovaOrdem(TabelaDeAtributos_VOL_1, VOL_Resp1)
TDA_VOL_2 = NovaOrdem(TabelaDeAtributos_VOL_2, VOL_Resp2)
TC1 = pd.concat([TDA_VOL_1,TDA_VOL_2], ignore_index=True)
RT1 = VOL.RespEmOrdem
RT280 = RT1.iloc[:280].copy()
mav_ssc_wl1 = NovaOrdem(MAV_SSC_WL1, VOL_Resp1)
mav_ssc_wl2 = NovaOrdem(MAV_SSC_WL2, VOL_Resp2)
mav_ssc_wl_T = pd.concat([mav_ssc_wl1,mav_ssc_wl2], ignore_index=True)

############################################################################################################################################################################
''' TREINAMENTO E TESTE '''
############################################################################################################################################################################


Val1 = TreinaValidacaoCruzada(TDA_VOL_1, RT280)
Val1.Parametros(mostraDivisao=False,group=False)
print(Val1.matrizDeConfusao)
print(Val1.tabelaDeClassificacao)

pca1 = PCA_Teste(TDA_VOL_1, RT280)
pca1.RealizaPCA()
sinalPCA1 = pca1.final1

Val12 = TreinaValidacaoCruzada(sinalPCA1, RT280)
Val12.Parametros(group=False)
print(Val12.matrizDeConfusao)
print(Val12.tabelaDeClassificacao)

pca1.pca.explained_variance_ratio_.sum()
pca1.pca.n_components_

#####################################################################################################################################################################
''' ADICIONANDO AS COMPONENTES PRINCIPAIS AS OUTRAS CARACTERÍSTICAS '''
#####################################################################################################################################################################

pcaAtributos = ConcatAtrib(sinalPCA1.iloc[:,:1], MAV_SSC_WL2)

Val13 = TreinaValidacaoCruzada(pcaAtributos, VOL_Resp2)
Val13.Parametros(group=True)
print(Val13.matrizDeConfusao)
print(Val13.tabelaDeClassificacao)

#####################################################################################################################################################################
''' PLOTAR DISPERÇÃO ENTRE ATRIBUTOS E FRONTEIRAS DE DECISÃO '''
#####################################################################################################################################################################

sinalPCA1['Resposta'] = RT280
sns.pairplot(sinalPCA1,hue='Resposta', aspect=1, size=1.7, palette=dict(Repousar = 'black', Pinçar = 'green', Supinar = 'royalblue', Pronar = 'red',Fechar = 'darkorange', Estender = 'darkmagenta', Flexionar = 'sienna'))
sinalPCA1['Resposta'] = RT1
plt.figure(1)
DesenhaFront(sinalPCA1, VOL.RespEmOrdem_Num[:280], 280, [0,1,1,2])
plt.figure(2)
sns.pairplot(sinalPCA1,hue='Resposta', aspect=1, size=2, palette=dict(Repousar = 'black', Pinçar = 'green', Supinar = 'royalblue', Pronar = 'red',Fechar = 'darkorange', Estender = 'darkmagenta', Flexionar = 'sienna'))
plt.show()

#################################################################################################################################################################################
''' TESTE COM CONJUNTO DE DADOS DIFERENTES '''
#################################################################################################################################################################################

Pred1 = Val1.grid_cv.predict(TabelaDeAtributos_VOL_1)
print(confusion_matrix(VOL_Resp1,Pred1))
print(classification_report(VOL_Resp1,Pred1))

##################################################################################################################################################################
##################################################################################################################################################################
''' TESTE COM COMBINAÇÃO DE ATRIBUTOS - CRIAÇÃO DE NOVOS ATRIBUTOS'''
##################################################################################################################################################################
##################################################################################################################################################################

novoRMS1, novoRMS2 = NormalizaDadosCOLUNA(ExtraiNovoAtributo(VOL_TabelasRMS[0], 'RMS')), NormalizaDadosCOLUNA(ExtraiNovoAtributo(VOL_TabelasRMS[1], 'RMS'))
novoSSC1, novoSSC2 = NormalizaDadosCOLUNA(ExtraiNovoAtributo(VOL_TabelasSSC[0], 'SSC')), NormalizaDadosCOLUNA(ExtraiNovoAtributo(VOL_TabelasSSC[1], 'SSC'))
novoZC1, novoZC2 = NormalizaDadosCOLUNA(ExtraiNovoAtributo(VOL_TabelasZC[0], 'ZC')), NormalizaDadosCOLUNA(ExtraiNovoAtributo(VOL_TabelasZC[1], 'ZC'))
novoVAR1, novoVAR2 = NormalizaDadosCOLUNA(ExtraiNovoAtributo(VOL_TabelasVAR[0], 'VAR')), NormalizaDadosCOLUNA(ExtraiNovoAtributo(VOL_TabelasVAR[1], 'VAR'))
novoWL1, novoWL2 = NormalizaDadosCOLUNA(ExtraiNovoAtributo(VOL_TabelasWL[0], 'WL')), NormalizaDadosCOLUNA(ExtraiNovoAtributo(VOL_TabelasWL[1], 'WL'))

novoSSC_WL1 = ConcatAtrib(novoSSC1, novoWL1)
novoSSC_WL2 = ConcatAtrib(novoSSC2, novoWL2)
novoSSC_WL_Total = pd.concat([novoSSC_WL1,novoSSC_WL2], ignore_index=True)

novaTabelaDeAtributos_VOL_1 = ConcatAtrib(novoRMS1, novoSSC1, novoZC1, novoVAR1)
novaTabelaDeAtributos_VOL_2 = ConcatAtrib(novoRMS2, novoSSC2, novoZC2, novoVAR2)

novaTOTALC1 = pd.concat([novaTabelaDeAtributos_VOL_1, novaTabelaDeAtributos_VOL_2], ignore_index=True)

############################################################################################################################################################################
''' TREINAMENTO E TESTE PARA O NOVO CONJUNTO DE ATRIBUTOS'''
############################################################################################################################################################################

Val2 = TreinaValidacaoCruzada(novoSSC_WL1, VOL_Resp1)
Val2.Parametros()
print(Val2.matrizDeConfusao)
print(Val2.tabelaDeClassificacao)

pca2 = PCA_Teste(novoSSC_WL1, VOL_Resp1)
pca2.RealizaPCA()
sinalPCA2 = pca2.final1

Val22 = TreinaValidacaoCruzada(sinalPCA2, VOL_Resp1)
Val22.Parametros()
print(Val22.matrizDeConfusao)
print(Val22.tabelaDeClassificacao)

pca22 = PCA_Teste(novoSSC_WL2, VOL_Resp2)
pca22.RealizaPCA()
sinalPCA22 = pca22.final1

novoSSC_WL1['Resposta'] = RT1
sns.pairplot(novoSSC_WL1,hue='Resposta', aspect=1, size=1.7, palette=dict(Repousar = 'black', Pinçar = 'green', Supinar = 'royalblue', Pronar = 'red',Fechar = 'darkorange', Estender = 'darkmagenta', Flexionar = 'sienna'))
novoSSC_WL2['Resposta'] = RT1
sns.pairplot(novoSSC_WL2,hue='Resposta', aspect=1, size=1.7, palette=dict(Repousar = 'black', Pinçar = 'green', Supinar = 'royalblue', Pronar = 'red',Fechar = 'darkorange', Estender = 'darkmagenta', Flexionar = 'sienna'))
plt.show()




#################################################################################################################################################################################
''' OUTRAS FUNÇÕES QUE PODEM SER USADAS '''
#################################################################################################################################################################################


def NormalizaDadosMATRIZ(DataFrame):
    df = DataFrame.copy()
    Max = float(df.max().max())
    for S in df:
        for i,v in enumerate(df[S]):
            df[S][i] = (100.0/Max)*float(v)
    return df
def TreinaSimples(TABELASCarac, TABELASResp):
    X_train, X_test, y_train, y_test = train_test_split(TABELASCarac, np.ravel(TABELASResp), test_size=0.25)
    Modelo = SVC()
    Modelo.fit(X_train, y_train)
    Predict = Modelo.predict(X_test)
    print(confusion_matrix(y_test,Predict))
    print(classification_report(y_test,Predict))
def TreinaMelhorParametro(TABELASCarac, TABELASResp):
    X_train, X_test, y_train, y_test = train_test_split(TABELASCarac, np.ravel(TABELASResp), test_size=0.25, stratify=np.ravel(TABELASResp))
    DicParametros =  {'C':[0.1,1,10,100,1000,10000, 100000, 10000000],'gamma':[100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]}
    grid_cv = GridSearchCV(SVC(),DicParametros,refit=True,verbose=2)
    grid_cv.fit(X_train,y_train)
    pred = grid_cv.predict(X_test)
    print(grid_cv.best_params_)
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred))
    print(grid_cv.best_params_)
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set.iloc[:,0:0].min() - 1, stop = X_set.iloc[:,0:0].max() + 1, step = 0.01),
                         np.arange(start = X_set.iloc[:,0:1].min() - 1, stop = X_set.iloc[:,0:1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.2, cmap = ListedColormap(('red', 'green', 'blue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
    plt.title('Logistic Regression (Test set)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()
def homomorphic_envelope(x, fs=1000, f_LPF=8, order=3):
    """
    Computes the homomorphic envelope of x

    Args:
        x : array
        fs : float
            Sampling frequency. Defaults to 1000 Hz
        f_LPF : float
            Lowpass frequency, has to be f_LPF < fs/2. Defaults to 8 Hz
    Returns:
        time : numpy array
    """
    b, a = signal.butter(order, 2 * f_LPF / fs, 'low')
    he = np.exp(signal.filtfilt(b, a, np.log(np.abs(hilbert(x)))))
    return he 
def calc_envelope(x, ind=range(760000)):
    x_abs = np.abs(x)
    loc = np.where(np.diff(np.sign(np.diff(x_abs))) < 0)[0] + 1
    peak = x_abs[loc]
    envelope = np.interp(ind, loc, peak)
    return envelope




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

X = np.array(sinalPCA1.iloc[:, :2]) 
y = VOL.RespEmOrdem_Num[:280].copy()

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('CP 1')
plt.ylabel('CP 2')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(VOL_Ordem_Resp)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("Três primeiras CP")
ax.set_xlabel("CP 1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("CP 2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("CP 3")
ax.w_zaxis.set_ticklabels([])

plt.show()




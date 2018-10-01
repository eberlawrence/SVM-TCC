import numpy as np
import pandas as pd

class SinalVoluntario():

    def __init__(self, nome):
        self.nome = nome

    def CarregaDados(self):
        L_df = []
        for i in range(2):
            for j in range(4):
                dados = pd.read_table('DataSet\\'+self.nome+str(i+1)+str(j+1)+'-Final.txt', sep=';', header=None)
                df = pd.DataFrame()
                df['CH1'], df['CH2'], df['CH3'], df['CH4'], df['Trigger'] = dados[0], dados[1], dados[2], dados[3], dados[4]
                L_df.append(df)
        return (L_df[0],L_df[1],L_df[2],L_df[3],L_df[4],L_df[5],L_df[6],L_df[7]) # Retorna um DATAFRAME com os dados do .txt

    def CarregaRespostas(self):
        '''
        Retorna um SERIES com a sequencia de movimentos armazenada em um .txt 
        Os movimentos são:
        0 -> Repousar (Sempre entre contrações)
        1 -> Supinar
        2 -> Pronar
        3 -> Pinçar
        4 -> Fechar
        5 -> Estender
        6 -> Flexionar
        O SERIES retornado possui todas as respostas das 8 coletas - Total de 480 respostas [0, ..., 479]
        '''
        listaresposta = []
        self.resposta = []
        lista = [0]*40
        for i in range(2):
            count = 0
            for j in range(4):
                resposta = np.array(pd.read_table('DataSet\\'+self.nome+str(i+1)+str(j+1)+'-Resposta.txt', header=None))
                for v in resposta:
                    if v == 1:
                        listaresposta.append('Supinar')
                    if v == 2:
                        listaresposta.append('Pronar')
                    if v == 3:
                        listaresposta.append('Pinçar')
                    if v == 4:
                        listaresposta.append('Fechar')
                    if v == 5:
                        listaresposta.append('Estender')
                    if v == 6:
                        listaresposta.append('Flexionar')
                self.resposta.extend(np.ravel(resposta))
            self.resposta.extend(lista)
            while count < 40:
                listaresposta.append('Repousar')
                count += 1
                
        TargetEMG = pd.DataFrame()
        TargetEMG['Resposta'] = listaresposta
        TargetEMG['R'] = self.resposta
        RespEmOrdem1 = TargetEMG[:int(len(TargetEMG)/2)].sort_values('Resposta').reset_index(drop=True).copy()
        RespEmOrdem2 = TargetEMG[int(len(TargetEMG)/2):].sort_values('Resposta').reset_index(drop=True).copy()
        self.RespEmOrdem = (pd.concat([RespEmOrdem1,RespEmOrdem2], ignore_index=True)).iloc[:,:1]
        self.RespEmOrdem_Num = np.ravel((pd.concat([RespEmOrdem1,RespEmOrdem2], ignore_index=True)).iloc[:,1:])
        return TargetEMG.iloc[:,:1]




import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class PCA_Teste:
    '''
    X -> Tabela com as características
    y -> Tabela com as respostas
    '''
    def __init__(self, Valor_X, Valor_y):
        self.X = np.array(Valor_X)
        self.y = np.array(Valor_y.copy())

    def RealizaPCA(self, n=.95):
        '''
        n -> Se (float) > 1, Porcentagem mínima requisitada da acurácia da classificação, pega a menor quantidade possível de Componentes principais.
             Se (int) >= 1, Pega as n primeiras Componentes principais (com maior variancia).
        '''
        self.pca = PCA(n)    
        principalComponents = self.pca.fit_transform(self.X)
        L = []
        for v in range(self.pca.n_components_):
            L.append('CP_CH'+str(v+1))
        self.final1 = pd.DataFrame(data = principalComponents, columns = L)  





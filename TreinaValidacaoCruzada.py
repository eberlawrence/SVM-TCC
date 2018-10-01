import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_val_predict, StratifiedKFold, GroupKFold

class TreinaValidacaoCruzada():
    def __init__(self, TABELASCarac, TABELASResp):
        self.TCarac = TABELASCarac
        self.TResp = TABELASResp

    def Parametros(self, splits=10, random=True, mostraDivisao=False, plota_MC=False, group=False):
        R = np.ravel(self.TResp).copy()
        DicParametros = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001]}
        i = 1
        L1 = [1,2,3,4,5,6,7,8,9,10]
        L2 = []
        L2 = L1.copy()
        L3 = []
        while i < 56:
            L1.extend(L2)
            i += 1
        if group == True: 
            L3 = L1.copy()
            kf = GroupKFold(n_splits=splits)
            kfTenFold = kf.split(self.TCarac, R, L3)            
        if group == False:
            L3 = L1[:280].copy()
            kf = StratifiedKFold(n_splits=splits, shuffle=random)
            kfTenFold = kf.split(self.TCarac, R, L3)    
        self.grid_cv = GridSearchCV(SVC(),DicParametros, cv=kfTenFold)
        self.grid_cv.fit(self.TCarac,R)
        self.cross_Score = cross_val_score(self.grid_cv.best_estimator_, self.TCarac, R, scoring='accuracy', cv=kf.split(self.TCarac, R, L3))
        self.mediaAcuracia = self.cross_Score.mean()
        self.cross_Predict = cross_val_predict(self.grid_cv.best_estimator_, self.TCarac, R, cv=kf.split(self.TCarac, R, L3))
        self.matrizDeConfusao = confusion_matrix(R, self.cross_Predict)
        self.tabelaDeClassificacao = classification_report(R, self.cross_Predict)
        if plota_MC == True:
            self.plotaMatriz = sns.heatmap(confusion_matrix(R,self.cross_Predict), cmap='hot', annot=True, fmt="d", linecolor='gray',linewidths=.5)
        if mostraDivisao == True:
            for train_index, test_index in kf.split(self.TCarac, R, L3):
                print("\n\nTRAIN:", train_index, "\nTEST:", test_index)
                X_train2, X_test2 = self.TCarac.index[train_index], self.TCarac.index[test_index]
                y_train2, y_test2 = self.TResp.index[train_index], self.TResp.index[test_index]
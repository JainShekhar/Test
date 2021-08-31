@author: shekharjain
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#temperature from 250 to 500 K as 50 points
Tpoints = 50
T = np.linspace(250,500,Tpoints)
#pressure from 1 to 100 bar as 50 points
Ppoints = 50
P = np.linspace(1,100,Ppoints)

n = Tpoints*Ppoints
T_grid, P_grid = np.meshgrid(T,P)
T_grid = T_grid.reshape(n,1)
P_grid = P_grid.reshape(n,1)
TP_grid = np.hstack((T_grid,P_grid))

#complete feature matrix for ethane
grp_CH3ne = np.ones_like(T_grid)*2
grp_CH2c = np.ones_like(T_grid)*0
grp_grid = np.hstack((grp_CH3ne,grp_CH2c))
#X = np.hstack((grp_grid,TP_grid))
X = TP_grid

#target matrix for ethane: 0: for v, 1: for l, 2: for supercritical
VP_Constants = {'ethane':[51.857,-2598.7,-5.1283,0.000014913,2]}
T_Constants = {'ethane': [305.32]}
P_Constants = {'ethane': [48.72]}
y = np.zeros(T_grid.shape[0])
for i in range(len(y)):
    if (X[i,0] > T_Constants['ethane'][0]):
        y[i] = 2
    else:
        lnVP_Pa = VP_Constants['ethane'][0] + VP_Constants['ethane'][1]/X[i,0] + VP_Constants['ethane'][2]*np.log(X[i,0]) + VP_Constants['ethane'][3]*X[i,0]**VP_Constants['ethane'][4]
        VP = np.exp(lnVP_Pa)/100000
        if (X[i,1] < VP):
            y[i] = 0
        else:
            y[i] = 1

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
ppn = SVC(kernel='rbf',gamma=0.01,C=100,random_state=1)
ppn.fit(X_train,y_train)
y_pred = ppn.predict(X_train)
miscls = np.sum(y_pred != y_train)
print('Misclassifications in training set is %d out of %d' %(miscls,len(y_train)))

y_pred = ppn.predict(X_test)
miscls = np.sum(y_pred != y_test)
print('Misclassifications in test set is %d out of %d' %(miscls,len(y_test)))




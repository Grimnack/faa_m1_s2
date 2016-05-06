#!/usr/bin/env python
#-*- coding: utf-8 -*-

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

##### DONNEES ##### 

def lecture(pathname) :
    '''
    renvoie le couple (matrix, vecteur) 
    '''
    dataset = np.loadtxt(pathname, delimiter=",")
    X = dataset[:,0:-1]
    Y = dataset[:,-1]
    taille = len(Y)
    Xlearn = X[:(taille//10)*9]
    Ylearn = Y[:(taille//10)*9]
    Xtest = X[(taille//10)*9+1:]
    Ytest = Y[(taille//10)*9+1:]
    return (Xlearn,Ylearn,Xtest,Ytest)

(diabeteX,diabeteY,diabeteXtest,diabeteYtest) = lecture('pima-indians-diabetes.data')
(spamX,spamY,spamXtest,spamYtest) = lecture('spambase.data')

print "###### DecisionTreeClassifier ######"

for prof in range(1,10) :
    TreeCL = tree.DecisionTreeClassifier(max_depth=prof)
    TreeCL = TreeCL.fit(diabeteX,diabeteY)
    nbError = 0
    tab = TreeCL.predict(diabeteXtest)
    for i in range(len(diabeteXtest)) :
        if tab[i] != diabeteYtest[i] :
            nbError = nbError + 1
    tauxErr = float(nbError)/float(len(diabeteYtest))
    print "pour une profondeur de ",prof," voici le taux d erreur ",tauxErr

print "###### RandomForestClassifier ######"

for prof in range(1,10) :
    RandomForestCL = RandomForestClassifier(n_estimators=10,max_depth=prof)
    RandomForestCL = RandomForestCL.fit(diabeteX,diabeteY)
    nbError = 0
    tab = RandomForestCL.predict(diabeteXtest)
    for i in range(len(diabeteXtest)) :
        if tab[i] != diabeteYtest[i] :
            nbError = nbError + 1
        tauxErr = float(nbError)/float(len(diabeteYtest))
    print "pour une profondeur de ",prof," voici le taux d erreur ",tauxErr

print "###### ExtraTreesClassifier ######"

for prof in range(1,20) :
    ExtraTreeCL = ExtraTreesClassifier(n_estimators=10,max_depth=prof)
    ExtraTreeCL = ExtraTreeCL.fit(diabeteX,diabeteY)
    nbError = 0
    tab = ExtraTreeCL.predict(diabeteXtest)
    for i in range(len(diabeteXtest)) :
        if tab[i] != diabeteYtest[i] :
            nbError = nbError + 1
        tauxErr = float(nbError)/float(len(diabeteYtest))
    print "pour une profondeur de ",prof," voici le taux d erreur ",tauxErr


print "####################################"
print "###### Influence du nombre d arbre ######"
print "####################################" 
print "###### RandomForestClassifier ######"

lesMoyennes = []
lesVariances = []
for arbre in range(5,20) :
    lesTaux = []
    for stat in range(30) :
        RandomForestCL = RandomForestClassifier(n_estimators=arbre,max_depth=5)
        RandomForestCL = RandomForestCL.fit(diabeteX,diabeteY)
        nbError = 0
        tab = RandomForestCL.predict(diabeteXtest)
        for i in range(len(diabeteXtest)) :
            if tab[i] != diabeteYtest[i] :
                nbError = nbError + 1
            tauxErr = float(nbError)/float(len(diabeteYtest))
            lesTaux.append(tauxErr)
    lesMoyennes.append((arbre,np.mean(lesTaux)))
    lesVariances.append((arbre,np.var(lesTaux)))

for (arbre,moyenne) in lesMoyennes :
    print "La MOYENNE des erreur pour un nombre d arbre de  ",arbre," = ",moyenne
    
for (arbre,variance) in lesVariances :
    print "La VARIANCE des erreur pour un nombre d arbre de ",arbre," = ",variance


print "###### ExtraTreesClassifier ######"

lesMoyennes = []
lesVariances = []
for arbre in range(5,20) :
    lesTaux = []
    for stat in range(30) :
        ExtraTreeCL = ExtraTreesClassifier(n_estimators=arbre,max_depth=5)
        ExtraTreeCL = ExtraTreeCL.fit(diabeteX,diabeteY)
        nbError = 0
        tab = ExtraTreeCL.predict(diabeteXtest)
        for i in range(len(diabeteXtest)) :
            if tab[i] != diabeteYtest[i] :
                nbError = nbError + 1
            tauxErr = float(nbError)/float(len(diabeteYtest))
            lesTaux.append(tauxErr)
    lesMoyennes.append((arbre,np.mean(lesTaux)))
    lesVariances.append((arbre,np.var(lesTaux)))    

for (arbre,moyenne) in lesMoyennes :
    print "La MOYENNE des erreur pour un nombre d arbre de  ",arbre," = ",moyenne
    
for (arbre,variance) in lesVariances :
    print "La VARIANCE des erreur pour un nombre d arbre de ",arbre," = ",variance
